// kq.cpp: the museglimmer silo (silo.h) - Muse Glimmer's K-quant matrices (Q4_K, Q6_K: the Q4_K_M and Q6_K files)
// repacked at load, losslessly, a row at a time, and their decode products on XMX. A repacked row, by groups of 64 (two
// blocks of 32):
//   its 4-bit parts [n_in / 2]: a group's 32 bytes, byte i element i's low 4 bits | element 32 + i's << 4 (Q4_K's own
//     order) - a dword of them masked is 4 consecutive elements of the first block, shifted the second's;
//   Q6_K: its high 2 bits [n_in / 4]: a group's 4 dwords, dword p's byte q element 8p + q's of the first block in bits
//     0-1, of the second in 2-3, element 8p + 4 + q's in 4-5 and 6-7 - each a mask and a shift from its 4-bit parts;
//   its 256-blocks' scales: Q4_K d, dmin, then the 8 scales and the 8 minimums unpacked from their 6 bits (20 bytes);
//     Q6_K its 16 int8 scales and d (18 bytes); padded to 16 bytes.
// Rows are 16-byte aligned, so a matrix's rows r0.. are a view of it.
//
// The product: the weights are the B operand (int8 DPAS 8 x 32 x 16): a lane a row of a 16-row tile, its group's 32
// bytes (and 16 of high bits) expanded by masks and shifts into dwords, written to local memory in the packed layout;
// the decode rows the A operand (8 tokens a tile, xq.hpp's token-major quants). The int32 sums land a row a lane, so
// its scales are the lane's own: x its d sc and the token's scale, less its dmin m x the token's block sum (Q4_K); Q6_K's
// scales are a 16's: a block is two tiles, the other half zero. A token's arithmetic is the same however many tokens a
// pass has (a verify row's sums its one-row pass's).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"
#include "xq.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdlib>

namespace {
using half = sycl::half;
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int SG = 16;
constexpr int Q4K = 12, Q6K = 14;

inline int64_t pad16(int64_t n) { return (n + 15) / 16 * 16; }
inline int64_t aux_at(int ty, int64_t n) { return ty == Q4K ? n / 2 : n / 2 + n / 4; }
inline int64_t aux_blk(int ty) { return ty == Q4K ? 20 : 18; }
inline int64_t row_bytes(int ty, int64_t n) { return aux_at(ty, n) + pad16(n / 256 * aux_blk(ty)); }

template <typename T>
inline auto gpm(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto lpm(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

// Q4_K's 6-bit scale and minimum j (0..7) of a block's 12 bytes (ggml's get_scale_min_k4)
inline void scale_min(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// a Q6_K dword's 4 bytes of 6 bits less 32, as int8 (no borrow between the bytes)
inline uint32_t less32(uint32_t v) { return ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u; }

// RT tiles (16 RT rows) a sub-group, SK sub-groups a work-group splitting the groups (KG consecutive ones a step: their
// loads issued first), TT tiles of 8 tokens
template <int TY, int RT, int SK, int TT, int KG>
void kmmvq(sycl::queue& q, const uint8_t* w, const int8_t* xa, const float* xs, const float* xm, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    const int64_t ng = n_in / 64;
    const int64_t rb = row_bytes(TY, n_in), aux = aux_at(TY, n_in);
    constexpr int ROWS = RT * 16;
    // its B tiles a group (512 bytes, 128 dwords each): Q4_K a block's; Q6_K a block's halves
    constexpr int NT = TY == Q4K ? 2 : 4;
    // Q4_K's tiles in two sets taken in turn (one barrier a step); Q6_K's, twice the size, in one (local memory bounds
    // the work-groups a core holds)
    constexpr int NS = TY == Q4K ? 2 : 1;
    const int64_t groups = (n_out + ROWS - 1) / ROWS;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(SK > 1 ? SK * RT * 16 * 16 : 1), h);
        sycl::local_accessor<uint32_t, 1> tiles(sycl::range<1>(SK * NS * NT * 128), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SK * SG), sycl::range<1>(SK * SG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int64_t row0 = it.get_group(0) * ROWS;
            uint32_t* tl0 = tiles.get_multi_ptr<sycl::access::decorated::no>().get() + sgi * NS * NT * 128;
            int par = 0;
            // the products' sums, and (Q4_K) the minimums' apart
            float acc[RT][TT][8], accm[RT][TT][8];
#pragma unroll
            for (int t = 0; t < RT; ++t)
#pragma unroll
                for (int u = 0; u < TT; ++u)
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][u][e] = accm[t][u][e] = 0.0f;
            for (int64_t g0 = (int64_t) sgi * KG; g0 < ng; g0 += SK * KG) {
              // the step's weights: KG groups of each tile's rows (a lane its row's)
              sycl::vec<uint32_t, 4> wn[KG][RT][2], wh[KG][RT];
#pragma unroll
              for (int kg = 0; kg < KG; ++kg)
#pragma unroll
                for (int t = 0; t < RT; ++t) {
                    const int64_t g = sycl::min(g0 + kg, ng - 1);
                    const int64_t row = row0 + t * 16 + lane;
                    const uint8_t* wr = w + (row < n_out ? row : n_out - 1) * rb;
                    wn[kg][t][0] = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wr + g * 32);
                    wn[kg][t][1] = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wr + g * 32 + 16);
                    if constexpr (TY == Q6K) wh[kg][t] = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(wr + n_in / 2 + g * 16);
                }
#pragma unroll
              for (int kg = 0; kg < KG; ++kg) {
                const int64_t g = g0 + kg;
                if (g >= ng) break;
                const int64_t k0 = 2 * g, blk = g / 4, j = g % 4;
                // the tokens' quants of the group's two blocks, their scales (and block sums)
                jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> am[2][TT];
#pragma unroll
                for (int b = 0; b < 2; ++b)
#pragma unroll
                    for (int u = 0; u < TT; ++u) jm::joint_matrix_load(sg, am[b][u], gpm(xa + (k0 + b) * 512 + u * 256), (size_t) 32);
                // (8 tokens' each a load: [block][16 tokens])
                sycl::vec<float, 8> ts[2][TT], tm[2][TT];
#pragma unroll
                for (int b = 0; b < 2; ++b)
#pragma unroll
                    for (int u = 0; u < TT; ++u) {
                        ts[b][u] = *reinterpret_cast<const sycl::vec<float, 8>*>(xs + (k0 + b) * 16 + u * 8);
                        if constexpr (TY == Q4K) tm[b][u] = *reinterpret_cast<const sycl::vec<float, 8>*>(xm + (k0 + b) * 16 + u * 8);
                    }
#pragma unroll
                for (int t = 0; t < RT; ++t) {
                    // two sets of tiles in turn, a set rewritten only after the barrier past its reads; one set: past
                    // its last reads
                    uint32_t* tl = tl0 + par * NT * 128;
                    if constexpr (NS == 2) par ^= 1;
                    else sycl::group_barrier(sg);
                    const int64_t row = row0 + t * 16 + lane;
                    const uint8_t* wr = w + (row < n_out ? row : n_out - 1) * rb;
                    const sycl::vec<uint32_t, 4> n0 = wn[kg][t][0], n1 = wn[kg][t][1];
                    uint32_t nd[8] = {n0[0], n0[1], n0[2], n0[3], n1[0], n1[1], n1[2], n1[3]};
                    // its scales: the lane's own row's
                    float sl[4], ml[2] = {0.0f, 0.0f};
                    if constexpr (TY == Q4K) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            tl[i * 16 + lane] = nd[i] & 0x0F0F0F0Fu;
                            tl[128 + i * 16 + lane] = (nd[i] >> 4) & 0x0F0F0F0Fu;
                        }
                        const uint8_t* a = wr + aux + blk * 20;
                        const float d = (float) *reinterpret_cast<const half*>(a), dm = (float) *reinterpret_cast<const half*>(a + 2);
                        sl[0] = d * (float) a[4 + 2 * j];
                        sl[1] = d * (float) a[4 + 2 * j + 1];
                        ml[0] = dm * (float) a[12 + 2 * j];
                        ml[1] = dm * (float) a[12 + 2 * j + 1];
                    } else {
                        const sycl::vec<uint32_t, 4> hv = wh[kg][t];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const uint32_t hp = hv[i / 2];
                            const uint32_t h0 = (i % 2 == 0 ? hp << 4 : hp) & 0x30303030u;
                            const uint32_t h1 = (i % 2 == 0 ? hp << 2 : hp >> 2) & 0x30303030u;
                            const uint32_t v0 = less32((nd[i] & 0x0F0F0F0Fu) | h0), v1 = less32(((nd[i] >> 4) & 0x0F0F0F0Fu) | h1);
                            // tile 2 b + half: block b's dwords of that half (K 16 half ..), the other half zero
                            const bool lo = i < 4;
                            tl[0 * 128 + i * 16 + lane] = lo ? v0 : 0u;
                            tl[1 * 128 + i * 16 + lane] = lo ? 0u : v0;
                            tl[2 * 128 + i * 16 + lane] = lo ? v1 : 0u;
                            tl[3 * 128 + i * 16 + lane] = lo ? 0u : v1;
                        }
                        const uint8_t* a = wr + aux + blk * 18;
                        const float d = (float) *reinterpret_cast<const half*>(a + 16);
#pragma unroll
                        for (int u = 0; u < 4; ++u) sl[u] = d * (float) (int8_t) a[4 * j + u];
                    }
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int bt = 0; bt < NT; ++bt) {
                        const int b = TY == Q4K ? bt : bt / 2;
                        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> bm;
                        jm::joint_matrix_load(sg, bm, lpm(reinterpret_cast<int8_t*>(tl + bt * 128)), (size_t) 64);
#pragma unroll
                        for (int u = 0; u < TT; ++u) {
                            jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> cm;
                            jm::joint_matrix_fill(sg, cm, 0);
                            jm::joint_matrix_mad(sg, cm, am[b][u], bm, cm);
                            int e = 0;
                            if constexpr (TY == Q4K) {
                                const float s = sl[b];
                                jm::joint_matrix_apply(sg, cm, [&](int32_t& v) {
                                    acc[t][u][e] = sycl::fma((float) v, s * ts[b][u][e], acc[t][u][e]);
                                    ++e;
                                });
#pragma unroll
                                for (int e2 = 0; e2 < 8; ++e2) accm[t][u][e2] = sycl::fma(ml[b], tm[b][u][e2], accm[t][u][e2]);
                            } else {
                                const float s = sl[bt];
                                jm::joint_matrix_apply(sg, cm, [&](int32_t& v) {
                                    acc[t][u][e] = sycl::fma((float) v, s * ts[b][u][e], acc[t][u][e]);
                                    ++e;
                                });
                            }
                        }
                    }
                }
              }
            }
            // element e of tile u: token 8u + e, the lane's row
            if constexpr (SK == 1) {
#pragma unroll
                for (int t = 0; t < RT; ++t) {
                    const int64_t row = row0 + t * 16 + lane;
                    if (row >= n_out) continue;
#pragma unroll
                    for (int u = 0; u < TT; ++u)
#pragma unroll
                        for (int e = 0; e < 8; ++e)
                            if (u * 8 + e < ncols) y[(u * 8 + e) * n_out + row] = acc[t][u][e] - accm[t][u][e];
                }
            } else {
                float* rd = red.get_multi_ptr<sycl::access::decorated::no>().get();
#pragma unroll
                for (int t = 0; t < RT; ++t)
#pragma unroll
                    for (int u = 0; u < TT; ++u)
#pragma unroll
                        for (int e = 0; e < 8; ++e) rd[((sgi * RT + t) * 16 + u * 8 + e) * 16 + lane] = acc[t][u][e] - accm[t][u][e];
                sycl::group_barrier(it.get_group());
                for (int i = (int) it.get_local_id(0); i < RT * 16 * 16; i += SK * SG) {
                    const int ln = i % 16, tk = (i / 16) % 16, t = i / 256;
                    const int64_t row = row0 + t * 16 + ln;
                    if (tk >= ncols || row >= n_out) continue;
                    float v = 0.0f;
                    for (int s2 = 0; s2 < SK; ++s2) v += rd[((s2 * RT + t) * 16 + tk) * 16 + ln];
                    y[tk * n_out + row] = v;
                }
            }
        });
    });
}

template <int TY, int TT>
void kmmvq_shape(int shape, sycl::queue& q, const uint8_t* w, const int8_t* xa, const float* xs, const float* xm, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    // by default a work-group's sub-groups split the row's groups evenly, 2 a step: 16, 13 (Muse's 6656 and 19968: 104
    // and 312 groups) or 8 of them
    if (shape == 0) {
        const int64_t ng = n_in / 64;
        shape = ng % 32 == 0 ? 5 : ng % 26 == 0 ? 6 : ng % 16 == 0 ? 1 : 5;
    }
    switch (shape) {
        case 1: kmmvq<TY, 1, 8, TT, 2>(q, w, xa, xs, xm, y, n_in, n_out, ncols); break;
        case 2: kmmvq<TY, 1, 4, TT, 2>(q, w, xa, xs, xm, y, n_in, n_out, ncols); break;
        case 3: kmmvq<TY, 1, 8, TT, 4>(q, w, xa, xs, xm, y, n_in, n_out, ncols); break;
        case 4: kmmvq<TY, 2, 8, TT, 2>(q, w, xa, xs, xm, y, n_in, n_out, ncols); break;
        case 6: kmmvq<TY, 1, 13, TT, 2>(q, w, xa, xs, xm, y, n_in, n_out, ncols); break;
        case 7: kmmvq<TY, 1, 13, TT, 4>(q, w, xa, xs, xm, y, n_in, n_out, ncols); break;
        default: kmmvq<TY, 1, 16, TT, 2>(q, w, xa, xs, xm, y, n_in, n_out, ncols); break;
    }
}

}  // namespace

extern "C" {

int ns_mg_kr_supported(int type, int64_t n_in) { return (type == Q4K || type == Q6K) && n_in % 256 == 0; }
size_t ns_mg_kr_row_bytes(int type, int64_t n_in) { return (size_t) row_bytes(type, n_in); }

int ns_mg_kr_pack(ns_gpu* g, int type, const void* raw, int64_t rows, int64_t n_in, void* dst) {
    NS_TRY
    if (!ns_mg_kr_supported(type, n_in)) return ns_fail("ns_mg_kr_pack: Q4_K or Q6_K, n_in a multiple of 256");
    const uint8_t* src = (const uint8_t*) raw;
    uint8_t* o = (uint8_t*) dst;
    const int64_t nbr = n_in / 256, rb = row_bytes(type, n_in), aux = aux_at(type, n_in);
    // a work-item a 256-block
    if (type == Q4K) {
        g->q.parallel_for(sycl::range<1>((size_t) (rows * nbr)), [=](sycl::id<1> id) {
            const int64_t row = id[0] / nbr, b = id[0] % nbr;
            const uint8_t* s = src + (row * nbr + b) * 144;
            uint8_t* orow = o + row * rb;
            for (int i = 0; i < 128; ++i) orow[b * 128 + i] = s[16 + i];
            uint8_t* a = orow + aux + b * 20;
            for (int i = 0; i < 4; ++i) a[i] = s[i];
            for (int jj = 0; jj < 8; ++jj) {
                int sc, m;
                scale_min(jj, s + 4, sc, m);
                a[4 + jj] = (uint8_t) sc;
                a[12 + jj] = (uint8_t) m;
            }
        });
    } else {
        g->q.parallel_for(sycl::range<1>((size_t) (rows * nbr)), [=](sycl::id<1> id) {
            const int64_t row = id[0] / nbr, b = id[0] % nbr;
            const uint8_t* s = src + (row * nbr + b) * 210;
            const uint8_t *ql = s, *qh = s + 128;
            uint8_t qv[256];
            // ggml's dequantize_row_q6_K order: element e's 6 bits (before the -32)
            for (int n = 0; n < 2; ++n)
                for (int l = 0; l < 32; ++l) {
                    const uint8_t* L = ql + 64 * n;
                    const uint8_t H = qh[32 * n + l];
                    qv[128 * n + l] = (uint8_t) ((L[l] & 0xF) | ((H & 3) << 4));
                    qv[128 * n + l + 32] = (uint8_t) ((L[l + 32] & 0xF) | (((H >> 2) & 3) << 4));
                    qv[128 * n + l + 64] = (uint8_t) ((L[l] >> 4) | (((H >> 4) & 3) << 4));
                    qv[128 * n + l + 96] = (uint8_t) ((L[l + 32] >> 4) | (((H >> 6) & 3) << 4));
                }
            uint8_t* orow = o + row * rb;
            for (int jj = 0; jj < 4; ++jj) {
                const uint8_t* q = qv + 64 * jj;
                const int64_t gi = b * 4 + jj;
                for (int i = 0; i < 32; ++i) orow[gi * 32 + i] = (uint8_t) ((q[i] & 15) | ((q[32 + i] & 15) << 4));
                // dword p byte qq: element 8p + qq (bits 0-1: the first block's, 2-3 the second's), 8p + 4 + qq (4-7)
                for (int p = 0; p < 4; ++p)
                    for (int qq = 0; qq < 4; ++qq) {
                        const int e0 = 8 * p + qq, e1 = 8 * p + 4 + qq;
                        orow[n_in / 2 + gi * 16 + p * 4 + qq] =
                            (uint8_t) ((q[e0] >> 4) | ((q[32 + e0] >> 4) << 2) | ((q[e1] >> 4) << 4) | ((q[32 + e1] >> 4) << 6));
                    }
            }
            for (int i = 0; i < 18; ++i) orow[aux + b * 18 + i] = s[192 + i];
        });
    }
    return 0;
    NS_CATCH
}

int ns_mg_kr_mmvq(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (ncols < 1 || ncols > mgxq::XCOLS || !ns_mg_kr_supported(type, n_in) || n_out % 16 != 0)
        return ns_fail("ns_mg_kr_mmvq: Q4_K or Q6_K, 1..16 columns, n_in a multiple of 256, n_out of 16");
    const int8_t* xa = (const int8_t*) xq + mgxq::rowq(n_in);
    const float* xs = (const float*) ((const uint8_t*) xq + mgxq::scales(n_in));
    const float* xm = (const float*) ((const uint8_t*) xq + mgxq::sums(n_in));
    // NS_MG_KQ_SHAPE: tiles a sub-group, sub-groups a work-group, groups a step (0: by the width; 1: 1, 8, 2; 2: 1, 4, 2;
    // 3: 1, 8, 4; 4: 2, 8, 2; 5: 1, 16, 2; 6: 1, 13, 2; 7: 1, 13, 4)
    static const int shape = [] { const char* v = std::getenv("NS_MG_KQ_SHAPE"); return v ? std::atoi(v) : 0; }();
    const uint8_t* wb = (const uint8_t*) w;
    if (type == Q4K) {
        if (ncols <= 8) kmmvq_shape<Q4K, 1>(shape, g->q, wb, xa, xs, xm, y, n_in, n_out, ncols);
        else kmmvq_shape<Q4K, 2>(shape, g->q, wb, xa, xs, xm, y, n_in, n_out, ncols);
    } else {
        if (ncols <= 8) kmmvq_shape<Q6K, 1>(shape, g->q, wb, xa, xs, xm, y, n_in, n_out, ncols);
        else kmmvq_shape<Q6K, 2>(shape, g->q, wb, xa, xs, xm, y, n_in, n_out, ncols);
    }
    return 0;
    NS_CATCH
}

int ns_mg_kr_f16(ns_gpu* g, int type, const void* w, int64_t rows, int64_t n_in, uint16_t* dst) {
    NS_TRY
    if (!ns_mg_kr_supported(type, n_in)) return ns_fail("ns_mg_kr_f16: Q4_K or Q6_K, n_in a multiple of 256");
    const uint8_t* wb = (const uint8_t*) w;
    half* o = (half*) dst;
    const int64_t nc = n_in / 16, rb = row_bytes(type, n_in), aux = aux_at(type, n_in);
    // a work-item 16 consecutive outputs (one scale: block b's half h of group gi), consecutive ones a row's next
    if (type == Q4K) {
        g->q.parallel_for(sycl::range<1>((size_t) (rows * nc)), [=](sycl::id<1> id) {
            const int64_t row = id[0] / nc, c = id[0] % nc, gi = c / 4, q = c % 4, b = q / 2, hh = q % 2, blk = gi / 4, j = gi % 4;
            const uint8_t* wr = wb + row * rb;
            const uint8_t* a = wr + aux + blk * 20;
            const float d = (float) *reinterpret_cast<const half*>(a), dm = (float) *reinterpret_cast<const half*>(a + 2);
            const float sc = d * (float) a[4 + 2 * j + b], mn = dm * (float) a[12 + 2 * j + b];
            const sycl::vec<uint8_t, 16> v = *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(wr + gi * 32 + hh * 16);
            sycl::vec<half, 16> out;
#pragma unroll
            for (int i = 0; i < 16; ++i) out[i] = (half) (sc * (float) (b ? v[i] >> 4 : v[i] & 15) - mn);
            *reinterpret_cast<sycl::vec<half, 16>*>(o + row * n_in + c * 16) = out;
        });
    } else {
        g->q.parallel_for(sycl::range<1>((size_t) (rows * nc)), [=](sycl::id<1> id) {
            const int64_t row = id[0] / nc, c = id[0] % nc, gi = c / 4, q = c % 4, b = q / 2, hh = q % 2, blk = gi / 4, j = gi % 4;
            const uint8_t* wr = wb + row * rb;
            const uint8_t* a = wr + aux + blk * 18;
            const float sc = (float) *reinterpret_cast<const half*>(a + 16) * (float) (int8_t) a[4 * j + 2 * b + hh];
            const sycl::vec<uint8_t, 16> v = *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(wr + gi * 32 + hh * 16);
            // its high bits: dwords 2 hh, 2 hh + 1 of the group's 4 (elements 16 hh ..)
            const sycl::vec<uint8_t, 8> hv = *reinterpret_cast<const sycl::vec<uint8_t, 8>*>(wr + n_in / 2 + gi * 16 + hh * 8);
            sycl::vec<half, 16> out;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int dw = i / 4, qq = i % 4;   // dword 4 hh + dw of the K order
                const int hb = (hv[(dw / 2) * 4 + qq] >> ((b ? 2 : 0) + (dw % 2 ? 4 : 0))) & 3;
                const int qv = (int) ((b ? v[i] >> 4 : v[i] & 15) | (hb << 4)) - 32;
                out[i] = (half) (sc * (float) qv);
            }
            *reinterpret_cast<sycl::vec<half, 16>*>(o + row * n_in + c * 16) = out;
        });
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
