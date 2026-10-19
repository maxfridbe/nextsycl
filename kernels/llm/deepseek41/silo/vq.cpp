// vq.cpp: the routed experts' vector quantization (ds41.h) and the fp8_32x32 matrices.
//
// A DQV3 payload (YoungAi's vq_fmt.h, ds4-dfm-rs's vq.rs): a 32-byte header, an f16 gain a row, the row's indices as a
// little-endian 12-bit stream (cols / 8 of them, 1.5 bytes each), and at 13 bits a plane of the 13th bits (a byte a
// row's 8 indices). value[r][c] = codebook[idx(r, c / 8)][c % 8] x gain[r]; the codebook is the layer's, E4M3 [2^nbit][8].
// Eight indices are 12 aligned bytes (three words) and 64 columns: the unit every kernel here reads.
//
// An E4M3 byte as an fp16's bits - sign to bit 15, exponent and mantissa shifted 7 up - is the value x 2^-8, the
// subnormals too (fp16's exponent bias 15 against E4M3's 7): one shift, one convert; the 2^8 goes into the gain.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

inline float e4m3_scaled(uint32_t b) {   // the value x 2^-8
    const uint16_t bits = (uint16_t) (((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return (float) sycl::bit_cast<sycl::half>(bits);
}

// the 8 indices of a chunk: 12 bytes at s (4-aligned), the plane's byte pb (13-bit layers; 0 otherwise)
inline void chunk_idx(const uint32_t* s, uint32_t pb, uint32_t idx[8]) {
    const uint64_t lo = (uint64_t) s[0] | ((uint64_t) s[1] << 32);
    const uint32_t hi = s[2];
#pragma unroll
    for (int i = 0; i < 5; ++i) idx[i] = (uint32_t) (lo >> (12 * i)) & 0xFFFu;
    idx[5] = (uint32_t) ((lo >> 60) | ((uint64_t) hi << 4)) & 0xFFFu;
    idx[6] = (hi >> 8) & 0xFFFu;
    idx[7] = (hi >> 20) & 0xFFFu;
#pragma unroll
    for (int i = 0; i < 8; ++i) idx[i] |= ((pb >> i) & 1u) << 12;
}

struct VqMat {
    const sycl::half* gains;
    const uint8_t* stream;
    const uint8_t* plane;
    int64_t mrow, prow;
};
inline VqMat vq_open(const uint8_t* p, int64_t rows, int64_t cols, int64_t nbit) {
    VqMat m;
    m.gains = reinterpret_cast<const sycl::half*>(p + 32);
    m.stream = p + 32 + rows * 2;
    m.mrow = cols / 8 * 12 / 8;
    m.prow = cols / 64;
    m.plane = nbit > 12 ? m.stream + rows * m.mrow : nullptr;
    return m;
}

}  // namespace

namespace {
// an index's 8 weights (x 2^-8): CB 0 from the E4M3 bytes a byte at a time, 1 from the bytes two at a time (a word's
// bytes 0 and 2, then 1 and 3, as fp16 pairs), 2 from the fp16 codebook (cbh)
template <int CB>
inline void vq_weights(const uint8_t* cb, const sycl::half* cbh, uint32_t ix, float wv[8]) {
    if constexpr (CB == 2) {
        const sycl::vec<sycl::half, 8> h = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(cbh + (size_t) ix * 8);
#pragma unroll
        for (int d = 0; d < 8; ++d) wv[d] = (float) h[d];
    } else if constexpr (CB == 1) {
        const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(cb + (size_t) ix * 8);
#pragma unroll
        for (int hw = 0; hw < 2; ++hw) {
            const uint32_t u = hw ? w.y() : w.x();
            const uint32_t e = ((u & 0x00800080u) << 8) | ((u & 0x007F007Fu) << 7);
            const uint32_t o = (u & 0x80008000u) | ((u & 0x7F007F00u) >> 1);
            const sycl::vec<sycl::half, 2> he = sycl::bit_cast<sycl::vec<sycl::half, 2>>(e), ho = sycl::bit_cast<sycl::vec<sycl::half, 2>>(o);
            wv[4 * hw + 0] = (float) he[0];
            wv[4 * hw + 2] = (float) he[1];
            wv[4 * hw + 1] = (float) ho[0];
            wv[4 * hw + 3] = (float) ho[1];
        }
    } else {
        const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(cb + (size_t) ix * 8);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            wv[d] = e4m3_scaled((w.x() >> (8 * d)) & 0xFF);
            wv[4 + d] = e4m3_scaled((w.y() >> (8 * d)) & 0xFF);
        }
    }
}

// vq_mmv's variants 1, 2 and 4: a lane an output row (WG rows a work-group), the activations read alike by all lanes, an
// index's weights from the E4M3 bytes (8 bytes gathered; 4 converts them two at a time) or the fp16 codebook (2: 16
// bytes). 6 experts, one token, gate | up / down in us - B70 12-bit: variant 0 239 / 158, 1 216 / 121, 2 199 / 105,
// 3 230 / 149, 4 191 / 106; 13-bit: 374 / 220, 266 / 138, 264 / 132, 226 / 153, 240 / 122; B65 12-bit: 440 / 279,
// 507 / 245, 452 / 209, 418 / 264, 433 / 210. Which wins depends on the GPU and the shape: the engine times them all
// at load (vq_pick).
template <bool PLANE, int CB>
void vq_mmv_rows(sycl::queue& q, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const uint8_t* cb, const sycl::half* cbh, const float* x, float* y,
                 int64_t rows, int64_t cols, int WG) {
    const int64_t rgs = (rows + WG - 1) / WG, chunks = cols / 64;
    q.parallel_for(sycl::nd_range<1>(entries * rgs * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t gi = it.get_group(0), e = gi / rgs, r = (gi % rgs) * WG + it.get_local_id(0);
        if (r >= rows) return;
        const int32_t* me = meta + e * 10;
        const int nc = me[0];
        const VqMat m = vq_open(reinterpret_cast<const uint8_t*>(ptrs[e]), rows, cols, PLANE ? 13 : 12);
        const uint32_t* srow = reinterpret_cast<const uint32_t*>(m.stream + r * m.mrow);
        const uint8_t* prow = PLANE ? m.plane + r * m.prow : nullptr;
        float acc[8] = {};
        for (int64_t k = 0; k < chunks; ++k) {
            uint32_t idx[8];
            chunk_idx(srow + 3 * k, PLANE ? prow[k] : 0u, idx);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float wv[8];
                vq_weights<CB>(cb, cbh, idx[i], wv);
                for (int c = 0; c < nc; ++c) {
                    const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(x + (int64_t) me[2 + c] * cols + k * 64 + i * 8);
                    const sycl::float4 a = xv[0], b = xv[1];
                    acc[c] += wv[0] * a.x() + wv[1] * a.y() + wv[2] * a.z() + wv[3] * a.w() + wv[4] * b.x() + wv[5] * b.y() + wv[6] * b.z() + wv[7] * b.w();
                }
            }
        }
        const float gn = (float) m.gains[r] * 256.f;
        for (int c = 0; c < nc; ++c) y[(int64_t) (me[1] + c) * rows + r] = acc[c] * gn;
    });
}
}  // namespace

// The decode products by lookup table (variant 5, ns_ds41_vq_lut): a column of x and a run of 8 of its values (a
// group) give every codebook entry's dot product with them, T[g][j] = cb[j] . x[8g .. 8g + 8) - 2^nbit of them, built
// in local memory - and a row's product is then the sum of T[g][idx(r, g)] over its groups: one 4-byte local read an
// index instead of an 8-value codebook gather and 8 multiply-adds. A work-group takes one column of x (xl lists the
// entries and their column slots that read it), a run of U 64-column units and R = L x RHO rows (each lane RHO), and
// builds the tables of P groups at a time (what local memory holds); a row's sum over the run is a partial, scaled by
// the row's gain. The partials of a row's runs are then added in order (the second kernel): the same arithmetic for a
// row whatever the other rows and columns of the launch (a verify pass's rows equal one-token rows).
namespace {
constexpr int LUT_L = 1024, LUT_RHO = 8, LUT_SLOTS = 16;

// index GI (0..7) of a chunk of 8 (its 12 bytes w0 w1 w2, the 13th bits pb)
template <int GI, bool PLANE>
inline uint32_t chunk_one(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t pb) {
    uint32_t v;
    if constexpr (GI < 5) v = (uint32_t) ((((uint64_t) w1 << 32) | w0) >> (12 * GI)) & 0xFFFu;
    else if constexpr (GI == 5) v = ((w1 >> 28) | (w2 << 4)) & 0xFFFu;
    else if constexpr (GI == 6) v = (w2 >> 8) & 0xFFFu;
    else v = (w2 >> 20) & 0xFFFu;
    if constexpr (PLANE) v |= ((pb >> GI) & 1u) << 12;
    return v;
}

template <bool PLANE, int P>
void vq_lut_run(sycl::queue& q, const uint64_t* ptrs, const int32_t* meta, const int32_t* xl, int64_t nx, int64_t kb, int64_t ubn, int U,
                const uint8_t* cb, const float* x, float* part, int64_t yrows, int64_t rows, int64_t cols) {
    constexpr int NE = PLANE ? 8192 : 4096;
    const int64_t units = cols / 64, R = (int64_t) LUT_L * LUT_RHO;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> T(sycl::range<1>(P * NE), h);
        h.parallel_for(sycl::nd_range<1>(nx * ubn * kb * LUT_L, LUT_L), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t gi = it.get_group(0);
            const int64_t c = gi / (ubn * kb), ub = (gi / kb) % ubn, k = gi % kb;
            const int lane = (int) it.get_local_id(0);
            const int32_t* xlc = xl + c * (1 + 2 * LUT_SLOTS);
            const int n = sycl::min(xlc[0], LUT_SLOTS);
            const int64_t r_lo = k * R;
            if (r_lo >= (int64_t) n * rows) return;   // the whole work-group alike, before any barrier
            const int64_t u0 = ub * U, u1 = sycl::min<int64_t>(units, u0 + U);
            // this lane's rows r_lo + lane + L i: their payloads and rows (-1: none)
            const uint8_t* base[LUT_RHO];
            int32_t rr[LUT_RHO];
            float acc[LUT_RHO];
#pragma unroll
            for (int i = 0; i < LUT_RHO; ++i) {
                const int64_t ri = r_lo + lane + (int64_t) LUT_L * i;
                acc[i] = 0.f;
                base[i] = nullptr;
                rr[i] = -1;
                if (ri < (int64_t) n * rows) {
                    base[i] = reinterpret_cast<const uint8_t*>(ptrs[xlc[1 + 2 * (int) (ri / rows)]]);
                    rr[i] = (int32_t) (ri % rows);
                }
            }
            const int64_t mrow = cols / 8 * 12 / 8, prow = cols / 64, s0 = 32 + rows * 2, p0 = s0 + rows * mrow;
            const float* xc = x + c * cols;
            for (int64_t u = u0; u < u1; ++u) {
                uint32_t w0[LUT_RHO], w1[LUT_RHO], w2[LUT_RHO], pb[LUT_RHO];
#pragma unroll
                for (int i = 0; i < LUT_RHO; ++i) {
                    if (rr[i] >= 0) {
                        const uint32_t* sp = reinterpret_cast<const uint32_t*>(base[i] + s0 + rr[i] * mrow + 12 * u);
                        w0[i] = sp[0];
                        w1[i] = sp[1];
                        w2[i] = sp[2];
                        pb[i] = PLANE ? (uint32_t) base[i][p0 + rr[i] * prow + u] : 0u;
                    } else {
                        w0[i] = w1[i] = w2[i] = pb[i] = 0u;
                    }
                }
                auto pass = [&](auto pc) {
                    constexpr int PS = decltype(pc)::value;
                    // the tables of groups 8u + P PS .. + P, a lane every L-th entry
                    float xv[P][8];
#pragma unroll
                    for (int gg = 0; gg < P; ++gg) {
                        const sycl::float4* xp = reinterpret_cast<const sycl::float4*>(xc + (u * 8 + PS * P + gg) * 8);
                        const sycl::float4 a = xp[0], b = xp[1];
                        xv[gg][0] = a.x(); xv[gg][1] = a.y(); xv[gg][2] = a.z(); xv[gg][3] = a.w();
                        xv[gg][4] = b.x(); xv[gg][5] = b.y(); xv[gg][6] = b.z(); xv[gg][7] = b.w();
                    }
                    sycl::group_barrier(it.get_group());
                    for (int j = lane; j < NE; j += LUT_L) {
                        float wv[8];
                        vq_weights<1>(cb, nullptr, (uint32_t) j, wv);
#pragma unroll
                        for (int gg = 0; gg < P; ++gg) {
                            float s = 0.f;
#pragma unroll
                            for (int d = 0; d < 8; ++d) s += wv[d] * xv[gg][d];
                            T[gg * NE + j] = s;
                        }
                    }
                    sycl::group_barrier(it.get_group());
#pragma unroll
                    for (int i = 0; i < LUT_RHO; ++i) {
                        const uint32_t a0 = w0[i], a1 = w1[i], a2 = w2[i], ap = pb[i];
                        float s;
                        if constexpr (P == 2) {
                            s = T[chunk_one<PS * 2, PLANE>(a0, a1, a2, ap)] + T[NE + chunk_one<PS * 2 + 1, PLANE>(a0, a1, a2, ap)];
                        } else if constexpr (P == 4) {
                            s = (T[chunk_one<PS * 4, PLANE>(a0, a1, a2, ap)] + T[NE + chunk_one<PS * 4 + 1, PLANE>(a0, a1, a2, ap)]) +
                                (T[2 * NE + chunk_one<PS * 4 + 2, PLANE>(a0, a1, a2, ap)] + T[3 * NE + chunk_one<PS * 4 + 3, PLANE>(a0, a1, a2, ap)]);
                        } else {
                            s = ((T[chunk_one<0, PLANE>(a0, a1, a2, ap)] + T[NE + chunk_one<1, PLANE>(a0, a1, a2, ap)]) +
                                 (T[2 * NE + chunk_one<2, PLANE>(a0, a1, a2, ap)] + T[3 * NE + chunk_one<3, PLANE>(a0, a1, a2, ap)])) +
                                ((T[4 * NE + chunk_one<4, PLANE>(a0, a1, a2, ap)] + T[5 * NE + chunk_one<5, PLANE>(a0, a1, a2, ap)]) +
                                 (T[6 * NE + chunk_one<6, PLANE>(a0, a1, a2, ap)] + T[7 * NE + chunk_one<7, PLANE>(a0, a1, a2, ap)]));
                        }
                        acc[i] += s;
                    }
                };
                pass(std::integral_constant<int, 0>{});
                if constexpr (P <= 4) pass(std::integral_constant<int, 1>{});
                if constexpr (P == 2) {
                    pass(std::integral_constant<int, 2>{});
                    pass(std::integral_constant<int, 3>{});
                }
            }
#pragma unroll
            for (int i = 0; i < LUT_RHO; ++i) {
                if (rr[i] < 0) continue;
                const int64_t ri = r_lo + lane + (int64_t) LUT_L * i;
                const int sl = (int) (ri / rows);
                const int e = xlc[1 + 2 * sl], qq = xlc[2 + 2 * sl];
                const float gn = (float) reinterpret_cast<const sycl::half*>(base[i] + 32)[rr[i]] * 256.f;
                part[ub * yrows * rows + ((int64_t) meta[e * 10 + 1] + qq) * rows + rr[i]] = acc[i] * gn;
            }
        });
    });
}

// the partials' memory, a GPU's (grown when a launch needs more; kept for the process)
float* lut_buf(ns_gpu* g, size_t floats) {
    static std::mutex mu;
    static std::vector<std::pair<ns_gpu*, std::pair<float*, size_t>>> bufs;
    std::lock_guard<std::mutex> lk(mu);
    for (auto& b : bufs) {
        if (b.first != g) continue;
        if (b.second.second >= floats) return b.second.first;
        g->q.wait();
        sycl::free(b.second.first, g->q);
        b.second.second = std::max(floats, (size_t) 4 << 20);
        b.second.first = sycl::malloc_device<float>(b.second.second, g->q);
        return b.second.first;
    }
    const size_t cap = std::max(floats, (size_t) 4 << 20);
    float* p = sycl::malloc_device<float>(cap, g->q);
    bufs.push_back({g, {p, cap}});
    return p;
}

size_t lut_local_bytes(ns_gpu* g) {
    static std::mutex mu;
    static std::vector<std::pair<ns_gpu*, size_t>> v;
    std::lock_guard<std::mutex> lk(mu);
    for (auto& b : v)
        if (b.first == g) return b.second;
    const size_t n = g->q.get_device().get_info<sycl::info::device::local_mem_size>();
    v.push_back({g, n});
    return n;
}
}  // namespace

// A prompt pass's expert products on XMX straight from the indices (ns_ds41_vq_gemm): y [n][ldy] = x [n][K] . W^T for
// a VQ W of `rows` rows (two payloads: rows < split the first's, the rest the second's - an expert's gate and up as one
// matrix). A sub-group takes 16 rows of W and 8 MT tokens: a lane decodes its row's 64 columns a unit (8 indices, each
// 8 fp16 codebook values) into local memory as four packed 16 x 16 B tiles, then each meets the MT token tiles (A, fp16
// rows of x from global memory) - half x half -> float. The rows' gains scale the sums on the way out. Nothing is
// expanded to memory: an expansion wrote and read the whole matrix in fp16 whatever its tokens.
namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;

template <typename T>
inline auto gm_lp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto gm_gp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

template <bool PLANE, int MT, int KS, bool DIRECT>
void vq_gemm_run(sycl::queue& q, const uint8_t* p0, const uint8_t* p1, int64_t split, int64_t rows, int64_t cols, const sycl::half* cbh,
                 const sycl::half* x, int64_t ldx, int64_t n, float* y, int64_t ldy) {
    const int64_t rtiles = rows / 16, ttiles = (n + 8 * MT - 1) / (8 * MT), units = cols / 64, ups = (units + KS - 1) / KS;
    q.submit([&](sycl::handler& h) {
        // per sub-group: a unit's four B tiles [4][8][16] words (two halves each); then its MT sums [8][16] floats
        sycl::local_accessor<uint32_t, 1> bt(sycl::range<1>(KS * 4 * 128), h);
        sycl::local_accessor<float, 1> cs(sycl::range<1>(KS * MT * 128), h);
        h.parallel_for(sycl::nd_range<1>(rtiles * ttiles * KS * 16, KS * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const int s = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
            const int64_t gi = it.get_group(0), tt = gi % ttiles, rt = gi / ttiles;
            const int64_t r = rt * 16 + lane;
            const bool second = r >= split;
            const int64_t mr = second ? rows - split : split, rr = second ? r - split : r;
            const VqMat m = vq_open(second ? p1 : p0, mr, cols, PLANE ? 13 : 12);
            const uint32_t* srow = reinterpret_cast<const uint32_t*>(m.stream + rr * m.mrow);
            const uint8_t* prow = PLANE ? m.plane + rr * m.prow : nullptr;
            uint32_t* b = &bt[s * 512];
            const int64_t tok0 = tt * 8 * MT;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> acc[MT];
#pragma unroll
            for (int mm = 0; mm < MT; ++mm) jm::joint_matrix_fill(sg, acc[mm], 0.0f);
            const int64_t ua = s * ups, ub = sycl::min<int64_t>(units, ua + ups);
            if constexpr (DIRECT) {
                // lane n holds column n of a B tile (16 consecutive k of weight row n: glm5next's probed layout): two
                // indices' 16 values straight into the tile, no local memory
                for (int64_t u = ua; u < ub; ++u) {
                    uint32_t idx[8];
                    chunk_idx(srow + 3 * u, PLANE ? prow[u] : 0u, idx);
#pragma unroll
                    for (int kt = 0; kt < 4; ++kt) {
                        const sycl::vec<sycl::half, 8> h0 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(cbh + (size_t) idx[2 * kt] * 8);
                        const sycl::vec<sycl::half, 8> h1 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(cbh + (size_t) idx[2 * kt + 1] * 8);
                        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> bm;
                        jm::joint_matrix_fill(sg, bm, sycl::half(0));
                        int e = 0;
                        jm::joint_matrix_apply(sg, bm, [&](sycl::half& hh) {
                            hh = e < 8 ? h0[e & 7] : h1[e & 7];
                            ++e;
                        });
#pragma unroll
                        for (int mm = 0; mm < MT; ++mm) {
                            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> am;
                            jm::joint_matrix_load(sg, am, gm_gp(x + (tok0 + 8 * mm) * ldx + u * 64 + kt * 16), (size_t) ldx);
                            jm::joint_matrix_mad(sg, acc[mm], am, bm, acc[mm]);
                        }
                    }
                }
            } else
            for (int64_t u = ua; u < ub; ++u) {
                uint32_t idx[8];
                chunk_idx(srow + 3 * u, PLANE ? prow[u] : 0u, idx);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const sycl::uint4 hv = *reinterpret_cast<const sycl::uint4*>(cbh + (size_t) idx[i] * 8);
                    // values k = 8i .. 8i + 7 as four pairs: tile k / 16, pair row (k % 16) / 2, this lane's column
                    const int t = i >> 1, pr = (i & 1) * 4;
                    b[t * 128 + (pr + 0) * 16 + lane] = hv.x();
                    b[t * 128 + (pr + 1) * 16 + lane] = hv.y();
                    b[t * 128 + (pr + 2) * 16 + lane] = hv.z();
                    b[t * 128 + (pr + 3) * 16 + lane] = hv.w();
                }
                sycl::group_barrier(sg);
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> bm;
                    jm::joint_matrix_load(sg, bm, gm_lp(reinterpret_cast<sycl::half*>(b + kt * 128)), (size_t) 32);
#pragma unroll
                    for (int mm = 0; mm < MT; ++mm) {
                        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> am;
                        jm::joint_matrix_load(sg, am, gm_gp(x + (tok0 + 8 * mm) * ldx + u * 64 + kt * 16), (size_t) ldx);
                        jm::joint_matrix_mad(sg, acc[mm], am, bm, acc[mm]);
                    }
                }
                sycl::group_barrier(sg);
            }
#pragma unroll
            for (int mm = 0; mm < MT; ++mm) jm::joint_matrix_store(sg, acc[mm], gm_lp(&cs[(s * MT + mm) * 128]), (size_t) 16, jm::layout::row_major);
            sycl::group_barrier(it.get_group());
            // the K slices' sums added in order, scaled by the rows' gains
            for (int o = (int) it.get_local_id(0); o < MT * 128; o += KS * 16) {
                const int mm = o / 128, t = (o % 128) / 16, col = o % 16;
                const int64_t tok = tok0 + 8 * mm + t;
                if (tok >= n) continue;
                float v = 0.f;
                for (int k = 0; k < KS; ++k) v += cs[(k * MT + mm) * 128 + t * 16 + col];
                const int64_t row = rt * 16 + col;
                const bool sec = row >= split;
                const uint8_t* pp = sec ? p1 : p0;
                const float gn = (float) reinterpret_cast<const sycl::half*>(pp + 32)[sec ? row - split : row] * 256.f;
                y[tok * ldy + row] = v * gn;
            }
        });
    });
}

// RT row tiles a sub-group (a lane decodes rows r, r + 16, ...): each A tile loaded meets RT B tiles - the A loads (x
// read again by every row tile of the matrix) were what the one-tile form spent most on, the B65 more than the B70
template <bool PLANE, int MT, int KS, int RT>
void vq_gemm_rt(sycl::queue& q, const uint8_t* p0, const uint8_t* p1, int64_t split, int64_t rows, int64_t cols, const sycl::half* cbh,
                const sycl::half* x, int64_t ldx, int64_t n, float* y, int64_t ldy) {
    const int64_t rgroups = rows / (16 * RT), ttiles = (n + 8 * MT - 1) / (8 * MT), units = cols / 64, ups = (units + KS - 1) / KS;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<(RT * MT > 8 ? 256 : 128)>};
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> cs(sycl::range<1>(KS * RT * MT * 128), h);
        h.parallel_for(sycl::nd_range<1>(rgroups * ttiles * KS * 16, KS * 16), props, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const int s = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
            const int64_t gi = it.get_group(0), tt = gi % ttiles, rg = gi / ttiles;
            const uint32_t* srow[RT];
            const uint8_t* prow[RT];
#pragma unroll
            for (int t = 0; t < RT; ++t) {
                const int64_t r = (rg * RT + t) * 16 + lane;
                const bool second = r >= split;
                const int64_t mr = second ? rows - split : split, rr = second ? r - split : r;
                const VqMat m = vq_open(second ? p1 : p0, mr, cols, PLANE ? 13 : 12);
                srow[t] = reinterpret_cast<const uint32_t*>(m.stream + rr * m.mrow);
                prow[t] = PLANE ? m.plane + rr * m.prow : nullptr;
            }
            const int64_t tok0 = tt * 8 * MT;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> acc[RT][MT];
#pragma unroll
            for (int t = 0; t < RT; ++t)
#pragma unroll
                for (int mm = 0; mm < MT; ++mm) jm::joint_matrix_fill(sg, acc[t][mm], 0.0f);
            const int64_t ua = s * ups, ub = sycl::min<int64_t>(units, ua + ups);
            for (int64_t u = ua; u < ub; ++u) {
                uint32_t idx[RT][8];
#pragma unroll
                for (int t = 0; t < RT; ++t) chunk_idx(srow[t] + 3 * u, PLANE ? prow[t][u] : 0u, idx[t]);
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> bm[RT];
#pragma unroll
                    for (int t = 0; t < RT; ++t) {
                        const sycl::vec<sycl::half, 8> h0 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(cbh + (size_t) idx[t][2 * kt] * 8);
                        const sycl::vec<sycl::half, 8> h1 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(cbh + (size_t) idx[t][2 * kt + 1] * 8);
                        jm::joint_matrix_fill(sg, bm[t], sycl::half(0));
                        int e = 0;
                        jm::joint_matrix_apply(sg, bm[t], [&](sycl::half& hh) {
                            hh = e < 8 ? h0[e & 7] : h1[e & 7];
                            ++e;
                        });
                    }
#pragma unroll
                    for (int mm = 0; mm < MT; ++mm) {
                        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> am;
                        jm::joint_matrix_load(sg, am, gm_gp(x + (tok0 + 8 * mm) * ldx + u * 64 + kt * 16), (size_t) ldx);
#pragma unroll
                        for (int t = 0; t < RT; ++t) jm::joint_matrix_mad(sg, acc[t][mm], am, bm[t], acc[t][mm]);
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < RT; ++t)
#pragma unroll
                for (int mm = 0; mm < MT; ++mm)
                    jm::joint_matrix_store(sg, acc[t][mm], gm_lp(&cs[((s * RT + t) * MT + mm) * 128]), (size_t) 16, jm::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int o = (int) it.get_local_id(0); o < RT * MT * 128; o += KS * 16) {
                const int t = o / (MT * 128), mm = (o / 128) % MT, ti = (o % 128) / 16, col = o % 16;
                const int64_t tok = tok0 + 8 * mm + ti;
                if (tok >= n) continue;
                float v = 0.f;
                for (int k = 0; k < KS; ++k) v += cs[((k * RT + t) * MT + mm) * 128 + ti * 16 + col];
                const int64_t row = (rg * RT + t) * 16 + col;
                const bool sec = row >= split;
                const uint8_t* pp = sec ? p1 : p0;
                const float gn = (float) reinterpret_cast<const sycl::half*>(pp + 32)[sec ? row - split : row] * 256.f;
                y[tok * ldy + row] = v * gn;
            }
        });
    });
}
}  // namespace

extern "C" {

int ns_ds41_vq_supported(int64_t rows, int64_t cols, int64_t nbit) {
    return rows > 0 && cols % 64 == 0 && (nbit == 12 || nbit == 13);
}

int ns_ds41_vq_codebook_f16(ns_gpu* g, const uint8_t* cb, int64_t n, void* outv) {
    NS_TRY
    sycl::half* out = static_cast<sycl::half*>(outv);
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] = (sycl::half) e4m3_scaled(cb[i]); });
    return 0;
    NS_CATCH
}

int ns_ds41_vq_codebook_f32(ns_gpu* g, const uint8_t* cb, int64_t n, float* out) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] = e4m3_scaled(cb[i]); });
    return 0;
    NS_CATCH
}

int ns_ds41_vq_dequant(ns_gpu* g, const void* payload, const uint8_t* cb, int64_t rows, int64_t cols, int64_t nbit, void* out, int f16) {
    NS_TRY
    if (!ns_ds41_vq_supported(rows, cols, nbit)) return ns_fail("ns_ds41_vq_dequant: cols a multiple of 64, 12 or 13 bits");
    const VqMat m = vq_open(static_cast<const uint8_t*>(payload), rows, cols, nbit);
    const int64_t chunks = cols / 64;
    float* of = static_cast<float*>(out);
    sycl::half* oh = static_cast<sycl::half*>(out);
    g->q.parallel_for(sycl::range<2>(rows, chunks), [=](sycl::id<2> id) {
        const int64_t r = id[0], k = id[1];
        uint32_t idx[8];
        chunk_idx(reinterpret_cast<const uint32_t*>(m.stream + r * m.mrow + 12 * k), m.plane ? m.plane[r * m.prow + k] : 0u, idx);
        const float gn = (float) m.gains[r] * 256.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // an index's 8 bytes in one load (byte loads were 8 gathers an index: what bound a prompt pass's experts)
            float wv[8];
            vq_weights<0>(cb, nullptr, idx[i], wv);
            const int64_t at = r * cols + k * 64 + i * 8;
            if (f16) {
                sycl::vec<sycl::half, 8> h;
#pragma unroll
                for (int d = 0; d < 8; ++d) h[d] = (sycl::half) (wv[d] * gn);
                *reinterpret_cast<sycl::vec<sycl::half, 8>*>(oh + at) = h;
            } else {
#pragma unroll
                for (int d = 0; d < 8; ++d) of[at + d] = wv[d] * gn;
            }
        }
    });
    return 0;
    NS_CATCH
}

// Variants 0 and 3: a sub-group of 16 an output row, RPG rows a work-group (RPS a sub-group in turn); each lane a run of 64-column chunks,
// every column at once. The codebook's gathers go through L1 (a lane an index): as float32 (cbf, widened once a layer:
// no conversion a weight), fp16 (3: cbf is the fp16 codebook) or the E4M3 bytes (cbf null). Measured on the B70 (6 experts of a 12-bit layer, one token): gate | up
// 208 us, down 121 us with cbf (218 / 144 from the bytes); kept local-memory codebooks (bytes, fp16; persistent or not;
// rows staged; 4 rows sharing the activations) all measured slower, 217-720 us.
int ns_ds41_vq_mmv(ns_gpu* g, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const uint8_t* cb, const float* cbf, const float* x, float* y,
                   int64_t rows, int64_t cols, int64_t nbit, int variant) {
    NS_TRY
    if (!ns_ds41_vq_supported(rows, cols, nbit)) return ns_fail("ns_ds41_vq_mmv: cols a multiple of 64, 12 or 13 bits");
    const sycl::half* cbh = reinterpret_cast<const sycl::half*>(cbf);
    if (variant == 1 || variant == 2 || variant == 4) {
        const bool p = nbit > 12;
#define VQR(P, C) vq_mmv_rows<P, C>(g->q, ptrs, meta, entries, cb, cbh, x, y, rows, cols, 64)
        if (variant == 1) { if (p) VQR(true, 0); else VQR(false, 0); }
        else if (variant == 2) { if (p) VQR(true, 2); else VQR(false, 2); }
        else { if (p) VQR(true, 1); else VQR(false, 1); }
#undef VQR
        return 0;
    }
    constexpr int SG = 16, MC = 8;
    static const int rpg_s = [] { const char* v = std::getenv("NS_DS41_VQ_RPG"); return v ? std::atoi(v) : 8; }();
    static const int rps_s = [] { const char* v = std::getenv("NS_DS41_VQ_RPS"); return v ? std::atoi(v) : 1; }();
    const int RPG = rpg_s, RPS = rps_s;
    const int64_t rgs = (rows + RPG * RPS - 1) / (RPG * RPS), chunks = cols / 64;
    const bool plane = nbit > 12;
    g->q.parallel_for(sycl::nd_range<1>(entries * rgs * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t gi = it.get_group(0), e = gi / rgs, r0 = ((gi % rgs) * RPG + sg.get_group_id()[0]) * RPS;
        const int lane = sg.get_local_id()[0];
        for (int q = 0; q < RPS; ++q) {
        const int64_t r = r0 + q;
        if (r >= rows) return;
        const int32_t* me = meta + e * 10;
        const int nc = me[0];
        const VqMat m = vq_open(reinterpret_cast<const uint8_t*>(ptrs[e]), rows, cols, plane ? 13 : 12);
        const uint8_t* srow = m.stream + r * m.mrow;
        const uint8_t* prow = plane ? m.plane + r * m.prow : nullptr;
        float acc[MC] = {};
        for (int64_t k = lane; k < chunks; k += SG) {
            uint32_t idx[8];
            chunk_idx(reinterpret_cast<const uint32_t*>(srow + 12 * k), prow ? prow[k] : 0u, idx);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float wv[8];
                if (variant == 3) {
                    vq_weights<2>(cb, cbh, idx[i], wv);
                } else if (cbf) {
                    const sycl::float4* cw = reinterpret_cast<const sycl::float4*>(cbf + (size_t) idx[i] * 8);
                    const sycl::float4 c0 = cw[0], c1 = cw[1];
                    wv[0] = c0.x(); wv[1] = c0.y(); wv[2] = c0.z(); wv[3] = c0.w();
                    wv[4] = c1.x(); wv[5] = c1.y(); wv[6] = c1.z(); wv[7] = c1.w();
                } else {
                    const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(cb + (size_t) idx[i] * 8);
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        wv[d] = e4m3_scaled((w.x() >> (8 * d)) & 0xFF);
                        wv[4 + d] = e4m3_scaled((w.y() >> (8 * d)) & 0xFF);
                    }
                }
                for (int c = 0; c < nc; ++c) {
                    const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(x + (int64_t) me[2 + c] * cols + k * 64 + i * 8);
                    const sycl::float4 a = xv[0], b = xv[1];
                    acc[c] += wv[0] * a.x() + wv[1] * a.y() + wv[2] * a.z() + wv[3] * a.w() + wv[4] * b.x() + wv[5] * b.y() + wv[6] * b.z() + wv[7] * b.w();
                }
            }
        }
        const float gn = (float) m.gains[r] * 256.f;
        for (int c = 0; c < nc; ++c) {
            const float s = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
            if (lane == 0) y[(int64_t) (me[1] + c) * rows + r] = s * gn;
        }
        }
    });
    return 0;
    NS_CATCH
}

int ns_ds41_fp8_dequant(ns_gpu* g, const uint8_t* w, int64_t rows, int64_t cols, int64_t r0, int64_t n, void* out, int f16) {
    NS_TRY
    if (r0 < 0 || r0 + n > rows) return ns_fail("ns_ds41_fp8_dequant: rows past the matrix");
    const int64_t tc = (cols + 31) / 32;
    const uint8_t* sc = w + rows * cols;
    float* of = static_cast<float*>(out);
    sycl::half* oh = static_cast<sycl::half*>(out);
    g->q.parallel_for(sycl::range<2>(n, cols), [=](sycl::id<2> id) {
        const int64_t r = r0 + id[0], c = id[1];
        const uint8_t s = sc[(r / 32) * tc + c / 32];
        const float scale = sycl::bit_cast<float>(s == 0 ? 0x00400000u : ((uint32_t) s << 23)) * 256.f;
        const float v = e4m3_scaled(w[r * cols + c]) * scale;
        if (f16) oh[id[0] * cols + c] = (sycl::half) v;
        else of[id[0] * cols + c] = v;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_fp8_mmv(ns_gpu* g, const uint8_t* w, int64_t rows, int64_t cols, const float* x, int64_t xcs, int64_t rpg, int64_t xgs, float* y, int64_t ycs,
                    int64_t t) {
    NS_TRY
    if (cols % 32 != 0 || t < 1 || t > 8) return ns_fail("ns_ds41_fp8_mmv: cols a multiple of 32, 1..8 rows of x");
    constexpr int SG = 16, RPG = 8;
    const int64_t tc = cols / 32;
    const uint8_t* sc = w + rows * cols;
    g->q.parallel_for(sycl::nd_range<1>((rows + RPG - 1) / RPG * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0) * RPG + sg.get_group_id()[0];
        const int lane = sg.get_local_id()[0];
        if (r >= rows) return;
        const float* xr = x + (rpg > 0 ? (r / rpg) * xgs : 0);
        float acc[8] = {};
        for (int64_t k = lane; k < tc; k += SG) {
            const uint8_t s = sc[(r / 32) * tc + k];
            const float scale = sycl::bit_cast<float>(s == 0 ? 0x00400000u : ((uint32_t) s << 23)) * 256.f;
            const uint32_t* wr = reinterpret_cast<const uint32_t*>(w + r * cols + k * 32);
            float wv[32];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const uint32_t u = wr[q];
#pragma unroll
                for (int d = 0; d < 4; ++d) wv[q * 4 + d] = e4m3_scaled((u >> (8 * d)) & 0xFF);
            }
            for (int c = 0; c < t; ++c) {
                const float* xc = xr + c * xcs + k * 32;
                float s2 = 0.f;
#pragma unroll
                for (int d = 0; d < 32; ++d) s2 += wv[d] * xc[d];
                acc[c] += s2 * scale;
            }
        }
        for (int c = 0; c < t; ++c) {
            const float s = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
            if (lane == 0) y[c * ycs + r] = s;
        }
    });
    return 0;
    NS_CATCH
}

int ns_ds41_vq_lut(ns_gpu* g, const uint64_t* ptrs, const int32_t* meta, const int32_t* xl, int64_t nx, int64_t maxn, const uint8_t* cb, const float* x,
                   float* y, int64_t yrows, int64_t rows, int64_t cols, int64_t nbit, int64_t units_per_group) {
    NS_TRY
    if (!ns_ds41_vq_supported(rows, cols, nbit)) return ns_fail("ns_ds41_vq_lut: cols a multiple of 64, 12 or 13 bits");
    if (nx < 1 || maxn < 1 || maxn > LUT_SLOTS || yrows < 1) return ns_fail("ns_ds41_vq_lut: 1..16 entries a column of x");
    const int64_t units = cols / 64;
    const int U = (int) std::min<int64_t>(std::max<int64_t>(units_per_group, 1), units);
    const int64_t ubn = (units + U - 1) / U, R = (int64_t) LUT_L * LUT_RHO, kb = (maxn * rows + R - 1) / R;
    float* part = lut_buf(g, (size_t) (ubn * yrows * rows));
    if (!part) return ns_fail("ns_ds41_vq_lut: no memory for the partials");
    const size_t lm = lut_local_bytes(g);
    const size_t ne = nbit > 12 ? 8192 : 4096;
    const bool plane = nbit > 12;
    // the most groups' tables local memory holds (less a little for the runtime's own)
    if (8 * ne * 4 + 1024 <= lm) {
        if (plane) vq_lut_run<true, 8>(g->q, ptrs, meta, xl, nx, kb, ubn, U, cb, x, part, yrows, rows, cols);
        else vq_lut_run<false, 8>(g->q, ptrs, meta, xl, nx, kb, ubn, U, cb, x, part, yrows, rows, cols);
    } else if (4 * ne * 4 + 1024 <= lm) {
        if (plane) vq_lut_run<true, 4>(g->q, ptrs, meta, xl, nx, kb, ubn, U, cb, x, part, yrows, rows, cols);
        else vq_lut_run<false, 4>(g->q, ptrs, meta, xl, nx, kb, ubn, U, cb, x, part, yrows, rows, cols);
    } else if (2 * ne * 4 + 1024 <= lm) {
        if (plane) vq_lut_run<true, 2>(g->q, ptrs, meta, xl, nx, kb, ubn, U, cb, x, part, yrows, rows, cols);
        else vq_lut_run<false, 2>(g->q, ptrs, meta, xl, nx, kb, ubn, U, cb, x, part, yrows, rows, cols);
    } else {
        return ns_fail("ns_ds41_vq_lut: too little local memory for two groups' tables");
    }
    // each listed row's partials added in order of their unit runs
    const int64_t total = nx * maxn * rows;
    g->q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        const int64_t c = id[0] / (maxn * rows), s = (id[0] / rows) % maxn, r = id[0] % rows;
        const int32_t* xlc = xl + c * (1 + 2 * LUT_SLOTS);
        if (s >= sycl::min(xlc[0], LUT_SLOTS)) return;
        const int e = xlc[1 + 2 * s], qq = xlc[2 + 2 * s];
        const int64_t at = ((int64_t) meta[e * 10 + 1] + qq) * rows + r;
        float v = 0.f;
        for (int64_t b = 0; b < ubn; ++b) v += part[b * yrows * rows + at];
        y[at] = v;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_vq_gemm(ns_gpu* g, const void* p0, const void* p1, int64_t split, int64_t rows, int64_t cols, int64_t nbit, const void* cbh, const void* x,
                    int64_t ldx, int64_t n, float* y, int64_t ldy) {
    NS_TRY
    if (rows % 16 != 0 || cols % 64 != 0 || split < 1 || split > rows || split % 16 != 0 || (nbit != 12 && nbit != 13))
        return ns_fail("ns_ds41_vq_gemm: rows and the split multiples of 16, cols of 64, 12 or 13 bits");
    if (n < 1) return 0;
    const auto* a = static_cast<const uint8_t*>(p0);
    const auto* b = static_cast<const uint8_t*>(p1 ? p1 : p0);
    const auto* ch = static_cast<const sycl::half*>(cbh);
    const auto* xh = static_cast<const sycl::half*>(x);
    // NS_DS41_VQ_GEMM_V: 0 the B tiles through local memory, 1 straight into the tiles' registers (8 K slices a
    // work-group); 2 / 3 as they with 4 K slices; 4 / 5 as 2 / 3 and past 32 tokens 8 token tiles a sub-group. B70 + B65,
    // an 8,734-token prompt: 379 / 415 / 435 / 421 / 441 / 472 tok/s (one run each: +-5%)
    static const int v = [] { const char* e = std::getenv("NS_DS41_VQ_GEMM_V"); return e ? std::atoi(e) : 5; }();
#define VR(P, M, KS, R) vq_gemm_rt<P, M, KS, R>(g->q, a, b, split, rows, cols, ch, xh, ldx, n, y, ldy)
    // 8 / 9: 2 row tiles a sub-group, 4 token tiles, 8 / 16 K slices a work-group (the occupancy the row tiles cost)
    if (v == 8 || v == 9) {
        if (rows % 32 != 0) return ns_fail("ns_ds41_vq_gemm: rows a multiple of 32");
        if (nbit > 12) { if (v == 8) VR(true, 4, 8, 2); else VR(true, 4, 16, 2); }
        else { if (v == 8) VR(false, 4, 8, 2); else VR(false, 4, 16, 2); }
        return 0;
    }
    // 6: 2 row tiles a sub-group, 7: 4 (the large register file)
    if (v == 6 || v == 7) {
        const bool w = n > 32;
        if (rows % (16 * (v == 7 ? 4 : 2)) != 0) return ns_fail("ns_ds41_vq_gemm: rows a multiple of the row tiles");
        if (nbit > 12) {
            if (v == 6) { if (w) VR(true, 8, 4, 2); else VR(true, 4, 4, 2); } else { if (w) VR(true, 8, 4, 4); else VR(true, 4, 4, 4); }
        } else {
            if (v == 6) { if (w) VR(false, 8, 4, 2); else VR(false, 4, 4, 2); } else { if (w) VR(false, 8, 4, 4); else VR(false, 4, 4, 4); }
        }
        return 0;
    }
#undef VR
#define VG(P, M, KS, D) vq_gemm_run<P, M, KS, D>(g->q, a, b, split, rows, cols, ch, xh, ldx, n, y, ldy)
    // past 32 tokens, 8 token tiles a sub-group (each decoded row meets 64 tokens)
    const bool wide = n > 32 && v >= 4;
    if (nbit > 12) {
        if (wide) { if (v == 4) VG(true, 8, 4, false); else VG(true, 8, 4, true); }
        else if (v == 0) VG(true, 4, 8, false); else if (v == 2 || v == 4) VG(true, 4, 4, false); else if (v == 3 || v == 5) VG(true, 4, 4, true); else VG(true, 4, 8, true);
    } else {
        if (wide) { if (v == 4) VG(false, 8, 4, false); else VG(false, 8, 4, true); }
        else if (v == 0) VG(false, 4, 8, false); else if (v == 2 || v == 4) VG(false, 4, 4, false); else if (v == 3 || v == 5) VG(false, 4, 4, true); else VG(false, 4, 8, true);
    }
#undef VG
    return 0;
    NS_CATCH
}

}  // extern "C"
