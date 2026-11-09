// xmm.cpp: the qwen35 silo's decode products on XMX for 3..8 columns (silo.h: ns_q35_mmvq routes here). The dp4a products
// (mmvq.cpp) read each weight once for all columns but spend 8 dp4a a column on every 32 weights: at 8 columns they are
// compute-bound (IQ4_XS 17408x5120: 183 us against the 101 us of one column, ~260 GB/s of weights where one column
// gets ~480), so a verify pass of 8 rows (DSpine's, MTP's with more drafts) cost 80 ms against 37 for one row.
//
// Here the weights are the B operand of int8 DPAS (8 x 32 x 16: 8 tokens, 32 K, 16 output rows): a lane a row of a
// 16-row tile expands its 32 bytes of a 64-weight group (two blocks of 32) from the stored blocks - no repacking -
// into the packed tile layout in local memory, and the DPAS does the 32-weight dot for all 16 rows x 8 tokens at once.
// Each int32 sum is scaled by the lane's row's sub-block scale and the token's block scale (K-quants: less the
// minimum x the token's quant sum). The sums are exact integers, so a column's value differs from the dp4a path only
// in the order its float terms are added (not bit-identical to a one-column pass; within rounding).
//
// Types: Q4_K, Q5_K, IQ4_XS (rows n_in / 256 blocks). The activations are read in a token-major form made by a small
// prep launch from the split layout (xmx_prep): block b's 8 tokens x 32 quants at b x 256, then the scales and the
// scale x quant-sum, [block][8] float each; the split buffer's tail holds them (silo.h: ns_q35_q8_split_bytes).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"
#include "common.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <cstdlib>

namespace {
using namespace q35silo;
namespace jm = sycl::ext::oneapi::experimental::matrix;
constexpr int SG = 16;
constexpr int Q80 = 8, Q4K = 12, Q5K = 13, Q6K = 14, IQ4XS = 23;

template <typename T>
inline auto gpm(T* p) { return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p); }
template <typename T>
inline auto lpm(T* p) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p); }

inline int64_t block_bytes(int ty) { return ty == Q4K ? 144 : ty == Q5K ? 176 : ty == Q80 ? 34 : 136; }   // Q8_0's is a block of 32
inline int64_t row_bytes(int ty, int64_t n_in) { return (n_in / (ty == Q80 ? 32 : 256)) * block_bytes(ty); }

// a byte of two nibbles -> their codebook bytes (unsigned) at bits 0 and 16 (mmvq.cpp's IQ4 table); four stored bytes ->
// the codebook bytes of their low nibbles (lo) and of their high nibbles (hi)
inline uint32_t iq4_entry(int b) {
    constexpr int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return uint32_t(uint8_t(kv[b & 15])) | uint32_t(uint8_t(kv[b >> 4])) << 16;
}
inline void iq4_tab4(const uint32_t* tab, uint32_t v, uint32_t& lo, uint32_t& hi) {
    const uint32_t a = tab[v & 0xff] | tab[(v >> 8) & 0xff] << 8, c = tab[(v >> 16) & 0xff] | tab[v >> 24] << 8;
    lo = (a & 0xffff) | c << 16;
    hi = a >> 16 | (c & 0xffff0000u);
}

// SK sub-groups a work-group split a 16-row tile's K groups (64 weights: KG consecutive ones a step, their loads first)
template <int TY, int SK, int KG>
void xmx_mmvq(sycl::queue& q, const uint8_t* w, const int8_t* xa, const float* ts, const float* tm, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    constexpr int NS = 2, NT = 2;   // tile sets used in turn, tiles a group (its two 32-weight blocks)
    const int64_t ng = n_in / 64, rb = row_bytes(TY, n_in);
    const int64_t groups = (n_out + 15) / 16;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(SK > 1 ? SK * 8 * 16 : 1), h);
        sycl::local_accessor<uint32_t, 1> tiles(sycl::range<1>(SK * NS * NT * 128), h);
        sycl::local_accessor<uint32_t, 1> lut(sycl::range<1>(TY == IQ4XS ? 256 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SK * SG), sycl::range<1>(SK * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            uint32_t* tab = lut.get_multi_ptr<sycl::access::decorated::no>().get();
            if constexpr (TY == IQ4XS) {
                for (int i = (int) it.get_local_id(0); i < 256; i += SK * SG) tab[i] = iq4_entry(i);
                sycl::group_barrier(it.get_group());
            }
            const int64_t row = it.get_group(0) * 16 + lane;
            const uint8_t* wr = w + (row < n_out ? row : n_out - 1) * rb;
            uint32_t* tl0 = tiles.get_multi_ptr<sycl::access::decorated::no>().get() + sgi * NS * NT * 128;
            int par = 0;
            float acc[8], accm[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = accm[e] = 0.0f;
            for (int64_t g0 = (int64_t) sgi * KG; g0 < ng; g0 += SK * KG) {
                // the step's weights: KG groups of the lane's row (32 bytes of nibbles, the block's head, Q5_K's high bits)
                uint32_t nd[KG][TY == Q80 ? 16 : 8], hq[KG][8];
                sycl::int4 hd[KG];
                float dq[KG][2];
#pragma unroll
                for (int kg = 0; kg < KG; ++kg) {
                    const int64_t g = sycl::min(g0 + kg, ng - 1), blk = g / 4, j = g % 4;
                    const uint8_t* b = wr + blk * block_bytes(TY);
                    if constexpr (TY == Q80) {
                        // two blocks of 32 (34 bytes, 2-byte aligned): their scales and 2 x 32 quants
#pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            const uint8_t* pb = wr + (2 * g + c) * 34;
                            dq[kg][c] = (float) *reinterpret_cast<const sycl::half*>(pb);
                            const sycl::int4 a = load16_a2(pb + 2), e = load16_a2(pb + 18);
                            nd[kg][8 * c] = a.x(); nd[kg][8 * c + 1] = a.y(); nd[kg][8 * c + 2] = a.z(); nd[kg][8 * c + 3] = a.w();
                            nd[kg][8 * c + 4] = e.x(); nd[kg][8 * c + 5] = e.y(); nd[kg][8 * c + 6] = e.z(); nd[kg][8 * c + 7] = e.w();
                        }
                    } else if constexpr (TY == IQ4XS) {
                        const sycl::int2 hh = *reinterpret_cast<const sycl::int2*>(b);
                        hd[kg] = sycl::int4(hh.x(), hh.y(), 0, 0);
                        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b + 8 + 32 * j);
#pragma unroll
                        for (int i = 0; i < 4; ++i) { nd[kg][2 * i] = (uint32_t) q2[i].x(); nd[kg][2 * i + 1] = (uint32_t) q2[i].y(); }
                    } else {
                        hd[kg] = *reinterpret_cast<const sycl::int4*>(b);
                        const uint8_t* qs = b + (TY == Q5K ? 48 : 16) + 32 * j;
                        const sycl::int4 a = reinterpret_cast<const sycl::int4*>(qs)[0], c = reinterpret_cast<const sycl::int4*>(qs)[1];
                        nd[kg][0] = a.x(); nd[kg][1] = a.y(); nd[kg][2] = a.z(); nd[kg][3] = a.w();
                        nd[kg][4] = c.x(); nd[kg][5] = c.y(); nd[kg][6] = c.z(); nd[kg][7] = c.w();
                        if constexpr (TY == Q5K) {
                            const sycl::int4 p = reinterpret_cast<const sycl::int4*>(b + 16)[0], r = reinterpret_cast<const sycl::int4*>(b + 16)[1];
                            hq[kg][0] = p.x(); hq[kg][1] = p.y(); hq[kg][2] = p.z(); hq[kg][3] = p.w();
                            hq[kg][4] = r.x(); hq[kg][5] = r.y(); hq[kg][6] = r.z(); hq[kg][7] = r.w();
                        }
                    }
                }
#pragma unroll
                for (int kg = 0; kg < KG; ++kg) {
                    const int64_t g = g0 + kg;
                    if (g >= ng) break;
                    const int64_t k0 = 2 * g, j = g % 4;
                    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> am[2];
#pragma unroll
                    for (int b = 0; b < 2; ++b) jm::joint_matrix_load(sg, am[b], gpm(xa + (k0 + b) * 256), (size_t) 32);
                    sycl::vec<float, 8> s8[2], m8[2];
#pragma unroll
                    for (int b = 0; b < 2; ++b) {
                        s8[b] = *reinterpret_cast<const sycl::vec<float, 8>*>(ts + (k0 + b) * 8);
                        if constexpr (TY == Q4K || TY == Q5K) m8[b] = *reinterpret_cast<const sycl::vec<float, 8>*>(tm + (k0 + b) * 8);
                    }
                    uint32_t* tl = tl0 + par * NT * 128;
                    par ^= 1;
                    float sl[2], ml[2] = {0.0f, 0.0f};
                    if constexpr (TY == Q80) {
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) tl[b * 128 + i * 16 + lane] = nd[kg][8 * b + i];
                            sl[b] = dq[kg][b];
                        }
                    } else if constexpr (TY == IQ4XS) {
                        // tile b = sub-block 2j + b = the group's dwords 4b .. 4b + 3 (16 bytes): the low nibbles' values are
                        // K 0..15, the high nibbles' K 16..31
#pragma unroll
                        for (int b = 0; b < 2; ++b)
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                uint32_t lo, hi;
                                iq4_tab4(tab, nd[kg][4 * b + i], lo, hi);
                                tl[b * 128 + i * 16 + lane] = lo;
                                tl[b * 128 + (4 + i) * 16 + lane] = hi;
                            }
                        const uint32_t h0 = (uint32_t) hd[kg].x(), sl4 = (uint32_t) hd[kg].y();
                        const float d = (float) sycl::bit_cast<sycl::half>(uint16_t(h0 & 0xffff));
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const int ib = 2 * (int) j + b;
                            const int ls = int((sl4 >> (4 * ib)) & 0x0f) | int(((h0 >> (16 + 2 * ib)) & 0x03) << 4);
                            sl[b] = d * (float) (ls - 32);
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            uint32_t lo = nd[kg][i] & 0x0F0F0F0Fu, hi = (nd[kg][i] >> 4) & 0x0F0F0F0Fu;
                            if constexpr (TY == Q5K) {
                                lo |= ((hq[kg][i] >> (2 * j)) & 0x01010101u) << 4;
                                hi |= ((hq[kg][i] >> (2 * j + 1)) & 0x01010101u) << 4;
                            }
                            tl[i * 16 + lane] = lo;
                            tl[128 + i * 16 + lane] = hi;
                        }
                        const uint32_t dmw = (uint32_t) hd[kg].x(), sc[3] = {(uint32_t) hd[kg].y(), (uint32_t) hd[kg].z(), (uint32_t) hd[kg].w()};
                        const float d = (float) sycl::bit_cast<sycl::half>(uint16_t(dmw & 0xffff)), dmin = (float) sycl::bit_cast<sycl::half>(uint16_t(dmw >> 16));
                        float s0, m0, s1, m1;
                        k4_scale_min(sc, 2 * (int) j, s0, m0);
                        k4_scale_min(sc, 2 * (int) j + 1, s1, m1);
                        sl[0] = d * s0; sl[1] = d * s1; ml[0] = dmin * m0; ml[1] = dmin * m1;
                    }
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int b = 0; b < 2; ++b) {
                        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> bm;
                        jm::joint_matrix_load(sg, bm, lpm(reinterpret_cast<int8_t*>(tl + b * 128)), (size_t) 64);
                        jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> cm;
                        jm::joint_matrix_fill(sg, cm, 0);
                        jm::joint_matrix_mad(sg, cm, am[b], bm, cm);
                        int e = 0;
                        const float s = sl[b];
                        jm::joint_matrix_apply(sg, cm, [&](int32_t& v) {
                            acc[e] = sycl::fma((float) v, s * s8[b][e], acc[e]);
                            ++e;
                        });
                        if constexpr (TY == Q4K || TY == Q5K) {
#pragma unroll
                            for (int e2 = 0; e2 < 8; ++e2) accm[e2] = sycl::fma(ml[b], m8[b][e2], accm[e2]);
                        }
                    }
                }
            }
            // element e: token e, the lane's row
            if constexpr (SK == 1) {
                if (row < n_out) {
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        if (e < ncols) y[e * n_out + row] = acc[e] - accm[e];
                }
            } else {
                float* rd = red.get_multi_ptr<sycl::access::decorated::no>().get();
#pragma unroll
                for (int e = 0; e < 8; ++e) rd[(sgi * 8 + e) * 16 + lane] = acc[e] - accm[e];
                sycl::group_barrier(it.get_group());
                for (int i = (int) it.get_local_id(0); i < 8 * 16; i += SK * SG) {
                    const int ln = i % 16, tk = i / 16;
                    const int64_t r = (int64_t) it.get_group(0) * 16 + ln;
                    if (tk >= ncols || r >= n_out) continue;
                    float v = 0.0f;
                    for (int s2 = 0; s2 < SK; ++s2) v += rd[(s2 * 8 + tk) * 16 + ln];
                    y[tk * n_out + r] = v;
                }
            }
        });
    });
}

// Q6_K (210-byte blocks, 2-byte aligned): a step is a 128-weight half of a block - ql (64 bytes), qh (32), eight scales of 16
// weights - as four 32-weight tiles (the four 2-bit-high-part planes of ggml's order). A tile holds two scales, so it is
// two DPAS: the activations with only the lower 16 quants (xa) and only the upper 16 (xb), each its own scale.
inline uint32_t less32(uint32_t v) { return ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u; }   // 4 bytes of 6 bits less 32, as int8
template <int SK, int KG>
void xmx_q6k(sycl::queue& q, const uint8_t* w, const int8_t* xa, const int8_t* xb, const float* ts, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    constexpr int NT = 4;
    const int64_t nch = n_in / 128, rb = (n_in / 256) * 210;
    const int64_t groups = (n_out + 15) / 16;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(SK * 8 * 16), h);
        sycl::local_accessor<uint32_t, 1> tiles(sycl::range<1>(SK * 2 * NT * 128), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SK * SG), sycl::range<1>(SK * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int64_t row = it.get_group(0) * 16 + lane;
            const uint8_t* wr = w + (row < n_out ? row : n_out - 1) * rb;
            uint32_t* tl0 = tiles.get_multi_ptr<sycl::access::decorated::no>().get() + sgi * 2 * NT * 128;
            int par = 0;
            float acc[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
            for (int64_t c0 = (int64_t) sgi * KG; c0 < nch; c0 += SK * KG) {
                sycl::int4 ql[KG][4], qh[KG][2], sc[KG];
                float dd[KG];
#pragma unroll
                for (int kg = 0; kg < KG; ++kg) {
                    const int64_t c = sycl::min(c0 + kg, nch - 1), blk = c / 2, n = c % 2;
                    const uint8_t* b = wr + blk * 210;
#pragma unroll
                    for (int i = 0; i < 4; ++i) ql[kg][i] = load16_a2(b + 64 * n + 16 * i);
#pragma unroll
                    for (int i = 0; i < 2; ++i) qh[kg][i] = load16_a2(b + 128 + 32 * n + 16 * i);
                    sc[kg] = load16_a2(b + 192);
                    dd[kg] = (float) *reinterpret_cast<const sycl::half*>(b + 208);
                }
#pragma unroll
                for (int kg = 0; kg < KG; ++kg) {
                    const int64_t c = c0 + kg;
                    if (c >= nch) break;
                    const int64_t n = c % 2;
                    uint32_t* tl = tl0 + par * NT * 128;
                    par ^= 1;
                    uint32_t qa[8], qb[8], qhd[8];
                    qa[0] = ql[kg][0].x(); qa[1] = ql[kg][0].y(); qa[2] = ql[kg][0].z(); qa[3] = ql[kg][0].w();
                    qa[4] = ql[kg][1].x(); qa[5] = ql[kg][1].y(); qa[6] = ql[kg][1].z(); qa[7] = ql[kg][1].w();
                    qb[0] = ql[kg][2].x(); qb[1] = ql[kg][2].y(); qb[2] = ql[kg][2].z(); qb[3] = ql[kg][2].w();
                    qb[4] = ql[kg][3].x(); qb[5] = ql[kg][3].y(); qb[6] = ql[kg][3].z(); qb[7] = ql[kg][3].w();
                    qhd[0] = qh[kg][0].x(); qhd[1] = qh[kg][0].y(); qhd[2] = qh[kg][0].z(); qhd[3] = qh[kg][0].w();
                    qhd[4] = qh[kg][1].x(); qhd[5] = qh[kg][1].y(); qhd[6] = qh[kg][1].z(); qhd[7] = qh[kg][1].w();
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        tl[0 * 128 + i * 16 + lane] = less32((qa[i] & 0x0F0F0F0Fu) | ((qhd[i] & 0x03030303u) << 4));
                        tl[1 * 128 + i * 16 + lane] = less32((qb[i] & 0x0F0F0F0Fu) | (((qhd[i] >> 2) & 0x03030303u) << 4));
                        tl[2 * 128 + i * 16 + lane] = less32(((qa[i] >> 4) & 0x0F0F0F0Fu) | (((qhd[i] >> 4) & 0x03030303u) << 4));
                        tl[3 * 128 + i * 16 + lane] = less32(((qb[i] >> 4) & 0x0F0F0F0Fu) | (((qhd[i] >> 6) & 0x03030303u) << 4));
                    }
                    // the 16 scales (int8) of the block, four words
                    const uint32_t sw[4] = {(uint32_t) sc[kg].x(), (uint32_t) sc[kg].y(), (uint32_t) sc[kg].z(), (uint32_t) sc[kg].w()};
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const int64_t kb = 4 * c + t;
                        const int si = (int) (8 * n) + 2 * t;
                        const float w0 = dd[kg] * (float) (int8_t) ((sw[si >> 2] >> (8 * (si & 3))) & 0xff);
                        const float w1 = dd[kg] * (float) (int8_t) ((sw[(si + 1) >> 2] >> (8 * ((si + 1) & 3))) & 0xff);
                        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> al, ah;
                        jm::joint_matrix_load(sg, al, gpm(xa + kb * 256), (size_t) 32);
                        jm::joint_matrix_load(sg, ah, gpm(xb + kb * 256), (size_t) 32);
                        const sycl::vec<float, 8> s8 = *reinterpret_cast<const sycl::vec<float, 8>*>(ts + kb * 8);
                        jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> bm;
                        jm::joint_matrix_load(sg, bm, lpm(reinterpret_cast<int8_t*>(tl + t * 128)), (size_t) 64);
                        jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> cl, ch;
                        jm::joint_matrix_fill(sg, cl, 0);
                        jm::joint_matrix_fill(sg, ch, 0);
                        jm::joint_matrix_mad(sg, cl, al, bm, cl);
                        jm::joint_matrix_mad(sg, ch, ah, bm, ch);
                        int e = 0, e2 = 0;
                        jm::joint_matrix_apply(sg, cl, [&](int32_t& v) { acc[e] = sycl::fma((float) v, w0 * s8[e], acc[e]); ++e; });
                        jm::joint_matrix_apply(sg, ch, [&](int32_t& v) { acc[e2] = sycl::fma((float) v, w1 * s8[e2], acc[e2]); ++e2; });
                    }
                }
            }
            float* rd = red.get_multi_ptr<sycl::access::decorated::no>().get();
#pragma unroll
            for (int e = 0; e < 8; ++e) rd[(sgi * 8 + e) * 16 + lane] = acc[e];
            sycl::group_barrier(it.get_group());
            for (int i = (int) it.get_local_id(0); i < 8 * 16; i += SK * SG) {
                const int ln = i % 16, tk = i / 16;
                const int64_t r = (int64_t) it.get_group(0) * 16 + ln;
                if (tk >= ncols || r >= n_out) continue;
                float v = 0.0f;
                for (int s2 = 0; s2 < SK; ++s2) v += rd[(s2 * 8 + tk) * 16 + ln];
                y[tk * n_out + r] = v;
            }
        });
    });
}

template <int TY>
void xmx_shape(int shape, sycl::queue& q, const uint8_t* w, const int8_t* xa, const float* ts, const float* tm, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    // sub-groups a work-group (they split a tile's K) and groups of 64 weights a step. By default 8 and 1 where there are 512 or
    // more 16-row tiles (B70, Q4_K: 10,240 rows 64 us, 17,408 rows 122 us), else 4 and 1 (5,120 rows: down 105 us against 135)
    if (shape == 0) shape = n_in < 8192 && n_out < 8192 ? 4 : (n_out + 15) / 16 >= 512 ? 10 : 9;
    switch (shape) {
        case 1: xmx_mmvq<TY, 4, 2>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        case 3: xmx_mmvq<TY, 8, 4>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        case 4: xmx_mmvq<TY, 16, 1>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        case 5: xmx_mmvq<TY, 16, 2>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        case 6: xmx_mmvq<TY, 4, 4>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        case 7: xmx_mmvq<TY, 2, 2>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        case 8: xmx_mmvq<TY, 2, 4>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        case 9: xmx_mmvq<TY, 4, 1>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        case 10: xmx_mmvq<TY, 8, 1>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
        default: xmx_mmvq<TY, 8, 2>(q, w, xa, ts, tm, y, n_in, n_out, ncols); break;
    }
}

int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : dflt;
}
}  // namespace

extern "C" {

// the split layout's tail (common.hpp) from the split layout of ncols columns
int ns_q35_xmx_prep(ns_gpu* g, const void* xs, int64_t n_in, int64_t ncols, int halves) {
    NS_TRY
    const int8_t* qs = static_cast<const int8_t*>(xs);
    const sycl::half* d = reinterpret_cast<const sycl::half*>(qs + ncols * n_in);
    const int16_t* sm = reinterpret_cast<const int16_t*>(d + ncols * (n_in / 32));
    int8_t* xa = const_cast<int8_t*>(qs) + split_base(n_in, ncols);
    float* ts = reinterpret_cast<float*>(xa + (n_in / 32) * 256);
    float* tm = ts + (n_in / 32) * 8;
    int8_t* xb = reinterpret_cast<int8_t*>(tm + (n_in / 32) * 8);
    const int64_t nb = n_in / 32;
    g->q.parallel_for(sycl::range<1>((size_t) (nb * 8)), [=](sycl::id<1> id) {
        const int64_t b = id[0] / 8, e = id[0] % 8;
        sycl::int4 lo(0, 0, 0, 0), hi(0, 0, 0, 0);
        float dd = 0.0f, sum = 0.0f;
        if (e < ncols) {
            const sycl::int4* src = reinterpret_cast<const sycl::int4*>(qs + e * n_in + 32 * b);
            lo = src[0];
            hi = src[1];
            dd = (float) d[e * nb + b];
            sum = dd * (float) (sm[e * (n_in / 16) + 2 * b] + sm[e * (n_in / 16) + 2 * b + 1]);
        }
        sycl::int4* dst = reinterpret_cast<sycl::int4*>(xa + b * 256 + e * 32);
        dst[0] = lo;
        dst[1] = hi;
        if (halves) {   // Q6_K: xa keeps the lower 16 quants of a block, xb the upper 16 (the other half zero)
            dst[1] = sycl::int4(0, 0, 0, 0);
            sycl::int4* db = reinterpret_cast<sycl::int4*>(xb + b * 256 + e * 32);
            db[0] = sycl::int4(0, 0, 0, 0);
            db[1] = hi;
        }
        ts[b * 8 + e] = dd;
        tm[b * 8 + e] = sum;
    });
    return 0;
    NS_CATCH
}

// the smallest column count routed here (NS_Q35_XMX; 0: never), whether a product is of a type and width it covers, and both
// (the engine asks this for types the dp4a products do not have: Q8_0)
int ns_q35_xmx_min_cols() {
    static const int v = env_int("NS_Q35_XMX", 5);
    return v;
}
int ns_q35_xmx_supported(int type, int64_t n_in, int64_t n_out) {
    static const int min_out = env_int("NS_Q35_XMX_MIN_OUT", 2048), q80 = env_int("NS_Q35_XMX_Q80", 1), q6 = env_int("NS_Q35_XMX_Q6K", 1);
    if (n_in < 4096 || n_out < min_out) return 0;
    if (type == Q80) return q80 && n_in % 64 == 0;
    return (type == Q4K || type == Q5K || type == IQ4XS || (type == Q6K && q6)) && n_in % 256 == 0;
}
int ns_q35_q80_small_covers(int64_t n_in, int64_t n_out, int64_t ncols) {
    static const int on = env_int("NS_Q35_Q80_SMALL", 1);
    return on && ns_q35_xmx_min_cols() > 0 && ncols >= ns_q35_xmx_min_cols() && ncols <= 8 && n_in % 32 == 0 && (n_out < 2048 || n_in < 4096);
}
int ns_q35_xmx_covers(int type, int64_t n_in, int64_t n_out, int64_t ncols) {
    if (ns_q35_xmx_min_cols() <= 0 || ncols < ns_q35_xmx_min_cols() || ncols > 8) return 0;
    return ns_q35_xmx_supported(type, n_in, n_out) || (type == Q80 && ns_q35_q80_small_covers(n_in, n_out, ncols));
}

int ns_q35_xmx_mmvq(ns_gpu* g, int type, const void* w, const void* xs, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (ncols < 1 || ncols > 8 || !ns_q35_xmx_supported(type, n_in, n_out)) return ns_fail("ns_q35_xmx_mmvq: Q4_K, Q5_K or IQ4_XS, 1..8 columns");
    const int rc = ns_q35_xmx_prep(g, xs, n_in, ncols, type == Q6K);
    if (rc != 0) return rc;
    const int8_t* xa = static_cast<const int8_t*>(xs) + split_base(n_in, ncols);
    const float* ts = reinterpret_cast<const float*>(xa + (n_in / 32) * 256);
    const float* tm = ts + (n_in / 32) * 8;
    static const int shape = env_int("NS_Q35_XMX_SHAPE", 0);
    const uint8_t* wb = static_cast<const uint8_t*>(w);
    if (type == Q6K) {
        const int8_t* xb = reinterpret_cast<const int8_t*>(tm + (n_in / 32) * 8);
        static const int sh6 = env_int("NS_Q35_XMX_SHAPE6", 0);
        const int sk = sh6 ? sh6 : n_in < 8192 && n_out < 8192 ? 6 : (n_out + 15) / 16 >= 512 ? 8 : 4;
        if (sk == 4) xmx_q6k<4, 1>(g->q, wb, xa, xb, ts, y, n_in, n_out, ncols);
        else if (sk == 5) xmx_q6k<8, 2>(g->q, wb, xa, xb, ts, y, n_in, n_out, ncols);
        else if (sk == 6) xmx_q6k<16, 1>(g->q, wb, xa, xb, ts, y, n_in, n_out, ncols);
        else xmx_q6k<8, 1>(g->q, wb, xa, xb, ts, y, n_in, n_out, ncols);
        return 0;
    }
    if (type == Q80) xmx_shape<Q80>(shape, g->q, wb, xa, ts, tm, y, n_in, n_out, ncols);
    else if (type == Q4K) xmx_shape<Q4K>(shape, g->q, wb, xa, ts, tm, y, n_in, n_out, ncols);
    else if (type == Q5K) xmx_shape<Q5K>(shape, g->q, wb, xa, ts, tm, y, n_in, n_out, ncols);
    else xmx_shape<IQ4XS>(shape, g->q, wb, xa, ts, tm, y, n_in, n_out, ncols);
    return 0;
    NS_CATCH
}

}  // extern "C"
