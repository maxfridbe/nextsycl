// moe.cpp: the minimaxm3 silo's routed-expert products for a decode token, in ggml's own row types (UD-IQ1_M: gate and up IQ1_M or
// IQ2_XXS, down IQ3_XXS or IQ4_XS) on float32 activations. The scheme is the deepseek41 silo's (std.cpp): a 256-value block is 32
// groups of 8 values, a function makes group g of a block (its scales, signs and grid read for that group alone), a product is a
// sub-group a row with lane g the block's group g, the lanes' sums reduced at the end. What is new here is the shape of the work:
// one launch takes up to four experts at once (the shared kernels take one expert and three launches each), the gate and up rows
// of an output row are worked together and leave as the SwiGLU-OAI value, and the down product writes its weighted row into the
// rank's plane of the sum. The grids are ggml's (ggml-common.h); the formulas are ggml's dequantize_row_*.
#include "ns.h"
#include "ns_internal.hpp"
#include "mm3s.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

namespace {

enum : int { T_F32 = 0, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XXS = 16, T_IQ3_XXS = 18, T_IQ4_XS = 23, T_IQ1_M = 29 };

constexpr int sb_bytes(int type) {
    switch (type) {
        case T_IQ2_XXS: return 66;
        case T_IQ3_XXS: return 98;
        case T_IQ4_XS: return 136;
        case T_IQ1_M: return 56;
        case T_Q5_K: return 176;
        case T_Q6_K: return 210;
        case T_F32: return 1024;
        default: return 0;
    }
}

// blocks are 2-byte aligned only
inline uint32_t ld16(const uint8_t* p) { return (uint32_t) p[0] | ((uint32_t) p[1] << 8); }
inline uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
inline float h2f(const uint8_t* p) { return (float) sycl::bit_cast<sycl::half>((uint16_t) ld16(p)); }
inline uint32_t ksign(uint32_t x) { x &= 127u; return x | ((sycl::popcount(x) & 1u) << 7); }
inline float sgn(uint32_t signs, int j) { return ((signs >> j) & 1u) ? -1.f : 1.f; }

struct Lut {
    const uint64_t* a;
    const uint32_t* b;
};
template <int T> struct Tabs {
    static constexpr int N64 = 0, N32 = 0;
    static uint64_t t64(int) { return 0; }
    static uint32_t t32(int) { return 0; }
};
template <> struct Tabs<T_IQ2_XXS> { static constexpr int N64 = 256, N32 = 0; static uint64_t t64(int i) { return iq2xxs_grid[i]; } static uint32_t t32(int) { return 0; } };
template <> struct Tabs<T_IQ3_XXS> { static constexpr int N64 = 0, N32 = 256; static uint64_t t64(int) { return 0; } static uint32_t t32(int i) { return iq3xxs_grid[i]; } };
template <> struct Tabs<T_IQ1_M> { static constexpr int N64 = 0, N32 = 2048; static uint64_t t64(int) { return 0; } static uint32_t t32(int i) { return iq1s_grid_gpu[i]; } };

template <int T> struct Deq;

// IQ2_XXS: d; 32 u16: per 32 values four grid bytes then 28 bits of signs (4 x 7) and a 4-bit scale
template <> struct Deq<T_IQ2_XXS> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib32 = g >> 2, l = g & 3;
        const uint8_t* q2 = b + 2 + 8 * ib32;
        const uint32_t a0 = ld32(q2), a1 = ld32(q2 + 4);
        const float db = h2f(b) * (0.5f + (a1 >> 28)) * 0.25f;
        const uint8_t* grid = (const uint8_t*) (L.a + ((a0 >> (8 * l)) & 255));
        const uint32_t signs = ksign(a1 >> (7 * l));
        for (int j = 0; j < 8; ++j) w[j] = db * grid[j] * sgn(signs, j);
    }
};

// IQ3_XXS: d; qs[96]: 64 grid indices (two a group of 8), then per 32 values a u32 of 28 sign bits and a 4-bit scale
template <> struct Deq<T_IQ3_XXS> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib32 = g >> 2, l = g & 3;
        const uint8_t* qs = b + 2 + 8 * ib32;
        const uint32_t aux = ld32(b + 2 + 64 + 4 * ib32);
        const float db = h2f(b) * (0.5f + (aux >> 28)) * 0.5f;
        const uint32_t signs = ksign(aux >> (7 * l));
        const uint8_t* g1 = (const uint8_t*) (L.b + qs[2 * l]);
        const uint8_t* g2 = (const uint8_t*) (L.b + qs[2 * l + 1]);
        for (int j = 0; j < 4; ++j) {
            w[j] = db * g1[j] * sgn(signs, j);
            w[j + 4] = db * g2[j] * sgn(signs, j + 4);
        }
    }
};

// IQ4_XS: d; scales_h (u16, 2 bits a 32 values); scales_l[4] (a nibble a 32 values); qs[128] (16 bytes a 32 values: values 0-15 the low
// nibbles, 16-31 the high), a nibble a value of kvalues_iq4nl; the scale (ls - 32) d
template <> struct Deq<T_IQ4_XS> {
    static void group(const uint8_t* b, int g, float* w, const Lut&) {
        const int ib = g >> 2, q = g & 3;
        const uint32_t sh = ld16(b + 2);
        const uint32_t ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = h2f(b) * ((int) ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib + 8 * (q & 1);
        const int s = 4 * (q >> 1);
        for (int j = 0; j < 8; ++j) w[j] = dl * (float) kvalues_iq4nl[(qs[j] >> s) & 15];
    }
};

// IQ1_M: qs[32] (8 index bits a group of 8), qh[16] (a byte for two groups: 3 high index bits and a delta sign each), scales[8] (u16s:
// three 3-bit scales for 16 values each... two a 32 values; the top nibbles of the four u16 make the fp16 d)
template <> struct Deq<T_IQ1_M> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib = g >> 2, l = g & 3;
        const uint32_t s0 = ld16(b + 48), s1 = ld16(b + 50), s2 = ld16(b + 52), s3 = ld16(b + 54);
        const uint32_t db = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
        const float d = (float) sycl::bit_cast<sycl::half>((uint16_t) db);
        const uint32_t sc = ld16(b + 48 + 2 * (ib >> 1));
        const float dl = d * (float) (2 * ((sc >> (6 * (ib & 1) + 3 * (l >> 1))) & 7) + 1);
        const uint32_t qhb = (b[32 + 2 * ib + (l >> 1)] >> (4 * (l & 1)));
        const float delta = (qhb & 8) ? -IQ1S_DELTA : IQ1S_DELTA;
        // SYCL builds have ggml's GPU form of the grid: the entry + 1 in nibbles, values 0-3 the even nibbles, 4-7 the odd
        const uint32_t e = L.b[b[4 * ib + l] | ((qhb & 7) << 8)];
        for (int j = 0; j < 8; ++j) w[j] = dl * ((int) ((e >> (j < 4 ? 8 * j : 8 * (j - 4) + 4)) & 0xF) - 1 + delta);
    }
};

// float32 rows (the indexer's): a group is 8 values as they are
template <> struct Deq<T_F32> {
    static void group(const uint8_t* b, int g, float* w, const Lut&) {
        const sycl::float4* p = reinterpret_cast<const sycl::float4*>(b + 32 * g);
        const sycl::float4 a = p[0], c = p[1];
        w[0] = a.x(); w[1] = a.y(); w[2] = a.z(); w[3] = a.w(); w[4] = c.x(); w[5] = c.y(); w[6] = c.z(); w[7] = c.w();
    }
};

// Q5_K and Q6_K: ggml's dequantize_row_q5_k / q6_k, group g of a block (the deepseek41 silo's)
template <> struct Deq<T_Q5_K> {
    static void group(const uint8_t* b, int g, float* w, const Lut&) {
        const int v = g * 8, j = v >> 6, u = v & 63, hi = u >> 5, l0 = u & 31;
        const int is = 2 * j + hi;
        const uint8_t* q = b + 4;
        const uint32_t sc = is < 4 ? (q[is] & 63) : ((q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4));
        const uint32_t mn = is < 4 ? (q[is + 4] & 63) : ((q[is + 4] >> 4) | ((q[is] >> 6) << 4));
        const float d1 = h2f(b) * sc, m1 = h2f(b + 2) * mn;
        const uint8_t* qh = b + 16 + l0;
        const uint8_t* qs = b + 48 + 32 * j + l0;
        const int bit = 1 << is;
        for (int k = 0; k < 8; ++k) w[k] = d1 * ((hi ? (qs[k] >> 4) : (qs[k] & 0xF)) + ((qh[k] & bit) ? 16 : 0)) - m1;
    }
};
template <> struct Deq<T_Q6_K> {
    static void group(const uint8_t* b, int g, float* w, const Lut&) {
        const int v = g * 8, n = v >> 7, u = v & 127, quarter = u >> 5, l0 = u & 31;
        const float d = h2f(b + 208);
        const float sc = d * (float) (int8_t) b[192 + n * 8 + (l0 >> 4) + 2 * quarter];
        const uint8_t* ql = b + n * 64 + (quarter & 1) * 32 + l0;
        const uint8_t* qh = b + 128 + n * 32 + l0;
        for (int k = 0; k < 8; ++k) {
            const int lo = quarter < 2 ? (ql[k] & 0xF) : (ql[k] >> 4);
            w[k] = sc * (float) ((lo | (((qh[k] >> (2 * quarter)) & 3) << 4)) - 32);
        }
    }
};

constexpr int SG = 32, RPG = 4;

// y_m[r] = m_m[r] . x for up to three matrices of one type and one width in a launch (a layer's q, k, v), rows of each a multiple of RPG * RPS
template <int T, int RPS>
void mv3(sycl::queue& q, const uint8_t* m0, const uint8_t* m1, const uint8_t* m2, const float* x, float* y0, float* y1, float* y2, int64_t r0n, int64_t r1n, int64_t r2n, int64_t cols) {
    constexpr int SB = sb_bytes(T), N64 = Tabs<T>::N64, N32 = Tabs<T>::N32;
    const int64_t total = r0n + r1n + r2n, nb = cols / 256, rowb = nb * SB;
    const int64_t rgs = (total + RPG * RPS - 1) / (RPG * RPS);
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<uint64_t, 1> l64(sycl::range<1>(N64 > 0 ? N64 : 1), hd);
        sycl::local_accessor<uint32_t, 1> l32(sycl::range<1>(N32 > 0 ? N32 : 1), hd);
        hd.parallel_for(sycl::nd_range<1>(rgs * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            for (int i = it.get_local_linear_id(); i < N64; i += RPG * SG) l64[i] = Tabs<T>::t64(i);
            for (int i = it.get_local_linear_id(); i < N32; i += RPG * SG) l32[i] = Tabs<T>::t32(i);
            if (N64 > 0 || N32 > 0) sycl::group_barrier(it.get_group());
            const Lut L{l64.template get_multi_ptr<sycl::access::decorated::no>().get(), l32.template get_multi_ptr<sycl::access::decorated::no>().get()};
            const int64_t r00 = (it.get_group(0) * RPG + sg.get_group_id()[0]) * RPS;
            const int lane = sg.get_local_id()[0];
            if (r00 >= total) return;
            const bool in0 = r00 < r0n, in1 = !in0 && r00 < r0n + r1n;
            const uint8_t* base = in0 ? m0 : in1 ? m1 : m2;
            float* yo = in0 ? y0 : in1 ? y1 : y2;
            const int64_t rs = in0 ? r00 : in1 ? r00 - r0n : r00 - r0n - r1n;
            const int64_t rn = in0 ? r0n : in1 ? r1n : r2n;
            float acc[RPS] = {};
            for (int64_t k = 0; k < nb; ++k) {
                float w[RPS][8];
#pragma unroll
                for (int j = 0; j < RPS; ++j) {
                    const int64_t r = rs + j < rn ? rs + j : rn - 1;
                    Deq<T>::group(base + r * rowb + k * SB, lane, w[j], L);
                }
                const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(x + k * 256 + lane * 8);
                const sycl::float4 a = xv[0], b = xv[1];
#pragma unroll
                for (int j = 0; j < RPS; ++j)
                    acc[j] += w[j][0] * a.x() + w[j][1] * a.y() + w[j][2] * a.z() + w[j][3] * a.w() + w[j][4] * b.x() + w[j][5] * b.y() + w[j][6] * b.z() + w[j][7] * b.w();
            }
#pragma unroll
            for (int j = 0; j < RPS; ++j) {
                const float v = sycl::reduce_over_group(sg, acc[j], sycl::plus<float>());
                if (lane == 0 && rs + j < rn) yo[rs + j] = v;
            }
        });
    });
}

// h[e * rows + r] = swiglu_oai(gate_e[r] . x, up_e[r] . x) for the entries' experts: bases[e] the expert's gate matrix, the up matrix
// `off_up` bytes after it. A sub-group works RPS output rows (both of their matrices) at once.
template <int T, int RPS, int KU>
void gu(sycl::queue& q, const uint64_t* bases, int64_t off_up, const float* x, float* h, int64_t entries, int64_t rows, int64_t cols, float alpha, float limit) {
    constexpr int SB = sb_bytes(T), N64 = Tabs<T>::N64, N32 = Tabs<T>::N32;
    const int64_t rgs = (rows + RPG * RPS - 1) / (RPG * RPS), nb = cols / 256, rowb = nb * SB;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<uint64_t, 1> l64(sycl::range<1>(N64 > 0 ? N64 : 1), hd);
        sycl::local_accessor<uint32_t, 1> l32(sycl::range<1>(N32 > 0 ? N32 : 1), hd);
        hd.parallel_for(sycl::nd_range<1>(entries * rgs * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            for (int i = it.get_local_linear_id(); i < N64; i += RPG * SG) l64[i] = Tabs<T>::t64(i);
            for (int i = it.get_local_linear_id(); i < N32; i += RPG * SG) l32[i] = Tabs<T>::t32(i);
            if (N64 > 0 || N32 > 0) sycl::group_barrier(it.get_group());
            const Lut L{l64.template get_multi_ptr<sycl::access::decorated::no>().get(), l32.template get_multi_ptr<sycl::access::decorated::no>().get()};
            const int64_t gi = it.get_group(0), e = gi / rgs, r0 = ((gi % rgs) * RPG + sg.get_group_id()[0]) * RPS;
            const int lane = sg.get_local_id()[0];
            if (r0 >= rows) return;
            const uint8_t* gate = reinterpret_cast<const uint8_t*>(bases[e]);
            const uint8_t* up = gate + off_up;
            float ag[RPS] = {}, au[RPS] = {};
            for (int64_t k0 = 0; k0 < nb; k0 += KU) {
                // KU blocks of RPS rows (gate and up) decoded first, their loads in flight together, then multiplied
                float wg[KU][RPS][8], wu[KU][RPS][8];
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    const int64_t k = k0 + u < nb ? k0 + u : nb - 1;
#pragma unroll
                    for (int j = 0; j < RPS; ++j) {
                        const int64_t r = r0 + j < rows ? r0 + j : rows - 1;
                        Deq<T>::group(gate + r * rowb + k * SB, lane, wg[u][j], L);
                        Deq<T>::group(up + r * rowb + k * SB, lane, wu[u][j], L);
                    }
                }
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    if (k0 + u >= nb) break;
                    const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(x + (k0 + u) * 256 + lane * 8);
                    const sycl::float4 a = xv[0], b = xv[1];
#pragma unroll
                    for (int j = 0; j < RPS; ++j) {
                        ag[j] += wg[u][j][0] * a.x() + wg[u][j][1] * a.y() + wg[u][j][2] * a.z() + wg[u][j][3] * a.w() + wg[u][j][4] * b.x() + wg[u][j][5] * b.y() + wg[u][j][6] * b.z() + wg[u][j][7] * b.w();
                        au[j] += wu[u][j][0] * a.x() + wu[u][j][1] * a.y() + wu[u][j][2] * a.z() + wu[u][j][3] * a.w() + wu[u][j][4] * b.x() + wu[u][j][5] * b.y() + wu[u][j][6] * b.z() + wu[u][j][7] * b.w();
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < RPS; ++j) {
                const float sgt = sycl::reduce_over_group(sg, ag[j], sycl::plus<float>());
                const float sup = sycl::reduce_over_group(sg, au[j], sycl::plus<float>());
                if (lane == 0 && r0 + j < rows) {
                    const float xg = sycl::fmin(sgt, limit);
                    const float yu = sycl::fmin(sycl::fmax(sup, -limit), limit);
                    h[e * rows + r0 + j] = (xg / (1.f + sycl::exp(alpha * (-xg)))) * (yu + 1.f);
                }
            }
        });
    });
}

// ys[rank[e] * plane + r] = w[e] * (down_e[r] . h_e) with h_e = h + e * cols
template <int T, int RPS, int KU>
void down(sycl::queue& q, const uint64_t* bases, int64_t off_down, const float* h, const float* w, const int32_t* rank, float* ys, int64_t plane, int64_t entries, int64_t rows, int64_t cols) {
    constexpr int SB = sb_bytes(T), N64 = Tabs<T>::N64, N32 = Tabs<T>::N32;
    const int64_t rgs = (rows + RPG * RPS - 1) / (RPG * RPS), nb = cols / 256, rowb = nb * SB;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<uint64_t, 1> l64(sycl::range<1>(N64 > 0 ? N64 : 1), hd);
        sycl::local_accessor<uint32_t, 1> l32(sycl::range<1>(N32 > 0 ? N32 : 1), hd);
        hd.parallel_for(sycl::nd_range<1>(entries * rgs * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            for (int i = it.get_local_linear_id(); i < N64; i += RPG * SG) l64[i] = Tabs<T>::t64(i);
            for (int i = it.get_local_linear_id(); i < N32; i += RPG * SG) l32[i] = Tabs<T>::t32(i);
            if (N64 > 0 || N32 > 0) sycl::group_barrier(it.get_group());
            const Lut L{l64.template get_multi_ptr<sycl::access::decorated::no>().get(), l32.template get_multi_ptr<sycl::access::decorated::no>().get()};
            const int64_t gi = it.get_group(0), e = gi / rgs, r0 = ((gi % rgs) * RPG + sg.get_group_id()[0]) * RPS;
            const int lane = sg.get_local_id()[0];
            if (r0 >= rows) return;
            const uint8_t* base = reinterpret_cast<const uint8_t*>(bases[e]) + off_down;
            const float* he = h + e * cols;
            float acc[RPS] = {};
            for (int64_t k0 = 0; k0 < nb; k0 += KU) {
                float wd[KU][RPS][8];
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    const int64_t k = k0 + u < nb ? k0 + u : nb - 1;
#pragma unroll
                    for (int j = 0; j < RPS; ++j) {
                        const int64_t r = r0 + j < rows ? r0 + j : rows - 1;
                        Deq<T>::group(base + r * rowb + k * SB, lane, wd[u][j], L);
                    }
                }
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    if (k0 + u >= nb) break;
                    const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(he + (k0 + u) * 256 + lane * 8);
                    const sycl::float4 a = xv[0], b = xv[1];
#pragma unroll
                    for (int j = 0; j < RPS; ++j)
                        acc[j] += wd[u][j][0] * a.x() + wd[u][j][1] * a.y() + wd[u][j][2] * a.z() + wd[u][j][3] * a.w() + wd[u][j][4] * b.x() + wd[u][j][5] * b.y() + wd[u][j][6] * b.z() + wd[u][j][7] * b.w();
                }
            }
#pragma unroll
            for (int j = 0; j < RPS; ++j) {
                const float s = sycl::reduce_over_group(sg, acc[j], sycl::plus<float>());
                if (lane == 0 && r0 + j < rows) ys[(int64_t) rank[e] * plane + r0 + j] = w[e] * s;
            }
        });
    });
}

}  // namespace

extern "C" {

int ns_mm3s_expert_supported(int type, int64_t cols) { return sb_bytes(type) != 0 && cols > 0 && cols % 256 == 0; }

int ns_mm3s_mv3(ns_gpu* g, int type, const void* m0, const void* m1, const void* m2, const float* x, float* y0, float* y1, float* y2, int64_t r0, int64_t r1, int64_t r2, int64_t cols) {
    NS_TRY
    if (!ns_mm3s_expert_supported(type, cols)) return ns_fail("ns_mm3s_mv3: type " + std::to_string(type) + " with cols " + std::to_string(cols));
    if ((r0 % (RPG * 2)) || (r1 % (RPG * 2))) return ns_fail("ns_mm3s_mv3: rows of the first matrices");
    const auto* a = (const uint8_t*) m0; const auto* b = (const uint8_t*) m1; const auto* c = (const uint8_t*) m2;
#define MV(T) mv3<T, 2>(g->q, a, b, c, x, y0, y1, y2, r0, r1, r2, cols); return 0;
    switch (type) {
        case T_F32: MV(T_F32)
        case T_Q5_K: MV(T_Q5_K)
        case T_Q6_K: MV(T_Q6_K)
        case T_IQ2_XXS: MV(T_IQ2_XXS)
        case T_IQ1_M: MV(T_IQ1_M)
        case T_IQ3_XXS: MV(T_IQ3_XXS)
        case T_IQ4_XS: MV(T_IQ4_XS)
        default: return ns_fail("ns_mm3s_mv3: type");
    }
#undef MV
    NS_CATCH
}

// After the projections of a decode token: q and k (and the indexer's q and k) RMS-normed per head and rotated (NEOX), k, v and the indexer's k stored as fp16,
// all in a launch, a work-group a head
int ns_mm3s_attn_post(ns_gpu* g, float* q, float* k, const float* v, float* iq, float* ik, const float* qn, const float* kn, const float* iqn, const float* ikn, uint16_t* kc, uint16_t* vc, uint16_t* ic,
                      int64_t H, int64_t G, int64_t IH, int64_t HD, int64_t n_rot, int64_t pos, float base, float eps) {
    NS_TRY
    if (n_rot % 2 || n_rot > HD) return ns_fail("ns_mm3s_attn_post: n_rot");
    const int64_t half = n_rot / 2, heads = H + 2 * G + (IH > 0 ? IH + 1 : 0);
    const float ts = powf(base, -2.0f / (float) n_rot);
    g->q.parallel_for(sycl::nd_range<1>((size_t) (heads * HD), (size_t) HD), [=](sycl::nd_item<1> it) {
        const int64_t gr = it.get_group(0);
        const int lid = (int) it.get_local_id(0);
        float* p;
        const float* w = nullptr;
        uint16_t* store = nullptr;
        if (gr < H) { p = q + gr * HD; w = qn; }
        else if (gr < H + G) { p = k + (gr - H) * HD; w = kn; store = kc + (gr - H) * HD; }
        else if (gr < H + 2 * G) { p = const_cast<float*>(v) + (gr - H - G) * HD; store = vc + (gr - H - G) * HD; }
        else if (gr < H + 2 * G + IH) { p = iq + (gr - H - 2 * G) * HD; w = iqn; }
        else { p = ik; w = ikn; store = ic; }
        float out = p[lid];
        if (w) {
            float sq = out * out;
            sq = sycl::reduce_over_group(it.get_group(), sq, sycl::plus<float>());
            const float inv = 1.f / sycl::sqrt(sq / (float) HD + eps);
            const bool rot = lid < 2 * half;
            if (rot) {
                const int i = lid % half;
                const int other = lid < half ? lid + half : lid - half;
                float theta = (float) pos;
                for (int j = 0; j < i; ++j) theta *= ts;
                const float c = sycl::cos(theta), s = sycl::sin(theta);
                const float me = out * inv * w[lid], them = p[other] * inv * w[other];
                out = lid < half ? me * c - them * s : them * s + me * c;
            } else {
                out = out * inv * w[lid];
            }
        }
        sycl::group_barrier(it.get_group());
        if (w) p[lid] = out;
        if (store) reinterpret_cast<sycl::half*>(store)[lid] = (sycl::half) out;
    });
    return 0;
    NS_CATCH
}

// att = o + x; hn = rms_norm(att) * w, in a launch
int ns_mm3s_add_norm(ns_gpu* g, float* att, const float* o, const float* x, const float* w, float* hn, int64_t C, float eps) {
    NS_TRY
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>((size_t) WG, WG), [=](sycl::nd_item<1> it) {
        float s = 0.f;
        const int lid = (int) it.get_local_id(0);
        for (int64_t c = lid; c < C; c += WG) { const float v = o[c] + x[c]; s += v * v; }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = 1.f / sycl::sqrt(s / (float) C + eps);
        for (int64_t c = lid; c < C; c += WG) { const float v = o[c] + x[c]; att[c] = v; hn[c] = v * inv * w[c]; }
    });
    return 0;
    NS_CATCH
}

// out = the rank planes summed in rank order + sh + att
int ns_mm3s_finish(ns_gpu* g, const float* ys, int64_t used, const float* sh, const float* att, float* out, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = id[0];
        float a = ys[c];
        for (int64_t r = 1; r < used; ++r) a += ys[r * C + c];
        out[c] = a + sh[c] + att[c];
    });
    return 0;
    NS_CATCH
}

int ns_mm3s_expert_gu(ns_gpu* g, int type, const uint64_t* bases, int64_t off_up, const float* x, float* h, int64_t entries, int64_t rows, int64_t cols, float alpha,
                      float limit) {
    NS_TRY
    if (!ns_mm3s_expert_supported(type, cols)) return ns_fail("ns_mm3s_expert_gu: type " + std::to_string(type) + " with cols " + std::to_string(cols));
    static const int rps = [] { const char* v = std::getenv("NS_MM3S_RPS"); return v ? std::atoi(v) : 2; }();
    static const int ku = [] { const char* v = std::getenv("NS_MM3S_KU"); return v ? std::atoi(v) : 1; }();
#define GU(T, R, K) gu<T, R, K>(g->q, bases, off_up, x, h, entries, rows, cols, alpha, limit)
#define RUN(T) if (rps == 1) { if (ku == 2) GU(T, 1, 2); else GU(T, 1, 1); } else if (rps == 4) { if (ku == 2) GU(T, 4, 2); else GU(T, 4, 1); } else { if (ku == 2) GU(T, 2, 2); else GU(T, 2, 1); } return 0;
    switch (type) {
        case T_IQ2_XXS: RUN(T_IQ2_XXS)
        case T_IQ1_M: RUN(T_IQ1_M)
        case T_IQ3_XXS: RUN(T_IQ3_XXS)
        case T_IQ4_XS: RUN(T_IQ4_XS)
        case T_Q5_K: RUN(T_Q5_K)
        case T_Q6_K: RUN(T_Q6_K)
        default: return ns_fail("ns_mm3s_expert_gu: type");
    }
#undef RUN
    NS_CATCH
}

int ns_mm3s_expert_down(ns_gpu* g, int type, const uint64_t* bases, int64_t off_down, const float* h, const float* w, const int32_t* rank, float* ys, int64_t plane, int64_t entries,
                        int64_t rows, int64_t cols) {
    NS_TRY
    if (!ns_mm3s_expert_supported(type, cols)) return ns_fail("ns_mm3s_expert_down: type " + std::to_string(type) + " with cols " + std::to_string(cols));
    static const int rps = [] { const char* v = std::getenv("NS_MM3S_RPS"); return v ? std::atoi(v) : 2; }();
    static const int ku = [] { const char* v = std::getenv("NS_MM3S_KU"); return v ? std::atoi(v) : 1; }();
#define DN(T, R, K) down<T, R, K>(g->q, bases, off_down, h, w, rank, ys, plane, entries, rows, cols)
#define RUN(T) if (rps == 1) { if (ku == 2) DN(T, 1, 2); else DN(T, 1, 1); } else if (rps == 4) { if (ku == 2) DN(T, 4, 2); else DN(T, 4, 1); } else { if (ku == 2) DN(T, 2, 2); else DN(T, 2, 1); } return 0;
    switch (type) {
        case T_IQ2_XXS: RUN(T_IQ2_XXS)
        case T_IQ1_M: RUN(T_IQ1_M)
        case T_IQ3_XXS: RUN(T_IQ3_XXS)
        case T_IQ4_XS: RUN(T_IQ4_XS)
        case T_Q5_K: RUN(T_Q5_K)
        case T_Q6_K: RUN(T_Q6_K)
        default: return ns_fail("ns_mm3s_expert_down: type");
    }
#undef RUN
    NS_CATCH
}

}  // extern "C"
