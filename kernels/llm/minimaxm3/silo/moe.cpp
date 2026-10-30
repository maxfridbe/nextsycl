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

#include <chrono>
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

// Blocks may be 2-byte aligned. Small global loads are a message each on Xe2 (a lane of a sub-group reads in step with the others): a lane's bytes come from one or two
// word loads, then shifts (Strata's lesson: check the loads first). ld32 is a word at any 2-byte address; ld8w the 8 bytes at one, as two words; the words beyond a block stay
// inside the allocation (matrices are padded by 64 bytes, experts sit in 256-byte slots).
inline uint32_t ld32(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    if (a & 2) {
        const sycl::uint2 v = *reinterpret_cast<const sycl::uint2*>(q);
        return (v.x() >> 16) | (v.y() << 16);
    }
    return q[0];
}
inline uint32_t ld16(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    return (*reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3)) >> (8 * (a & 3))) & 0xffffu;
}
inline void ld8w(const uint8_t* p, uint32_t& lo, uint32_t& hi) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::uint2 v = *reinterpret_cast<const sycl::uint2*>(a & ~uintptr_t(3));
    if (a & 2) {
        const uint32_t x = *reinterpret_cast<const uint32_t*>((a & ~uintptr_t(3)) + 8);
        lo = (v.x() >> 16) | (v.y() << 16);
        hi = (v.y() >> 16) | (x << 16);
    } else {
        lo = v.x();
        hi = v.y();
    }
}
inline float h2f16(uint32_t h) { return (float) sycl::bit_cast<sycl::half>((uint16_t) (h & 0xffffu)); }
inline float h2f(const uint8_t* p) { return h2f16(ld16(p)); }
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

// kvalues_iq4nl, little end first: low 8 values, then the high 8
constexpr uint64_t iq4_word(int from) {
    constexpr int8_t K[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t) (uint8_t) K[from + i] << (8 * i);
    return v;
}
constexpr uint64_t IQ4_LO = iq4_word(0), IQ4_HI = iq4_word(8);

// IQ2_XXS: d; 32 u16: per 32 values four grid bytes then 28 bits of signs (4 x 7) and a 4-bit scale
template <> struct Deq<T_IQ2_XXS> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib32 = g >> 2, l = g & 3;
        uint32_t a0, a1;
        ld8w(b + 2 + 8 * ib32, a0, a1);
        const float db = h2f(b) * (0.5f + (a1 >> 28)) * 0.25f;
        const sycl::uint2 ge = *reinterpret_cast<const sycl::uint2*>(L.a + ((a0 >> (8 * l)) & 255));
        const uint32_t signs = ksign(a1 >> (7 * l));
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = db * (float) ((ge.x() >> (8 * j)) & 255) * sgn(signs, j);
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j + 4] = db * (float) ((ge.y() >> (8 * j)) & 255) * sgn(signs, j + 4);
    }
};

// IQ3_XXS: d; qs[96]: 64 grid indices (two a group of 8), then per 32 values a u32 of 28 sign bits and a 4-bit scale
template <> struct Deq<T_IQ3_XXS> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib32 = g >> 2, l = g & 3;
        const uint32_t idx = ld16(b + 2 + 8 * ib32 + 2 * l);
        const uint32_t aux = ld32(b + 2 + 64 + 4 * ib32);
        const float db = h2f(b) * (0.5f + (aux >> 28)) * 0.5f;
        const uint32_t signs = ksign(aux >> (7 * l));
        const uint32_t g1 = L.b[idx & 255], g2 = L.b[idx >> 8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j] = db * (float) ((g1 >> (8 * j)) & 255) * sgn(signs, j);
            w[j + 4] = db * (float) ((g2 >> (8 * j)) & 255) * sgn(signs, j + 4);
        }
    }
};

// IQ4_XS: d; scales_h (u16, 2 bits a 32 values); scales_l[4] (a nibble a 32 values); qs[128] (16 bytes a 32 values: values 0-15 the low
// nibbles, 16-31 the high), a nibble a value of kvalues_iq4nl; the scale (ls - 32) d
template <> struct Deq<T_IQ4_XS> {
    static void group(const uint8_t* b, int g, float* w, const Lut&) {
        const int ib = g >> 2, q = g & 3;
        const sycl::uint2 hd = *reinterpret_cast<const sycl::uint2*>(b);  // d, scales_h, scales_l[4]; 136-byte blocks: 8-aligned
        const uint32_t sh = hd.x() >> 16;
        const uint32_t ls = ((hd.y() >> (8 * (ib >> 1)) >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = h2f16(hd.x()) * ((int) ls - 32);
        const sycl::uint2 qq = *reinterpret_cast<const sycl::uint2*>(b + 8 + 16 * ib + 8 * (q & 1));
        const int s = 4 * (q >> 1);
        // kvalues_iq4nl as two words of 16 int8 in nibble order, sign extended below
        auto kv = [](uint32_t n) { const int8_t t = (int8_t) ((n < 8 ? IQ4_LO : IQ4_HI) >> (8 * (n & 7))); return (float) t; };
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = dl * kv((qq.x() >> (8 * j + s)) & 15);
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j + 4] = dl * kv((qq.y() >> (8 * j + s)) & 15);
    }
};

// IQ1_M: qs[32] (8 index bits a group of 8), qh[16] (a byte for two groups: 3 high index bits and a delta sign each), scales[8] (u16s:
// three 3-bit scales for 16 values each... two a 32 values; the top nibbles of the four u16 make the fp16 d)
template <> struct Deq<T_IQ1_M> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib = g >> 2, l = g & 3;
        const sycl::uint2 sv = *reinterpret_cast<const sycl::uint2*>(b + 48);  // the four scale u16s; 56-byte blocks: 8-aligned
        const uint32_t s0 = sv.x() & 0xffffu, s1 = sv.x() >> 16, s2 = sv.y() & 0xffffu, s3 = sv.y() >> 16;
        const uint32_t db = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
        const float d = (float) sycl::bit_cast<sycl::half>((uint16_t) db);
        const int si = ib >> 1;
        const uint32_t sc = (si & 2 ? sv.y() : sv.x()) >> (16 * (si & 1));
        const float dl = d * (float) (2 * ((sc >> (6 * (ib & 1) + 3 * (l >> 1))) & 7) + 1);
        const int qo = 2 * ib + (l >> 1);
        const uint32_t qhw = *reinterpret_cast<const uint32_t*>(b + 32 + (qo & ~3));
        const uint32_t qhb = (qhw >> (8 * (qo & 3))) >> (4 * (l & 1));
        const float delta = (qhb & 8) ? -IQ1S_DELTA : IQ1S_DELTA;
        // SYCL builds have ggml's GPU form of the grid: the entry + 1 in nibbles, values 0-3 the even nibbles, 4-7 the odd
        const int qi = 4 * ib + l;
        const uint32_t qw = *reinterpret_cast<const uint32_t*>(b + (qi & ~3));
        const uint32_t e = L.b[((qw >> (8 * (qi & 3))) & 255) | ((qhb & 7) << 8)];
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
        const sycl::uint4 hd = *reinterpret_cast<const sycl::uint4*>(b);  // d, dmin, 12 scale bytes; 176-byte blocks: 16-aligned
        const uint32_t sw[3] = {hd.y(), hd.z(), hd.w()};
        auto qb = [&](int i) { return (sw[i >> 2] >> (8 * (i & 3))) & 0xffu; };
        const uint32_t sc = is < 4 ? (qb(is) & 63) : ((qb(is + 4) & 0xF) | ((qb(is - 4) >> 6) << 4));
        const uint32_t mn = is < 4 ? (qb(is + 4) & 63) : ((qb(is + 4) >> 4) | ((qb(is) >> 6) << 4));
        const float d1 = h2f16(hd.x()) * sc, m1 = h2f16(hd.x() >> 16) * mn;
        const sycl::uint2 qh = *reinterpret_cast<const sycl::uint2*>(b + 16 + l0);
        const sycl::uint2 qs = *reinterpret_cast<const sycl::uint2*>(b + 48 + 32 * j + l0);
        const int s = hi ? 4 : 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t h0 = (qh.x() >> (8 * k + is)) & 1u, h1 = (qh.y() >> (8 * k + is)) & 1u;
            w[k] = d1 * (float) (((qs.x() >> (8 * k + s)) & 0xF) + (h0 << 4)) - m1;
            w[k + 4] = d1 * (float) (((qs.y() >> (8 * k + s)) & 0xF) + (h1 << 4)) - m1;
        }
    }
};
template <> struct Deq<T_Q6_K> {
    static void group(const uint8_t* b, int g, float* w, const Lut&) {
        const int v = g * 8, n = v >> 7, u = v & 127, quarter = u >> 5, l0 = u & 31;
        const uintptr_t a = reinterpret_cast<uintptr_t>(b);
        const uint32_t ds = *reinterpret_cast<const uint32_t*>((a + 208) & ~uintptr_t(3));
        const float d = h2f16(ds >> (8 * ((a + 208) & 3)));
        const int so = 192 + n * 8 + (l0 >> 4) + 2 * quarter;
        const uint32_t sw = *reinterpret_cast<const uint32_t*>((a + so) & ~uintptr_t(3));
        const float sc = d * (float) (int8_t) (sw >> (8 * ((a + so) & 3)));
        uint32_t ql0, ql1, qh0, qh1;
        ld8w(b + n * 64 + (quarter & 1) * 32 + l0, ql0, ql1);
        ld8w(b + 128 + n * 32 + l0, qh0, qh1);
        const int ls = quarter < 2 ? 0 : 4, hs = 2 * quarter;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            w[k] = sc * (float) ((int) (((ql0 >> (8 * k + ls)) & 0xF) | (((qh0 >> (8 * k + hs)) & 3) << 4)) - 32);
            w[k + 4] = sc * (float) ((int) (((ql1 >> (8 * k + ls)) & 0xF) | (((qh1 >> (8 * k + hs)) & 3) << 4)) - 32);
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
void gu(sycl::queue& q, const uint64_t* bases, int64_t off_up, const float* x, const int32_t* tok, float* h, int64_t entries, int64_t rows, int64_t cols, float alpha, float limit) {
    // a work-group fills its table once: more rows a group, fewer table reads (IQ1_M: 8 KB a group, against 45 KB of weights a row pair)
    constexpr int SB = sb_bytes(T), N64 = Tabs<T>::N64, N32 = Tabs<T>::N32, RG = (N64 > 0 || N32 > 0) ? 16 : 4;
    const int64_t rgs = (rows + RG * RPS - 1) / (RG * RPS), nb = cols / 256, rowb = nb * SB;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<uint64_t, 1> l64(sycl::range<1>(N64 > 0 ? N64 : 1), hd);
        sycl::local_accessor<uint32_t, 1> l32(sycl::range<1>(N32 > 0 ? N32 : 1), hd);
        hd.parallel_for(sycl::nd_range<1>(entries * rgs * RG * SG, RG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            for (int i = it.get_local_linear_id(); i < N64; i += RG * SG) l64[i] = Tabs<T>::t64(i);
            for (int i = it.get_local_linear_id(); i < N32; i += RG * SG) l32[i] = Tabs<T>::t32(i);
            if (N64 > 0 || N32 > 0) sycl::group_barrier(it.get_group());
            const Lut L{l64.template get_multi_ptr<sycl::access::decorated::no>().get(), l32.template get_multi_ptr<sycl::access::decorated::no>().get()};
            const int64_t gi = it.get_group(0), e = gi / rgs, r0 = ((gi % rgs) * RG + sg.get_group_id()[0]) * RPS;
            const int lane = sg.get_local_id()[0];
            if (r0 >= rows) return;
            const uint8_t* gate = reinterpret_cast<const uint8_t*>(bases[e]);
            const uint8_t* up = gate + off_up;
            const float* xr = x + (int64_t) tok[e] * cols;
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
                    const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(xr + (k0 + u) * 256 + lane * 8);
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
void down(sycl::queue& q, const uint64_t* bases, int64_t off_down, const float* h, const float* w, const int32_t* rank, const int32_t* tok, float* ys, int64_t plane, int64_t entries, int64_t rows, int64_t cols) {
    // a work-group fills its table once: more rows a group, fewer table reads (IQ1_M: 8 KB a group, against 45 KB of weights a row pair)
    constexpr int SB = sb_bytes(T), N64 = Tabs<T>::N64, N32 = Tabs<T>::N32, RG = (N64 > 0 || N32 > 0) ? 16 : 4;
    const int64_t rgs = (rows + RG * RPS - 1) / (RG * RPS), nb = cols / 256, rowb = nb * SB;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<uint64_t, 1> l64(sycl::range<1>(N64 > 0 ? N64 : 1), hd);
        sycl::local_accessor<uint32_t, 1> l32(sycl::range<1>(N32 > 0 ? N32 : 1), hd);
        hd.parallel_for(sycl::nd_range<1>(entries * rgs * RG * SG, RG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            for (int i = it.get_local_linear_id(); i < N64; i += RG * SG) l64[i] = Tabs<T>::t64(i);
            for (int i = it.get_local_linear_id(); i < N32; i += RG * SG) l32[i] = Tabs<T>::t32(i);
            if (N64 > 0 || N32 > 0) sycl::group_barrier(it.get_group());
            const Lut L{l64.template get_multi_ptr<sycl::access::decorated::no>().get(), l32.template get_multi_ptr<sycl::access::decorated::no>().get()};
            const int64_t gi = it.get_group(0), e = gi / rgs, r0 = ((gi % rgs) * RG + sg.get_group_id()[0]) * RPS;
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
                if (lane == 0 && r0 + j < rows) ys[(int64_t) rank[e] * plane + (int64_t) tok[e] * rows + r0 + j] = w[e] * s;
            }
        });
    });
}

// out[nb][256] = the blocks as fp16 (a prompt pass's experts, for the half GEMM): a sub-group a block, each lane its 8 values as one 16-byte store
template <int T>
void to_half(sycl::queue& q, const uint8_t* src, uint16_t* out, int64_t nb) {
    constexpr int SB = sb_bytes(T), N64 = Tabs<T>::N64, N32 = Tabs<T>::N32, BPG = 32;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<uint64_t, 1> l64(sycl::range<1>(N64 > 0 ? N64 : 1), hd);
        sycl::local_accessor<uint32_t, 1> l32(sycl::range<1>(N32 > 0 ? N32 : 1), hd);
        const int64_t groups = (nb + BPG - 1) / BPG;
        hd.parallel_for(sycl::nd_range<1>(groups * BPG * SG, BPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            for (int i = it.get_local_linear_id(); i < N64; i += BPG * SG) l64[i] = Tabs<T>::t64(i);
            for (int i = it.get_local_linear_id(); i < N32; i += BPG * SG) l32[i] = Tabs<T>::t32(i);
            if (N64 > 0 || N32 > 0) sycl::group_barrier(it.get_group());
            const Lut L{l64.template get_multi_ptr<sycl::access::decorated::no>().get(), l32.template get_multi_ptr<sycl::access::decorated::no>().get()};
            const int64_t k = it.get_group(0) * BPG + sg.get_group_id()[0];
            if (k >= nb) return;
            const int lane = sg.get_local_id()[0];
            float w[8];
            Deq<T>::group(src + k * SB, lane, w, L);
            uint32_t o[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t a = sycl::bit_cast<uint16_t>((sycl::half) w[2 * j]), b = sycl::bit_cast<uint16_t>((sycl::half) w[2 * j + 1]);
                o[j] = a | (b << 16);
            }
            *reinterpret_cast<sycl::uint4*>(out + k * 256 + lane * 8) = sycl::uint4(o[0], o[1], o[2], o[3]);
        });
    });
}

}  // namespace

extern "C" {

// the blocks of a matrix as fp16: nv values (a multiple of 256), out 16-byte aligned
int ns_mm3s_to_half(ns_gpu* g, int type, const void* src, int64_t nv, uint16_t* out) {
    NS_TRY
    if (nv % 256 || type == T_F32 || !sb_bytes(type)) return ns_fail("ns_mm3s_to_half: type " + std::to_string(type));
    const auto* b = (const uint8_t*) src;
    switch (type) {
        case T_IQ2_XXS: to_half<T_IQ2_XXS>(g->q, b, out, nv / 256); return 0;
        case T_IQ1_M: to_half<T_IQ1_M>(g->q, b, out, nv / 256); return 0;
        case T_IQ3_XXS: to_half<T_IQ3_XXS>(g->q, b, out, nv / 256); return 0;
        case T_IQ4_XS: to_half<T_IQ4_XS>(g->q, b, out, nv / 256); return 0;
        case T_Q5_K: to_half<T_Q5_K>(g->q, b, out, nv / 256); return 0;
        case T_Q6_K: to_half<T_Q6_K>(g->q, b, out, nv / 256); return 0;
        default: return ns_fail("ns_mm3s_to_half: type");
    }
    NS_CATCH
}

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

// A doorbell: when the work queued before it is done, *flag (pinned host memory) = seq, so the host learns it without waiting for the queue (it runs on behind this)
int ns_mm3s_doorbell(ns_gpu* g, uint32_t* flag, uint32_t seq) {
    NS_TRY
    g->q.single_task([=]() {
        sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system, sycl::access::address_space::global_space> a(*flag);
        a.store(seq);
    });
    return 0;
    NS_CATCH
}

// f[r * stride + at + c] = x[r * C + c]: a hidden state into its column of the drafter's features
int ns_mm3s_put_cols(ns_gpu* g, const float* x, float* f, int64_t T, int64_t C, int64_t stride, int64_t at) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / C, c = id[0] % C;
        f[r * stride + at + c] = x[r * C + c];
    });
    return 0;
    NS_CATCH
}

// What queuing a launch costs the host, microseconds: [0] g's queue, [1] a plain in-order queue, [2] the same with discarded events, [3] enqueue functions on it,
// [4] on g's queue with a local accessor (the expert kernels' shape)
int ns_mm3s_launch_bench(ns_gpu* g, double* out) {
    NS_TRY
    constexpr int N = 256;
    int* p = sycl::malloc_device<int>(1024, g->q);
    auto time = [&](auto&& fn) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) fn();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
        return us;
    };
    out[0] = time([&] { g->q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) { p[i] = 1; }); });
    g->q.wait();
    sycl::queue qb(g->ctx, g->dev, sycl::property::queue::in_order());
    out[1] = time([&] { qb.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) { p[i] = 2; }); });
    qb.wait();
    sycl::queue qd(g->ctx, g->dev, sycl::property_list{sycl::property::queue::in_order(), sycl::ext::oneapi::property::queue::discard_events()});
    out[2] = time([&] { qd.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) { p[i] = 3; }); });
    qd.wait();
    namespace oe = sycl::ext::oneapi::experimental;
    out[3] = time([&] { oe::nd_launch(qd, sycl::nd_range<1>(256, 32), [=](sycl::nd_item<1> it) { p[it.get_global_linear_id()] = 4; }); });
    qd.wait();
    out[4] = time([&] {
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int, 1> l(sycl::range<1>(64), h);
            h.parallel_for(sycl::nd_range<1>(256, 32), [=](sycl::nd_item<1> it) { l[0] = 1; p[it.get_global_linear_id()] = l[0]; });
        });
    });
    g->q.wait();
    sycl::free(p, g->q);
    return 0;
    NS_CATCH
}

// A small table from pinned memory into VRAM in a kernel, not through the runtime's memcpy (which goes to the copy engine, behind the experts' swaps)
int ns_mm3s_kcopy(ns_gpu* g, uint32_t* dst, const uint32_t* src, int64_t words) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>((size_t) words), [=](sycl::id<1> i) { dst[i] = src[i]; });
    return 0;
    NS_CATCH
}

// *best = the argmax of base + bias over n values, in a launch
int ns_mm3s_argmax_add(ns_gpu* g, const float* base, const float* bias, int64_t n, int32_t* best) {
    NS_TRY
    constexpr int WG = 512;
    g->q.parallel_for(sycl::nd_range<1>((size_t) WG, WG), [=](sycl::nd_item<1> it) {
        const int lid = (int) it.get_local_id(0);
        float bv = -INFINITY;
        int32_t bi = -1;
        for (int64_t i = lid; i < n; i += WG) {
            const float v = base[i] + bias[i];
            if (v > bv) { bv = v; bi = (int32_t) i; }
        }
        const float m = sycl::reduce_over_group(it.get_group(), bv, sycl::maximum<float>());
        const int32_t w = sycl::reduce_over_group(it.get_group(), bv == m ? bi : (int32_t) 0x7fffffff, sycl::minimum<int32_t>());
        if (lid == 0) *best = w;
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

int ns_mm3s_expert_gu(ns_gpu* g, int type, const uint64_t* bases, int64_t off_up, const float* x, const int32_t* tok, float* h, int64_t entries, int64_t rows, int64_t cols, float alpha,
                      float limit) {
    NS_TRY
    if (!ns_mm3s_expert_supported(type, cols)) return ns_fail("ns_mm3s_expert_gu: type " + std::to_string(type) + " with cols " + std::to_string(cols));
    static const int rps = [] { const char* v = std::getenv("NS_MM3S_RPS"); return v ? std::atoi(v) : 2; }();
    static const int ku = [] { const char* v = std::getenv("NS_MM3S_KU"); return v ? std::atoi(v) : 1; }();
#define GU(T, R, K) gu<T, R, K>(g->q, bases, off_up, x, tok, h, entries, rows, cols, alpha, limit)
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

int ns_mm3s_expert_down(ns_gpu* g, int type, const uint64_t* bases, int64_t off_down, const float* h, const float* w, const int32_t* rank, const int32_t* tok, float* ys, int64_t plane, int64_t entries,
                        int64_t rows, int64_t cols) {
    NS_TRY
    if (!ns_mm3s_expert_supported(type, cols)) return ns_fail("ns_mm3s_expert_down: type " + std::to_string(type) + " with cols " + std::to_string(cols));
    static const int rps = [] { const char* v = std::getenv("NS_MM3S_RPS"); return v ? std::atoi(v) : 2; }();
    static const int ku = [] { const char* v = std::getenv("NS_MM3S_KU"); return v ? std::atoi(v) : 1; }();
#define DN(T, R, K) down<T, R, K>(g->q, bases, off_down, h, w, rank, tok, ys, plane, entries, rows, cols)
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
