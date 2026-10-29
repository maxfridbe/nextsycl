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

#include <cstdint>
#include <cstdlib>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

namespace {

enum : int { T_IQ2_XXS = 16, T_IQ3_XXS = 18, T_IQ4_XS = 23, T_IQ1_M = 29 };

constexpr int sb_bytes(int type) {
    switch (type) {
        case T_IQ2_XXS: return 66;
        case T_IQ3_XXS: return 98;
        case T_IQ4_XS: return 136;
        case T_IQ1_M: return 56;
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

constexpr int SG = 32, RPG = 4;

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
        default: return ns_fail("ns_mm3s_expert_down: type");
    }
#undef RUN
    NS_CATCH
}

}  // extern "C"
