// std.cpp: the routed experts in ggml's own row types (the llama.cpp / Unsloth form of DeepSeek-V4.1-Flash: merged
// ffn_{gate,up,down}_exps tensors whose type varies by layer - MXFP4, Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S) (ds41.h).
//
// One scheme for every type: a 256-value block is 32 groups of 8 values, and each type has one function that makes group
// g of a block (its scales and signs and grid read for that group alone). A decode product is a sub-group a row, lane g
// the block's group g (a lane's 8 weights meet 8 contiguous activations: coalesced), the lanes' sums reduced at the end;
// the dequantizing kernel is a work-item a group. The activations are float32: no Q8 layout, the one set of kernels for
// all the types. The grids are ggml's (ggml-common.h). Formulas are ggml's dequantize_row_*.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

namespace {

// ggml type numbers
enum : int { T_Q8_0 = 8, T_Q2_K = 10, T_Q3_K = 11, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19, T_IQ3_S = 21, T_IQ2_S = 22, T_MXFP4 = 39 };

// bytes of 256 values
constexpr int sb_bytes(int type) {
    switch (type) {
        case T_Q8_0: return 8 * 34;
        case T_Q2_K: return 84;
        case T_Q3_K: return 110;
        case T_Q5_K: return 176;
        case T_Q6_K: return 210;
        case T_IQ2_XXS: return 66;
        case T_IQ2_XS: return 74;
        case T_IQ3_XXS: return 98;
        case T_IQ1_S: return 50;
        case T_IQ3_S: return 110;
        case T_IQ2_S: return 82;
        case T_MXFP4: return 8 * 17;
        default: return 0;
    }
}

// blocks are 2-byte aligned only (66, 74, 82, 98, 110 bytes)
inline uint32_t ld16(const uint8_t* p) { return (uint32_t) p[0] | ((uint32_t) p[1] << 8); }
inline uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
inline float h2f(const uint8_t* p) { return (float) sycl::bit_cast<sycl::half>((uint16_t) ld16(p)); }

// ggml's ksigns_iq2xs[x] (x | an odd-parity bit in bit 7) and kmask_iq2xs[j] (1 << j), computed
inline uint32_t ksign(uint32_t x) { x &= 127u; return x | ((sycl::popcount(x) & 1u) << 7); }
inline float sgn(uint32_t signs, int j) { return ((signs >> j) & 1u) ? -1.f : 1.f; }

// The grids a type reads, staged in local memory by each work-group (a gather from the global table is a transaction a lane)
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
template <> struct Tabs<T_IQ2_XS> { static constexpr int N64 = 512, N32 = 0; static uint64_t t64(int i) { return iq2xs_grid[i]; } static uint32_t t32(int) { return 0; } };
template <> struct Tabs<T_IQ2_S> { static constexpr int N64 = 1024, N32 = 0; static uint64_t t64(int i) { return iq2s_grid[i]; } static uint32_t t32(int) { return 0; } };
template <> struct Tabs<T_IQ3_XXS> { static constexpr int N64 = 0, N32 = 256; static uint64_t t64(int) { return 0; } static uint32_t t32(int i) { return iq3xxs_grid[i]; } };
template <> struct Tabs<T_IQ3_S> { static constexpr int N64 = 0, N32 = 512; static uint64_t t64(int) { return 0; } static uint32_t t32(int i) { return iq3s_grid[i]; } };
template <> struct Tabs<T_IQ1_S> { static constexpr int N64 = 0, N32 = 2048; static uint64_t t64(int) { return 0; } static uint32_t t32(int i) { return iq1s_grid_gpu[i]; } };

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

// IQ2_XS: d; 32 u16 (9 grid bits, 7 sign bits); 8 scale bytes (a nibble per 16 values)
template <> struct Deq<T_IQ2_XS> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib32 = g >> 2, l = g & 3;
        const uint32_t q = ld16(b + 2 + 2 * (4 * ib32 + l));
        const float db = h2f(b) * (0.5f + ((b[66 + ib32] >> (4 * (l >> 1))) & 0xf)) * 0.25f;
        const uint8_t* grid = (const uint8_t*) (L.a + (q & 511));
        const uint32_t signs = ksign(q >> 9);
        for (int j = 0; j < 8; ++j) w[j] = db * grid[j] * sgn(signs, j);
    }
};

// IQ2_S: d; qs[64] (32 grid bytes, then 32 sign bytes); qh[8]; scales[8]
template <> struct Deq<T_IQ2_S> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib32 = g >> 2, l = g & 3;
        const uint8_t* qs = b + 2;
        const uint32_t qh = b[66 + ib32];
        const float dl = h2f(b) * (0.5f + ((b[74 + ib32] >> (4 * (l >> 1))) & 0xf)) * 0.25f;
        const uint8_t* grid = (const uint8_t*) (L.a + (qs[ib32 * 4 + l] | ((qh << (8 - 2 * l)) & 0x300)));
        const uint32_t signs = qs[32 + ib32 * 4 + l];
        for (int j = 0; j < 8; ++j) w[j] = dl * grid[j] * sgn(signs, j);
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

// IQ3_S: d; qs[64]; qh[8]; signs[32]; scales[4] (a nibble per 32 values)
template <> struct Deq<T_IQ3_S> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib32 = g >> 2, l = g & 3;
        const uint8_t* qs = b + 2 + 8 * ib32;
        const uint32_t qh = b[66 + ib32];
        const float db = h2f(b) * (1 + 2 * ((b[106 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xf));
        const uint32_t signs = b[74 + 4 * ib32 + l];
        const uint8_t* g1 = (const uint8_t*) (L.b + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
        const uint8_t* g2 = (const uint8_t*) (L.b + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
        for (int j = 0; j < 4; ++j) {
            w[j] = db * g1[j] * sgn(signs, j);
            w[j + 4] = db * g2[j] * sgn(signs, j + 4);
        }
    }
};

// IQ1_S: d; qs[32] (8 index bits); qh[8] u16: three high index bits a group of 8, a 3-bit scale, a delta sign
template <> struct Deq<T_IQ1_S> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int ib = g >> 2, l = g & 3;
        const uint32_t qh = ld16(b + 34 + 2 * ib);
        const float dl = h2f(b) * (2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
        // SYCL builds have ggml's GPU form of the grid: the entry + 1 in nibbles, values 0-3 the even nibbles, 4-7 the odd
        const uint32_t e = L.b[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
        for (int j = 0; j < 8; ++j) w[j] = dl * ((int) ((e >> (j < 4 ? 8 * j : 8 * (j - 4) + 4)) & 0xF) - 1 + delta);
    }
};

// Q2_K: scales[16] (a nibble pair a 16 values), qs[64], d, dmin
template <> struct Deq<T_Q2_K> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int v = g * 8, half = v >> 7, u = v & 127, jj = u >> 5, sec = (u & 31) >> 4, l0 = u & 15;
        const uint32_t sc = b[half * 8 + jj * 2 + sec];
        const float dl = h2f(b + 80) * (sc & 15), ml = h2f(b + 82) * (sc >> 4);
        const uint8_t* q = b + 16 + half * 32 + sec * 16 + l0;
        for (int j = 0; j < 8; ++j) w[j] = dl * ((q[j] >> (2 * jj)) & 3) - ml;
    }
};

// Q3_K: hmask[32], qs[64], scales[12] (16 six-bit scales), d
template <> struct Deq<T_Q3_K> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
        const int v = g * 8, half = v >> 7, u = v & 127, jj = u >> 5, sec = (u & 31) >> 4, l0 = u & 15;
        const int i = half * 8 + jj * 2 + sec;
        const uint8_t* sc = b + 96;
        const int lo = i < 8 ? (sc[i] & 0xF) : (sc[i - 8] >> 4);
        const int hi = (sc[8 + (i & 3)] >> (2 * (i >> 2))) & 3;
        const float dl = h2f(b + 108) * ((lo | (hi << 4)) - 32);
        const uint8_t* q = b + 32 + half * 32 + sec * 16 + l0;
        const uint8_t* hm = b + sec * 16 + l0;
        const int m = 1 << (half * 4 + jj);
        for (int j = 0; j < 8; ++j) w[j] = dl * ((int) ((q[j] >> (2 * jj)) & 3) - ((hm[j] & m) ? 0 : 4));
    }
};

// Q5_K: d, dmin, scales[12] (eight 6-bit scale / min pairs), qh[32], qs[128]
template <> struct Deq<T_Q5_K> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
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

// Q6_K: ql[128], qh[64], scales[16] (int8), d
template <> struct Deq<T_Q6_K> {
    static void group(const uint8_t* b, int g, float* w, const Lut& L) {
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

// MXFP4: 8 blocks of (E8M0 scale, 16 bytes of E2M1 nibbles: value j the low nibble of byte j, j + 16 the high)
template <> struct Deq<T_MXFP4> {
    static void group(const uint8_t* r, int g, float* w, const Lut& L) {
        const uint8_t* b = r + (g >> 2) * 17;
        const uint32_t e = b[0];
        const float d = sycl::bit_cast<float>(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
        const int s = (g & 3) * 8;
        for (int j = 0; j < 8; ++j) {
            const int v = s + j;
            w[j] = kvalues_mxfp4[v < 16 ? (b[1 + v] & 0xF) : (b[1 + v - 16] >> 4)] * d;
        }
    }
};

// Q8_0: 8 blocks of (f16 scale, 32 int8)
template <> struct Deq<T_Q8_0> {
    static void group(const uint8_t* r, int g, float* w, const Lut& L) {
        const uint8_t* b = r + (g >> 2) * 34;
        const float d = h2f(b);
        const int8_t* q = (const int8_t*) (b + 2) + (g & 3) * 8;
        for (int j = 0; j < 8; ++j) w[j] = d * q[j];
    }
};

// y[(y0 + c) * rows + r] = W_e[r, :] . x[col_c] for the entries' matrices: entry e's rows at ptrs[e], meta[e] = {ncol, y0,
// col_0 .. col_7}. A sub-group works RPS rows at once (their blocks' loads issued together, the sums then): a row alone is a
// chain of dependent small loads
template <int T, int RPS>
void mmv(sycl::queue& q, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const float* x, float* y, int64_t rows, int64_t cols) {
    constexpr int SG = 32, RPG = 4, MC = 8, SB = sb_bytes(T), N64 = Tabs<T>::N64, N32 = Tabs<T>::N32;
    const int64_t rgs = (rows + RPG * RPS - 1) / (RPG * RPS), nb = cols / 256, rowb = nb * SB;
    q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint64_t, 1> l64(sycl::range<1>(N64 > 0 ? N64 : 1), h);
    sycl::local_accessor<uint32_t, 1> l32(sycl::range<1>(N32 > 0 ? N32 : 1), h);
    h.parallel_for(sycl::nd_range<1>(entries * rgs * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        // the grids into local memory first (every work-item of the group takes part, then all wait)
        for (int i = it.get_local_linear_id(); i < N64; i += RPG * SG) l64[i] = Tabs<T>::t64(i);
        for (int i = it.get_local_linear_id(); i < N32; i += RPG * SG) l32[i] = Tabs<T>::t32(i);
        if (N64 > 0 || N32 > 0) sycl::group_barrier(it.get_group());
        const Lut L{l64.template get_multi_ptr<sycl::access::decorated::no>().get(), l32.template get_multi_ptr<sycl::access::decorated::no>().get()};
        const int64_t gi = it.get_group(0), e = gi / rgs, r0 = ((gi % rgs) * RPG + sg.get_group_id()[0]) * RPS;
        const int lane = sg.get_local_id()[0];
        const int32_t* me = meta + e * 10;
        const int nc = me[0];
        if (nc == 0 || r0 >= rows) return;
        const uint8_t* base = reinterpret_cast<const uint8_t*>(ptrs[e]);
        float acc[RPS][MC] = {};
        for (int64_t k = 0; k < nb; ++k) {
            float w[RPS][8];
#pragma unroll
            for (int j = 0; j < RPS; ++j) {
                const int64_t r = r0 + j < rows ? r0 + j : rows - 1;
                Deq<T>::group(base + r * rowb + k * SB, lane, w[j], L);
            }
            for (int c = 0; c < nc; ++c) {
                const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(x + (int64_t) me[2 + c] * cols + k * 256 + lane * 8);
                const sycl::float4 a = xv[0], b = xv[1];
#pragma unroll
                for (int j = 0; j < RPS; ++j)
                    acc[j][c] += w[j][0] * a.x() + w[j][1] * a.y() + w[j][2] * a.z() + w[j][3] * a.w() + w[j][4] * b.x() + w[j][5] * b.y() + w[j][6] * b.z() + w[j][7] * b.w();
            }
        }
        for (int c = 0; c < nc; ++c) {
#pragma unroll
            for (int j = 0; j < RPS; ++j) {
                const float s = sycl::reduce_over_group(sg, acc[j][c], sycl::plus<float>());
                if (lane == 0 && r0 + j < rows) y[(int64_t) (me[1] + c) * rows + r0 + j] = s;
            }
        }
    });
    });
}

// out [rows][cols] = the matrix as fp16 or float32, a work-item a group of 8 (work-groups of 256, the grids in local memory)
template <int T, bool F16>
void dequant(sycl::queue& q, const uint8_t* src, int64_t rows, int64_t cols, void* out) {
    constexpr int SB = sb_bytes(T), WG = 256, N64 = Tabs<T>::N64, N32 = Tabs<T>::N32;
    const int64_t nb = cols / 256, n = rows * nb * 32, groups = (n + WG - 1) / WG;
    q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint64_t, 1> l64(sycl::range<1>(N64 > 0 ? N64 : 1), h);
    sycl::local_accessor<uint32_t, 1> l32(sycl::range<1>(N32 > 0 ? N32 : 1), h);
    h.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (int i = it.get_local_linear_id(); i < N64; i += WG) l64[i] = Tabs<T>::t64(i);
        for (int i = it.get_local_linear_id(); i < N32; i += WG) l32[i] = Tabs<T>::t32(i);
        if (N64 > 0 || N32 > 0) sycl::group_barrier(it.get_group());
        const Lut L{l64.template get_multi_ptr<sycl::access::decorated::no>().get(), l32.template get_multi_ptr<sycl::access::decorated::no>().get()};
        const int64_t i = it.get_global_linear_id();
        if (i >= n) return;
        const int64_t gr = i % 32, k = (i / 32) % nb, r = i / (32 * nb);
        float w[8];
        Deq<T>::group(src + (r * nb + k) * SB, (int) gr, w, L);
        const int64_t o = r * cols + k * 256 + gr * 8;
        if (F16) {
            sycl::half* hh = reinterpret_cast<sycl::half*>(out) + o;
            for (int j = 0; j < 8; ++j) hh[j] = (sycl::half) w[j];
        } else {
            float* f = reinterpret_cast<float*>(out) + o;
            for (int j = 0; j < 8; ++j) f[j] = w[j];
        }
    });
    });
}

}  // namespace

extern "C" {

int ns_ds41_std_supported(int type, int64_t cols) {
    return sb_bytes(type) != 0 && cols > 0 && cols % 256 == 0;
}

int ns_ds41_std_mmv(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const float* x, float* y, int64_t rows, int64_t cols) {
    NS_TRY
    if (!ns_ds41_std_supported(type, cols)) return ns_fail("ns_ds41_std_mmv: type " + std::to_string(type) + " with cols " + std::to_string(cols));
    // rows a sub-group (NS_DS41_STD_RPS: 1, 2 or 4)
    static const int rps = [] { const char* v = std::getenv("NS_DS41_STD_RPS"); return v ? std::atoi(v) : 2; }();
    switch (type) {
#define CASE(T) case T: if (rps == 4) mmv<T, 4>(g->q, ptrs, meta, entries, x, y, rows, cols); else if (rps == 2) mmv<T, 2>(g->q, ptrs, meta, entries, x, y, rows, cols); else mmv<T, 1>(g->q, ptrs, meta, entries, x, y, rows, cols); return 0;
        CASE(T_Q8_0) CASE(T_Q2_K) CASE(T_Q3_K) CASE(T_Q5_K) CASE(T_Q6_K) CASE(T_IQ2_XXS) CASE(T_IQ2_XS) CASE(T_IQ3_XXS) CASE(T_IQ1_S) CASE(T_IQ3_S) CASE(T_IQ2_S) CASE(T_MXFP4)
#undef CASE
        default: return ns_fail("ns_ds41_std_mmv: type");
    }
    NS_CATCH
}

int ns_ds41_std_dequant(ns_gpu* g, int type, const void* src, int64_t rows, int64_t cols, void* out, int f16) {
    NS_TRY
    if (!ns_ds41_std_supported(type, cols)) return ns_fail("ns_ds41_std_dequant: type " + std::to_string(type) + " with cols " + std::to_string(cols));
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    switch (type) {
#define CASE(T) case T: if (f16) dequant<T, true>(g->q, s, rows, cols, out); else dequant<T, false>(g->q, s, rows, cols, out); return 0;
        CASE(T_Q8_0) CASE(T_Q2_K) CASE(T_Q3_K) CASE(T_Q5_K) CASE(T_Q6_K) CASE(T_IQ2_XXS) CASE(T_IQ2_XS) CASE(T_IQ3_XXS) CASE(T_IQ1_S) CASE(T_IQ3_S) CASE(T_IQ2_S) CASE(T_MXFP4)
#undef CASE
        default: return ns_fail("ns_ds41_std_dequant: type");
    }
    NS_CATCH
}

}  // extern "C"
