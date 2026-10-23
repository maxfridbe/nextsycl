// mmvq.cpp: decode's expert products for the Unsloth form (ds41.h) with the activations in a split Q8 layout and dp4a -
// the deepseek4 silo's scheme (copied, not shared): 8 lanes a 256-block, a lane a sub-block of 32 values, a sub-group
// a run of consecutive rows with its lanes' activation blocks held in registers (loaded again only where the column
// changes), a bounded number of work-groups (each fills its grid table in local memory once). std.cpp's float32
// kernel decodes a lane's 8 values a block and re-reads the row's float32 activations for every row (20 KiB against
// ~1.3 KiB of IQ2_XXS weights a row): 133-229 GB/s on the B70. Types: IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// Q2_K, Q5_K, Q8_0, MXFP4 (deepseek4's, and IQ1_S / IQ2_XS / IQ3_S / Q2_K new here), and the XQ file's mixed Q3_K / Q2_0 /
// PTQ1_0 experts (codes 1000 / 1001). The table is std_mmv's (meta[e] = {ncol, y0,
// col_0 ..}), one column an entry; an entry with ncol 0 is skipped (no VRAM slot: the host's products fill it).
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

namespace {
constexpr int QK = 256, Q8K = 32, WARPS = 4, WARP = 32;

inline int dp4a(int a, int b, int c) { return dpct::dp4a(a, b, c); }
inline int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = dp4a(a.x(), b.x(), acc); acc = dp4a(a.y(), b.y(), acc); acc = dp4a(a.z(), b.z(), acc); return dp4a(a.w(), b.w(), acc);
}
// 16 bytes at a 2-byte aligned address from aligned 16-byte loads (the second only when unaligned)
inline sycl::int4 load16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;
    const bool half = (a & 2) != 0;
    const sycl::int4 lo = q[0], hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t) lo.x(), (uint32_t) lo.y(), (uint32_t) lo.z(), (uint32_t) lo.w(),
                           (uint32_t) hi.x(), (uint32_t) hi.y(), (uint32_t) hi.z(), (uint32_t) hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int) o[0], (int) o[1], (int) o[2], (int) o[3]);
}
inline uint32_t word_of(const sycl::int4& v, int i) { return uint32_t(i == 0 ? v.x() : i == 1 ? v.y() : i == 2 ? v.z() : v.w()); }
inline uint32_t load4_a2(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return uint32_t(h[0]) | (uint32_t(h[1]) << 16);
}

// One column of the split layout: block b's quants at qs + 32 b, its scale d[b], its 16-value groups' sums sm[2b..]
struct XA {
    const int8_t* qs;
    const sycl::half* d;
    const int16_t* sm;
    sycl::int4 q16(int b, int half) const { return *reinterpret_cast<const sycl::int4*>(qs + 32 * b + 16 * half); }
    float dd(int b) const { return (float) d[b]; }
    int sum32(int b) const { return sm[2 * b] + sm[2 * b + 1]; }
    int sum16(int b, int half) const { return sm[2 * b + half]; }
};
inline XA xcol(const int8_t* xs, int n_in, int ncols, int j) {
    const sycl::half* d = reinterpret_cast<const sycl::half*>(xs + std::size_t(ncols) * n_in);
    const int16_t* sm = reinterpret_cast<const int16_t*>(d + std::size_t(ncols) * (n_in / 32));
    return XA{xs + std::size_t(j) * n_in, d + std::size_t(j) * (n_in / 32), sm + std::size_t(j) * (n_in / 16)};
}

// a lane's activations: sub-block b0 + g's 32 quants, scale and sum
struct X { sycl::int4 u0, u1; float d; int s; };
inline X xload(int g, const XA& xa, int b0) { return X{xa.q16(b0 + g, 0), xa.q16(b0 + g, 1), xa.dd(b0 + g), xa.sum32(b0 + g)}; }
// the same with the two 16-value halves' sums (the K types' mins a half)
struct X2 { sycl::int4 u0, u1; float d; int s0, s1; };
inline X2 xload2(int g, const XA& xa, int b0) { return X2{xa.q16(b0 + g, 0), xa.q16(b0 + g, 1), xa.dd(b0 + g), xa.sum16(b0 + g, 0), xa.sum16(b0 + g, 1)}; }

// the signs: ksigns_iq2xs[b] is b with its parity as bit 7, a nibble of it widened to byte masks by a multiply
inline uint32_t ksign(uint32_t b7) { return b7 | ((sycl::popcount(b7) & 1u) << 7); }
inline uint32_t sign_mask4(uint32_t b) { return ((b * 0x00204081u) & 0x01010101u) * 0xffu; }   // bit i -> 0xff in byte i
inline void grid_group(sycl::uint2 e, uint32_t sb, int* v, int l) {
    const uint32_t m0 = sign_mask4(sb & 0xf), m1 = sign_mask4((sb >> 4) & 0xf);
    v[2 * l] = int((e.x() ^ m0) + (m0 & 0x01010101u));
    v[2 * l + 1] = int((e.y() ^ m1) + (m1 & 0x01010101u));
}

// ---- IQ2_XXS (66-byte blocks): lane g = sub-block g - four grid entries with their signs, a 4-bit scale
struct WideIQ2XXS {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_iq2_xxs;
    static constexpr int LUT = 2 * 256;
    static uint32_t lut(int i) {
        const uint64_t e = iq2xxs_grid[i >> 1];
        return uint32_t(i & 1 ? e >> 32 : e);
    }
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::uint2* grid = reinterpret_cast<const sycl::uint2*>(tab);
        // (2-byte loads of its 8 bytes measured slower here: B70 78 -> 89 us; for IQ3_XXS and Q2_K they won)
        const sycl::int4 q4 = load16_a2(reinterpret_cast<const uint8_t*>(b->qs) + 16 * (g >> 1));
        const uint32_t a0 = word_of(q4, 2 * (g & 1)), a1 = word_of(q4, 2 * (g & 1) + 1);   // grid bytes; signs + scale
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) grid_group(grid[(a0 >> (8 * l)) & 0xff], ksign((a1 >> (7 * l)) & 127), v, l);
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.d = (float) b->d * (0.5f + (float) (a1 >> 28)) * 0.25f;
        return r;
    }
    static float dot(const W& r, const X& v) { return r.d * v.d * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};

// ---- IQ2_XS (74-byte blocks: d, qs[32] u16 of a 9-bit grid index and 7 sign bits, scales[8] a nibble a 16 values):
// lane g = sub-block g - its four u16 entries, the scale nibbles of scales[g] (the first 16 values', the last 16's)
struct WideIQ2XS {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_iq2_xs;
    static constexpr int LUT = 2 * 512;
    static uint32_t lut(int i) {
        const uint64_t e = iq2xs_grid[i >> 1];
        return uint32_t(i & 1 ? e >> 32 : e);
    }
    struct W { sycl::int4 lo, hi; float s0, s1; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::uint2* grid = reinterpret_cast<const sycl::uint2*>(tab);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const uint32_t w0 = load4_a2(p + 2 + 8 * g), w1 = load4_a2(p + 2 + 8 * g + 4);
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q = ((l < 2 ? w0 : w1) >> (16 * (l & 1))) & 0xffff;
            grid_group(grid[q & 511], ksign(q >> 9), v, l);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        const uint32_t sc = p[2 + 64 + g];
        const float d = (float) b->d;
        r.s0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f;
        r.s1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        return r;
    }
    static float dot(const W& r, const X& v) { return v.d * (r.s0 * (float) dp4a4(r.lo, v.u0, 0) + r.s1 * (float) dp4a4(r.hi, v.u1, 0)); }
};

// ---- IQ3_XXS (98-byte blocks): lane g = sub-block g - eight grid entries of 4 unsigned bytes with their signs
struct WideIQ3XXS {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_iq3_xxs;
    static constexpr int LUT = 256;
    static uint32_t lut(int i) { return iq3xxs_grid[i]; }
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        // 2-byte loads, not load16_a2's selects: the kernel is bound by its arithmetic (B70 105 -> 75 us, B65 164 -> 117)
        const uint32_t i0 = load4_a2(p + 2 + 8 * g), i1 = load4_a2(p + 2 + 8 * g + 4);   // the 8 grid indices
        const uint32_t a1 = load4_a2(p + 2 + 64 + 4 * g);                                   // signs + scale
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t iw = l < 2 ? i0 : i1, sh = 16 * (l & 1);
            grid_group(sycl::uint2(tab[(iw >> sh) & 0xff], tab[(iw >> (sh + 8)) & 0xff]), ksign((a1 >> (7 * l)) & 127), v, l);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.d = (float) b->d * (0.5f + (float) (a1 >> 28)) * 0.5f;
        return r;
    }
    static float dot(const W& r, const X& v) { return r.d * v.d * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};

// ---- IQ2_S (82-byte blocks): lane g = sub-block g - four 10-bit grid indices with their sign bytes, two scales
struct WideIQ2S {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_iq2_s;
    static constexpr int LUT = 2 * 1024;
    static uint32_t lut(int i) {
        const uint64_t e = iq2s_grid[i >> 1];
        return uint32_t(i & 1 ? e >> 32 : e);
    }
    struct W { sycl::int4 lo, hi; float s0, s1; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::uint2* grid = reinterpret_cast<const sycl::uint2*>(tab);
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const uint32_t q4 = load4_a2(p + 2 + 4 * g), s4 = load4_a2(p + 2 + 32 + 4 * g);
        const uint32_t qh = p[2 + 64 + g], sc = p[2 + 64 + 8 + g];
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) grid_group(grid[((q4 >> (8 * l)) & 0xff) | ((qh << (8 - 2 * l)) & 0x300)], (s4 >> (8 * l)) & 0xff, v, l);
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        const float d = (float) b->d;
        r.s0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f;
        r.s1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        return r;
    }
    static float dot(const W& r, const X& v) { return v.d * (r.s0 * (float) dp4a4(r.lo, v.u0, 0) + r.s1 * (float) dp4a4(r.hi, v.u1, 0)); }
};

// ---- IQ3_S (110-byte blocks: d, qs[64] the index's low 8 bits, qh[8] a ninth bit an entry, signs[32], scales[4] a
// nibble a 32 values): lane g = sub-block g - eight 9-bit grid indices of 4 unsigned bytes with their sign bytes
struct WideIQ3S {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_iq3_s;
    static constexpr int LUT = 512;
    static uint32_t lut(int i) { return iq3s_grid[i]; }
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const uint32_t i0 = load4_a2(p + 2 + 8 * g), i1 = load4_a2(p + 2 + 8 * g + 4), s4 = load4_a2(p + 74 + 4 * g);
        const uint32_t qh = p[66 + g];
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t iw = l < 2 ? i0 : i1, sh = 16 * (l & 1);
            grid_group(sycl::uint2(tab[((iw >> sh) & 0xff) | ((qh << (8 - 2 * l)) & 256)], tab[((iw >> (sh + 8)) & 0xff) | ((qh << (7 - 2 * l)) & 256)]),
                       (s4 >> (8 * l)) & 0xff, v, l);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.d = (float) b->d * (float) (1 + 2 * ((p[106 + (g >> 1)] >> (4 * (g & 1))) & 0xf));
        return r;
    }
    static float dot(const W& r, const X& v) { return r.d * v.d * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};

// ---- IQ1_S (50-byte blocks: d, qs[32] the index's low 8 bits, qh[8] u16 a sub-block: 3 high bits an entry, a 3-bit
// scale, the delta's sign): lane g = sub-block g - four entries of ggml's GPU grid (the value + 1 in nibbles: values
// 0-3 the low nibbles of its bytes, 4-7 the high), so a dot is dl (q . x + (delta - 1) sum x)
struct WideIQ1S {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_iq1_s;
    static constexpr int LUT = 2048;
    static uint32_t lut(int i) { return iq1s_grid_gpu[i]; }
    struct W { sycl::int4 lo, hi; float d, delta; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const uint32_t q4 = load4_a2(p + 2 + 4 * g), qh = *reinterpret_cast<const uint16_t*>(p + 34 + 2 * g);
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t e = tab[((q4 >> (8 * l)) & 0xff) | (((qh >> (3 * l)) & 7) << 8)];
            v[2 * l] = int(e & 0x0f0f0f0fu);
            v[2 * l + 1] = int((e >> 4) & 0x0f0f0f0fu);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.d = (float) b->d * (float) (2 * ((qh >> 12) & 7) + 1);
        r.delta = (qh & 0x8000) ? -1.f - IQ1S_DELTA : -1.f + IQ1S_DELTA;
        return r;
    }
    static float dot(const W& r, const X& v) { return r.d * v.d * ((float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)) + r.delta * (float) v.s); }
};

// ---- Q2_K (84-byte blocks: scales[16] a nibble pair a 16 values, qs[64], d, dmin): lane g = sub-block g - its 32
// values are bits 2j of qs[32 n + l] (n = g / 4, j = g % 4), the first 16 with scales[8 n + 2 j], the last 16 with the next
struct WideQ2K {
    using XT = X2;
    static X2 xl(int g, const XA& xa, int b0) { return xload2(g, xa, b0); }
    using Block = block_q2_K;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    struct W { sycl::int4 lo, hi; float d0, m0, d1, m1; };
    static W load(const Block* b, int g, const uint32_t*) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const int n = g >> 2, sh = 2 * (g & 3);
        // 84-byte blocks: 4-byte aligned word loads, not load16_a2's selects (B70 154 -> 67 us, B65 151 -> 99)
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(p + 16 + 32 * n);
        const sycl::uint4 a(qw[0], qw[1], qw[2], qw[3]), c(qw[4], qw[5], qw[6], qw[7]);
        auto code = [&](uint32_t w) { return int((w >> sh) & 0x03030303u); };
        W r;
        r.lo = sycl::int4(code(a.x()), code(a.y()), code(a.z()), code(a.w()));
        r.hi = sycl::int4(code(c.x()), code(c.y()), code(c.z()), code(c.w()));
        const uint32_t s0 = p[8 * n + 2 * (g & 3)], s1 = p[8 * n + 2 * (g & 3) + 1];
        const sycl::half* dd = reinterpret_cast<const sycl::half*>(p + 80);
        const float d = (float) dd[0], dm = (float) dd[1];
        r.d0 = d * (float) (s0 & 15); r.m0 = dm * (float) (s0 >> 4);
        r.d1 = d * (float) (s1 & 15); r.m1 = dm * (float) (s1 >> 4);
        return r;
    }
    static float dot(const W& r, const X2& v) {
        return v.d * (r.d0 * (float) dp4a4(r.lo, v.u0, 0) - r.m0 * (float) v.s0 + r.d1 * (float) dp4a4(r.hi, v.u1, 0) - r.m1 * (float) v.s1);
    }
};

// ---- MXFP4 (eight 17-byte blocks of 32 make the 256-value step, lane g block g)
inline float e8m0_half(uint32_t x) {
    const uint32_t bits = x < 2 ? (0x00200000u << x) : ((x - 1) << 23);
    return sycl::bit_cast<float>(bits);
}
constexpr int8_t FP4V[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
struct Mx8 { block_mxfp4 b[8]; };
static_assert(sizeof(Mx8) == 136, "eight MXFP4 blocks are 136 bytes");
struct WideMXFP4 {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = Mx8;
    static constexpr int LUT = 256;
    static uint32_t lut(int i) { return uint32_t(uint8_t(FP4V[i & 15])) | (uint32_t(uint8_t(FP4V[i >> 4])) << 8); }
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b) + 17 * g;
        const uint8_t* q = p + 1;
        const uintptr_t a = reinterpret_cast<uintptr_t>(q);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
        const int sh = int(a & 3) * 8;
        uint32_t ww[5];
#pragma unroll
        for (int i = 0; i < 4; ++i) ww[i] = w[i];
        ww[4] = sh ? w[4] : 0;
        uint32_t lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t c = sh ? (ww[i] >> sh) | (ww[i + 1] << (32 - sh)) : ww[i];
            uint32_t l = 0, h = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t e = tab[(c >> (8 * k)) & 0xff];
                l |= (e & 0xff) << (8 * k);
                h |= ((e >> 8) & 0xff) << (8 * k);
            }
            lo[i] = l;
            hi[i] = h;
        }
        r.lo = sycl::int4(int(lo[0]), int(lo[1]), int(lo[2]), int(lo[3]));
        r.hi = sycl::int4(int(hi[0]), int(hi[1]), int(hi[2]), int(hi[3]));
        r.d = e8m0_half(p[0]);
        return r;
    }
    static float dot(const W& r, const X& v) { return r.d * v.d * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};

// ---- Q8_0 (eight 34-byte blocks of 32 make the 256-value step, lane g block g)
struct Q8x8 { block_q8_0 b[8]; };
static_assert(sizeof(Q8x8) == 272, "eight Q8_0 blocks are 272 bytes");
struct WideQ8_0 {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = Q8x8;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int g, const uint32_t*) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b) + 34 * g;
        return W{load16_a2(p + 2), load16_a2(p + 18), (float) *reinterpret_cast<const sycl::half*>(p)};
    }
    static float dot(const W& r, const X& v) { return r.d * v.d * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};

// ---- Q5_K (176-byte blocks): lane g = sub-block g - 5-bit unsigned codes, value = d sc_g q - dmin m_g
struct WideQ5K {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_q5_K;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    struct W { sycl::int4 lo, hi; float dsc, dmn; };
    static W load(const Block* b, int g, const uint32_t*) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const sycl::uint4* qs = reinterpret_cast<const sycl::uint4*>(p + 48 + 32 * (g >> 1));
        const sycl::uint4* qh = reinterpret_cast<const sycl::uint4*>(p + 16);
        const sycl::uint4 l0 = qs[0], l1 = qs[1], h0 = qh[0], h1 = qh[1];
        const int sh = 4 * (g & 1);
        auto code = [&](uint32_t l, uint32_t h) { return int(((l >> sh) & 0x0f0f0f0fu) | (((h >> g) & 0x01010101u) << 4)); };
        W r;
        r.lo = sycl::int4(code(l0.x(), h0.x()), code(l0.y(), h0.y()), code(l0.z(), h0.z()), code(l0.w(), h0.w()));
        r.hi = sycl::int4(code(l1.x(), h1.x()), code(l1.y(), h1.y()), code(l1.z(), h1.z()), code(l1.w(), h1.w()));
        const uint8_t* sc = p + 4;
        uint32_t s, m;
        if (g < 4) {
            s = sc[g] & 63;
            m = sc[g + 4] & 63;
        } else {
            s = (sc[g + 4] & 0xf) | ((sc[g - 4] >> 6) << 4);
            m = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
        }
        const sycl::half* dd = reinterpret_cast<const sycl::half*>(p);
        r.dsc = (float) dd[0] * (float) s;
        r.dmn = (float) dd[1] * (float) m;
        return r;
    }
    static float dot(const W& r, const X& v) { return v.d * (r.dsc * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)) - r.dmn * (float) v.s); }
};

// ---- q4_K (144-byte blocks: d, dmin, 12 bytes of 6-bit scales and mins, qs [128]): lane g = sub-block g - its 32
// values' nibbles (qs[32 (g / 2) + j], the high nibble for odd g) as unsigned codes; value = d sc_g q - dmin m_g (the
// engine's dense weights requantized at load, NS_DS41_DENSE_Q4K)
struct WideQ4K {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_q4_K;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    struct W { sycl::int4 lo, hi; float dsc, dmn; };
    static W load(const Block* b, int g, const uint32_t*) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const sycl::uint4* qs = reinterpret_cast<const sycl::uint4*>(p + 16 + 32 * (g >> 1));
        const sycl::uint4 l0 = qs[0], l1 = qs[1];
        const int sh = 4 * (g & 1);
        auto code = [&](uint32_t l) { return int((l >> sh) & 0x0f0f0f0fu); };
        W r;
        r.lo = sycl::int4(code(l0.x()), code(l0.y()), code(l0.z()), code(l0.w()));
        r.hi = sycl::int4(code(l1.x()), code(l1.y()), code(l1.z()), code(l1.w()));
        const uint8_t* sc = p + 4;
        uint32_t s, m;
        if (g < 4) {
            s = sc[g] & 63;
            m = sc[g + 4] & 63;
        } else {
            s = (sc[g + 4] & 0xf) | ((sc[g - 4] >> 6) << 4);
            m = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
        }
        const sycl::half* dd = reinterpret_cast<const sycl::half*>(p);
        r.dsc = (float) dd[0] * (float) s;
        r.dmn = (float) dd[1] * (float) m;
        return r;
    }
    static float dot(const W& r, const X& v) { return v.d * (r.dsc * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)) - r.dmn * (float) v.s); }
};

// ---- the XQ file's expert types (their own 256-value steps; each lane's 32 values as unsigned codes, the offset taken
// off with the activations' sums)
// Q3_K (110-byte blocks: hmask[32], qs[64], scales[12], d): lane g = sub-block g - code (qs[32 (g / 4) + l] >> 2 (g % 4))
// & 3 | hmask bit g of byte l << 2, value d (scale - 32) (code - 4), scales 2g and 2g + 1 a 16 values
struct MQ3K {
    static constexpr int BYTES = 110;
    struct W { sycl::int4 lo, hi; float d0, d1; };
    static W load(const uint8_t* p, int g) {
        const sycl::int4 h0 = load16_a2(p), h1 = load16_a2(p + 16);
        const uint8_t* q = p + 32 + 32 * (g >> 2);
        const sycl::int4 q0 = load16_a2(q), q1 = load16_a2(q + 16);
        const int sh = 2 * (g & 3);
        auto code = [&](uint32_t qw, uint32_t hw) { return int(((qw >> sh) & 0x03030303u) | (((hw >> g) & 0x01010101u) << 2)); };
        W r;
        r.lo = sycl::int4(code(q0.x(), h0.x()), code(q0.y(), h0.y()), code(q0.z(), h0.z()), code(q0.w(), h0.w()));
        r.hi = sycl::int4(code(q1.x(), h1.x()), code(q1.y(), h1.y()), code(q1.z(), h1.z()), code(q1.w(), h1.w()));
        // lane g's scales 2g and 2g + 1: their low nibbles in the byte pair at (2g) % 8 (high nibbles for g >= 4), their top
        // 2 bits in the pair at 8 + (2g) % 4 (shifted by 2 (g / 2)) - two 2-byte loads; byte loads at the lanes' own
        // offsets were half this product's time
        const uint8_t* sc = p + 96;
        const uint32_t lw = *reinterpret_cast<const uint16_t*>(sc + ((2 * g) & 7)), hw = *reinterpret_cast<const uint16_t*>(sc + 8 + ((2 * g) & 3));
        const int ls = g >= 4 ? 4 : 0, hs = 2 * (g >> 1);
        auto scale = [&](int h) { return int(((lw >> (8 * h + ls)) & 15) | (((hw >> (8 * h + hs)) & 3) << 4)) - 32; };
        const float d = (float) *reinterpret_cast<const sycl::half*>(p + 108);
        r.d0 = d * (float) scale(0);
        r.d1 = d * (float) scale(1);
        return r;
    }
    static float dot(const W& r, const X2& v) {
        return v.d * (r.d0 * (float) (dp4a4(r.lo, v.u0, 0) - 4 * v.s0) + r.d1 * (float) (dp4a4(r.hi, v.u1, 0) - 4 * v.s1));
    }
};
// Q2_0 (four 18-byte blocks of 64 make the step: d, qs[16], element j (qs[j / 4] >> 2 (j % 4)) & 3): lane g = half g % 2
// of block g / 2, value d (code - 1)
struct MQ2_0 {
    static constexpr int BYTES = 72;
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const uint8_t* p, int g) {
        const uint8_t* bl = p + 18 * (g >> 1);
        const uint8_t* q = bl + 2 + 8 * (g & 1);
        const uint32_t a = load4_a2(q), b = load4_a2(q + 4);
        auto spread = [](uint32_t c) { return int((c & 3) | (((c >> 2) & 3) << 8) | (((c >> 4) & 3) << 16) | (((c >> 6) & 3) << 24)); };
        W r;
        r.lo = sycl::int4(spread(a & 0xff), spread((a >> 8) & 0xff), spread((a >> 16) & 0xff), spread(a >> 24));
        r.hi = sycl::int4(spread(b & 0xff), spread((b >> 8) & 0xff), spread((b >> 16) & 0xff), spread(b >> 24));
        r.d = (float) *reinterpret_cast<const sycl::half*>(bl);
        return r;
    }
    static float dot(const W& r, const X2& v) { return r.d * v.d * (float) (dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)) - v.s0 - v.s1); }
};
// PTQ1_0 (two 28-byte blocks of 128 make the step: value m + 16 n trit n of qs[m], 80 + m + 8 n of qs[16 + m], 120 + h +
// 2 n of qs[24 + h]; trit n of byte b ((uint8) (b 3^n) 3) >> 8; d at byte 26): lane g = values 32 (g % 4) .. of block g / 4,
// value d (trit - 1)
struct MPTQ1_0 {
    static constexpr int BYTES = 56;
    struct W { sycl::int4 lo, hi; float d; };
    // trit n (power p = 3^n) of each byte of w, a byte each: two bytes a multiply in 16-bit lanes ((b p) & 0xff) 3 >> 8 -
    // b p <= 20655 and 765 fit a lane, nothing carries across
    static uint32_t trits(uint32_t w, uint32_t p) {
        const uint32_t e = ((((w & 0x00ff00ffu) * p) & 0x00ff00ffu) * 3u >> 8) & 0x00030003u;
        const uint32_t o = (((((w >> 8) & 0x00ff00ffu) * p) & 0x00ff00ffu) * 3u >> 8) & 0x00030003u;
        return e | (o << 8);
    }
    static uint32_t trit1(uint32_t b, uint32_t p) { return (((b * p) & 0xffu) * 3u) >> 8; }
    static W load(const uint8_t* p, int g) {
        const uint32_t* bw = reinterpret_cast<const uint32_t*>(p + 28 * (g >> 2));
        uint32_t w[7];
#pragma unroll
        for (int i = 0; i < 7; ++i) w[i] = bw[i];
        // lane g's 32 values 32 q .. (q = g % 4): words 0-3 hold qs[0..15] (values m + 16 n), 4-5 qs[16..23] (80 + m + 8 n),
        // byte 0-1 of word 6 qh (120 + h + 2 n)
        const int q = g & 3;
        uint32_t o[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t src = q < 3 ? w[k] : w[4 + (k & 1)];
            const uint32_t pw = q == 0 ? 1u : q == 1 ? 9u : q == 2 ? 81u : (k < 2 ? 9u : 27u);
            o[k] = trits(src, pw);
        }
#pragma unroll
        for (int k = 4; k < 8; ++k) {
            const uint32_t src = q < 2 ? w[k - 4] : w[4 + (k & 1)];
            const uint32_t pw = q == 0 ? 3u : q == 1 ? 27u : q == 2 ? (k < 6 ? 1u : 3u) : 81u;
            o[k] = trits(src, pw);
        }
        if (q == 3) {
            // values 112-119 are words 4-5's trit 4 (o[4], o[5]); 120-127 qh's trits 0-3, two bytes a trit
            const uint32_t h0 = w[6] & 0xffu, h1 = (w[6] >> 8) & 0xffu;
            o[6] = trit1(h0, 1) | (trit1(h1, 1) << 8) | (trit1(h0, 3) << 16) | (trit1(h1, 3) << 24);
            o[7] = trit1(h0, 9) | (trit1(h1, 9) << 8) | (trit1(h0, 27) << 16) | (trit1(h1, 27) << 24);
        }
        W r;
        r.lo = sycl::int4(int(o[0]), int(o[1]), int(o[2]), int(o[3]));
        r.hi = sycl::int4(int(o[4]), int(o[5]), int(o[6]), int(o[7]));
        r.d = (float) *reinterpret_cast<const sycl::half*>(reinterpret_cast<const uint8_t*>(bw) + 26);
        return r;
    }
    static float dot(const W& r, const X2& v) { return r.d * v.d * (float) (dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)) - v.s0 - v.s1); }
};
// the mixed form: an entry's type in bits 8-11 (gate | up) or 12-15 (down) of its first meta word - 0 Q3_K, 1 Q2_0, 2
// PTQ1_0 (std.cpp's T_MIXED_GU / T_MIXED_DOWN)
template <bool DOWN>
struct Mixed {
    using XT = X2;
    static X2 xl(int g, const XA& xa, int b0) { return xload2(g, xa, b0); }
    using Block = uint8_t;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
};
template <typename F> constexpr bool is_mixed = false;
template <bool D> constexpr bool is_mixed<Mixed<D>> = true;
template <bool D> constexpr int mixed_shift(Mixed<D>*) { return D ? 12 : 8; }
// R rows of one entry from its row `row` (wr: the entry's first byte), the step's NB blocks a lane
template <typename M, int NB, int R>
inline void mixed_rows(const uint8_t* wr, int row, int bpr, int sub, int g, const X2* xv, float* v) {
    const std::size_t rb = std::size_t(bpr) * M::BYTES;
#pragma unroll
    for (int u = 0; u < NB; ++u) {
        if (sub + 4 * u >= bpr) continue;
        typename M::W w[R];
#pragma unroll
        for (int j = 0; j < R; ++j) w[j] = M::load(wr + (std::size_t(row) + j) * rb + std::size_t(sub + 4 * u) * M::BYTES, g);
#pragma unroll
        for (int j = 0; j < R; ++j) v[j] += M::dot(w[j], xv[u]);
    }
}

// the large register file for the other types too (NS_DS41_MMVQ_GRF=1; the mixed form always has it)
bool big_grf() {
    static const bool v = std::getenv("NS_DS41_MMVQ_GRF") && std::atoi(std::getenv("NS_DS41_MMVQ_GRF")) != 0;
    return v;
}

int groups() {   // NS_DS41_MMVQ_GROUPS: the work-groups of a launch (each fills its grid table once; default 1024)
    static const int v = std::getenv("NS_DS41_MMVQ_GROUPS") ? std::max(1, std::atoi(std::getenv("NS_DS41_MMVQ_GROUPS"))) : 1024;
    return v;
}

template <typename F>
void fill_tab(uint32_t* tab, const sycl::nd_item<3>& item) {
    if constexpr (F::LUT <= 1) return;
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    for (int i = warp * WARP + lane; i < F::LUT; i += WARPS * WARP) tab[i] = F::lut(i);
    sycl::group_barrier(item.get_group());
}

// y[(meta[e].y0) * n_out + row] for the rows of every entry: a sub-group a run of consecutive rows, its lanes' NB
// activation blocks (4 blocks a step, NB steps: n_in up to 4 NB blocks) loaded where the column changes; R rows at once
// when they are the same entry's (their blocks' loads in flight together: a row alone is a chain of a load, the grid's
// lookups and the reduction)
template <typename F, int NB, int R, bool BIG> class RunK;   // the kernels' names: one a register file size
template <typename F, int NB, int R>
void run(sycl::queue& q, const uint64_t* ptrs, const int32_t* meta, int entries, const int8_t* x, int ncols, float* y, int n_in, int n_out) {
    const std::size_t rows = std::size_t(entries) * n_out;
    const unsigned blocks = (unsigned) std::min<std::size_t>((rows + WARPS - 1) / WARPS, std::size_t(groups()));
    const std::size_t nsg = std::size_t(blocks) * WARPS, chunk = ((rows + nsg - 1) / nsg + R - 1) / R * R;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        const sycl::nd_range<3> ndr(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP));
        auto kern = [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                           uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
                           fill_tab<F>(t, it);
                           const int lane = int(it.get_local_id(2)), warp = int(it.get_local_id(1));
                           const int g = lane & 7, sub = lane >> 3;
                           const int bpr = n_in / QK;
                           auto sg = it.get_sub_group();
                           const std::size_t r0 = (std::size_t(it.get_group(2)) * WARPS + warp) * chunk, r1 = std::min(rows, r0 + chunk);
                           int cur = -1;
                           typename F::XT xv[NB];
                           for (std::size_t r = r0; r < r1; r += R) {
                               const int e = int(r / n_out), row = int(r % n_out);
                               const int32_t* me = meta + 10 * e;
                               // R rows of one entry (n_out a multiple of R: an entry's rows never straddle a step)
                               if ((me[0] & 255) == 0) continue;
                               const int col = me[2];
                               if (col != cur) {
                                   const XA xa = xcol(x, n_in, ncols, col);
#pragma unroll
                                   for (int u = 0; u < NB; ++u)
                                       if (sub + 4 * u < bpr) xv[u] = F::xl(g, xa, (sub + 4 * u) * (QK / Q8K));
                                   cur = col;
                               }
                               float v[R] = {};
                               if constexpr (is_mixed<F>) {
                                   const uint8_t* wr = reinterpret_cast<const uint8_t*>(ptrs[e]);
                                   const int k = (me[0] >> mixed_shift(static_cast<F*>(nullptr))) & 15;
                                   if (k == 0) mixed_rows<MQ3K, NB, R>(wr, row, bpr, sub, g, xv, v);
                                   else if (k == 1) mixed_rows<MQ2_0, NB, R>(wr, row, bpr, sub, g, xv, v);
                                   else mixed_rows<MPTQ1_0, NB, R>(wr, row, bpr, sub, g, xv, v);
                               } else {
                                   const auto* wr = reinterpret_cast<const typename F::Block*>(ptrs[e]) + std::size_t(row) * bpr;
#pragma unroll
                                   for (int u = 0; u < NB; ++u) {
                                       if (sub + 4 * u >= bpr) continue;
                                       typename F::W w[R];
#pragma unroll
                                       for (int j = 0; j < R; ++j) w[j] = F::load(wr + std::size_t(j) * bpr + sub + 4 * u, g, t);
#pragma unroll
                                       for (int j = 0; j < R; ++j) v[j] += F::dot(w[j], xv[u]);
                                   }
                               }
#pragma unroll
                               for (int j = 0; j < R; ++j) {
#pragma unroll
                                   for (int o = WARP / 2; o > 0; o >>= 1) v[j] += sycl::permute_group_by_xor(sg, v[j], o);
                                   if (lane == 0 && r + j < r1) y[std::size_t(me[1]) * n_out + row + j] = v[j];
                               }
                           }
                       };
        // the mixed form's three loaders in one kernel: the large register file (half the threads a core) instead of spills
        if (is_mixed<F> || big_grf()) h.parallel_for<RunK<F, NB, R, true>>(ndr, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, kern);
        else h.parallel_for<RunK<F, NB, R, false>>(ndr, kern);
    });
}

int rows_at_once() {   // NS_DS41_MMVQ_ROWS: rows a sub-group works at once (1, 2 or 4; default 2)
    static const int v = std::getenv("NS_DS41_MMVQ_ROWS") ? std::atoi(std::getenv("NS_DS41_MMVQ_ROWS")) : 2;
    return v;
}

template <typename F, int NB>
void by_r(sycl::queue& q, const uint64_t* ptrs, const int32_t* meta, int entries, const int8_t* x, int ncols, float* y, int n_in, int n_out) {
    const int r = rows_at_once();
    if (r >= 4 && n_out % 4 == 0) return run<F, NB, 4>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
    if (r >= 2 && n_out % 2 == 0) return run<F, NB, 2>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
    return run<F, NB, 1>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
}

template <typename F>
void by_nb(sycl::queue& q, const uint64_t* ptrs, const int32_t* meta, int entries, const int8_t* x, int ncols, float* y, int n_in, int n_out) {
    switch ((n_in / QK + 3) / 4) {
        case 1: return by_r<F, 1>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
        case 2: return by_r<F, 2>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
        case 3: return by_r<F, 3>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
        case 4: return by_r<F, 4>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
        case 5: return by_r<F, 5>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
        case 6: return by_r<F, 6>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
        case 7: return by_r<F, 7>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
        default: return by_r<F, 8>(q, ptrs, meta, entries, x, ncols, y, n_in, n_out);
    }
}
}  // namespace

extern "C" {

size_t ns_ds41_q8_bytes(int64_t n_in, int64_t ncols) { return (size_t) (ncols * n_in + ncols * (n_in / 32) * 2 + ncols * (n_in / 16) * 2); }

// x [ncols, n_in] float32 -> the split layout: a sub-group of 32 a block (its amax / 127 rounded to fp16 first, as
// Q8_1's scale), each 16 quants' sum
int ns_ds41_quantize_q8(ns_gpu* g, const float* x, void* xs, int64_t n_in, int64_t ncols) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_ds41_quantize_q8: n_in a multiple of 32");
    const int64_t n = ncols * n_in;
    int8_t* qs = static_cast<int8_t*>(xs);
    sycl::half* d = reinterpret_cast<sycl::half*>(qs + n);
    int16_t* sm = reinterpret_cast<int16_t*>(d + n / 32);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((n + 255) / 256 * 256)), sycl::range<1>(256)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const int64_t i = it.get_global_id(0);
        const float v = i < n ? x[i] : 0.0f;
        float amax = sycl::fabs(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
        const sycl::half dh = (sycl::half) (amax / 127.0f);
        const float dd = (float) dh;
        const int8_t qv = (int8_t) (dd > 0.f ? sycl::clamp(sycl::round(v / dd), -127.f, 127.f) : 0.f);
        int t = qv;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) t += sycl::permute_group_by_xor(sg, t, o);
        if (i >= n) return;
        const int64_t b = i / 32, l = i % 32;
        qs[i] = qv;
        if (l == 0) d[b] = dh;
        if (l % 16 == 0) sm[i / 16] = (int16_t) t;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_mmvq_supported(int type, int64_t n_in) {
    switch (type) {
        case 8: case 10: case 12: case 13: case 16: case 17: case 18: case 19: case 21: case 22: case 39: case 1000: case 1001: return n_in % QK == 0 && n_in <= 32 * QK;
        default: return 0;
    }
}

int ns_ds41_std_mmvq(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const void* xs, int64_t ncols, float* y,
                     int64_t n_out, int64_t n_in) {
    NS_TRY
    if (!ns_ds41_mmvq_supported(type, n_in)) return ns_fail("ns_ds41_std_mmvq: ggml type " + std::to_string(type) + " at width " + std::to_string(n_in));
    if (entries < 1) return 0;
    const auto* x = static_cast<const int8_t*>(xs);
    const int E = (int) entries, NC = (int) ncols, K = (int) n_in, N = (int) n_out;
    switch (type) {
        case 8: by_nb<WideQ8_0>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 10: by_nb<WideQ2K>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 12: by_nb<WideQ4K>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 13: by_nb<WideQ5K>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 16: by_nb<WideIQ2XXS>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 17: by_nb<WideIQ2XS>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 18: by_nb<WideIQ3XXS>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 19: by_nb<WideIQ1S>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 21: by_nb<WideIQ3S>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 22: by_nb<WideIQ2S>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 1000: by_nb<Mixed<false>>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        case 1001: by_nb<Mixed<true>>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
        default: by_nb<WideMXFP4>(g->q, ptrs, meta, E, x, NC, y, K, N); break;
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
