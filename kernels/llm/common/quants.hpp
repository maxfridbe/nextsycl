// quants.hpp: the decode-width readers of the GGUF block types (kernels/llm/common) - for each type, one lane's part of a
// block as signed bytes and their scales, so a product is dp4a against Q8_1 activations and an expansion is the same
// values times their scales. Written from the formats (ns_gguf.h: the layouts and codebooks).
//
// A 256-value type is read by 8 lanes a block, lane g taking sub-block g (values 32g .. 32g + 31): `W` holds them as
// 32 signed bytes (lo = values 0-15, hi = 16-31) with a scale for each half (d0, d1) and the type's offset term:
//   value = d0 x q (or d1 x q)  - m0 / m1  (K-quants' mins: a whole sub-block's m0, or each half's m0 / m1)
//   value = d0 x (q + o[k])               (IQ1: q the grid's 0..2, o = delta - 1 for each group k of 8 values)
// A 32-value type is read a block a lane, the same `W` (d0 = d1, no offset).
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

#include "ns_gguf.h"

namespace nsq {

constexpr int QK = 256;

inline int dp4a(int a, int b, int c) { return dpct::dp4a(a, b, c); }   // signed bytes, int32 accumulate
inline int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = dp4a(a.x(), b.x(), acc); acc = dp4a(a.y(), b.y(), acc); acc = dp4a(a.z(), b.z(), acc); return dp4a(a.w(), b.w(), acc);
}
inline int sum4(const sycl::int4 b) { return dp4a4(sycl::int4(0x01010101), b, 0); }
inline uint32_t word_of(const sycl::int4& v, int i) { return uint32_t(i == 0 ? v.x() : i == 1 ? v.y() : i == 2 ? v.z() : v.w()); }
inline sycl::int4 i4(const int* v) { return sycl::int4(v[0], v[1], v[2], v[3]); }
inline float h2f(uint32_t bits) { return (float) sycl::bit_cast<sycl::half>(uint16_t(bits)); }

// 16 bytes at a 2-byte aligned address from aligned 16-byte loads (the second only when the address is unaligned, so
// nothing past an allocation's last chunk is read)
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
// 16 bytes at a 4-byte aligned address: four word loads
inline sycl::int4 load16_a4(const void* p) { const int* q = static_cast<const int*>(p); return sycl::int4(q[0], q[1], q[2], q[3]); }

// bytewise v - k for bytes v in [0, 256 - 128 - k) as signed bytes, no borrow between bytes: (v + 128 - k) ^ 0x80
inline int sub_bytes(uint32_t v, uint32_t k) { return int((v + (0x80808080u - k * 0x01010101u)) ^ 0x80808080u); }
// bit i of b -> 0xff in byte i (four bits)
inline uint32_t sign_mask4(uint32_t b) { return ((b * 0x00204081u) & 0x01010101u) * 0xffu; }

// one sub-block of a Q8_1 activation column: its 32 quants, its scale d and s = d x their sum
struct X { sycl::int4 lo, hi; float d, s; };
inline X xload(const block_q8_1* b) {
    const sycl::half2 ds = b->ds;
    return X{load16_a4(b->qs), load16_a4(b->qs + 16), (float) ds[0], (float) ds[1]};
}

enum Off { NONE, MIN32, MIN16, IQ1 };
struct W { sycl::int4 lo, hi; float d0, d1, m0, m1; float o[4]; };

// ---- the K-quants' 6-bit scales and mins (Q4_K, Q5_K): 12 bytes as three words
inline int k4_byte(const uint32_t sc[3], int i) { return int((sc[i >> 2] >> (8 * (i & 3))) & 0xff); }
inline void k4_scale_min(const uint32_t sc[3], int j, int& s, int& m) {
    if (j < 4) { s = k4_byte(sc, j) & 63; m = k4_byte(sc, j + 4) & 63; }
    else {
        s = (k4_byte(sc, j + 4) & 0xF) | ((k4_byte(sc, j - 4) >> 6) << 4);
        m = (k4_byte(sc, j + 4) >> 4) | ((k4_byte(sc, j) >> 6) << 4);
    }
}

// ---- 256-value types: F::load(block, g, tab) -> W of sub-block g; LUT words of local table (0 = none), F::lut(i)
struct TQ4K {
    using Block = block_q4_K;
    static constexpr int LUT = 0; static constexpr Off OFF = MIN32;
    static uint32_t lut(int) { return 0; }
    static W load(const Block* b, int g, const uint32_t*) {
        W r;
        const sycl::int4 hd = *reinterpret_cast<const sycl::int4*>(b);   // 144-byte blocks: 16-byte aligned
        const uint32_t sc[3] = {uint32_t(hd.y()), uint32_t(hd.z()), uint32_t(hd.w())};
        const float d = h2f(uint32_t(hd.x())), dmin = h2f(uint32_t(hd.x()) >> 16);
        const sycl::int4* q = reinterpret_cast<const sycl::int4*>(b->qs) + 2 * (g >> 1);
        const int sh = 4 * (g & 1);
        const sycl::int4 a = q[0], c = q[1];
        r.lo = sycl::int4((a.x() >> sh) & 0x0f0f0f0f, (a.y() >> sh) & 0x0f0f0f0f, (a.z() >> sh) & 0x0f0f0f0f, (a.w() >> sh) & 0x0f0f0f0f);
        r.hi = sycl::int4((c.x() >> sh) & 0x0f0f0f0f, (c.y() >> sh) & 0x0f0f0f0f, (c.z() >> sh) & 0x0f0f0f0f, (c.w() >> sh) & 0x0f0f0f0f);
        int s, m; k4_scale_min(sc, g, s, m);
        r.d0 = r.d1 = d * (float) s; r.m0 = r.m1 = dmin * (float) m;
        return r;
    }
};
struct TQ5K {
    using Block = block_q5_K;
    static constexpr int LUT = 0; static constexpr Off OFF = MIN32;
    static uint32_t lut(int) { return 0; }
    static W load(const Block* b, int g, const uint32_t* t) {
        W r = TQ4K::load(reinterpret_cast<const block_q4_K*>(b), g, t);   // the same head; qs is 32 bytes later
        const sycl::int4* q = reinterpret_cast<const sycl::int4*>(b->qs) + 2 * (g >> 1);
        const sycl::int4* h = reinterpret_cast<const sycl::int4*>(b->qh);
        const int sh = 4 * (g & 1);
        const sycl::int4 a = q[0], c = q[1], ha = h[0], hc = h[1];
        auto v = [&](int qv, int hv) { return ((qv >> sh) & 0x0f0f0f0f) | (((hv >> g) & 0x01010101) << 4); };
        r.lo = sycl::int4(v(a.x(), ha.x()), v(a.y(), ha.y()), v(a.z(), ha.z()), v(a.w(), ha.w()));
        r.hi = sycl::int4(v(c.x(), hc.x()), v(c.y(), hc.y()), v(c.z(), hc.z()), v(c.w(), hc.w()));
        return r;
    }
};
// Q6_K: sub-block g = half n (g / 4), quarter j (g % 4): ql bytes 64n + 32 (j & 1) .., nibble j >> 1; qh bytes 32n ..,
// bits 2j; scales 8n + 2j (values 0-15) and + 1 (16-31); q - 32
struct TQ6K {
    using Block = block_q6_K;
    static constexpr int LUT = 0; static constexpr Off OFF = NONE;
    static uint32_t lut(int) { return 0; }
    static W load(const Block* b, int g, const uint32_t*) {
        W r;
        const int n = g >> 2, j = g & 3;
        const uint8_t* ql = b->ql + 64 * n + 32 * (j & 1);
        const uint8_t* qh = b->qh + 32 * n;
        const int lsh = 4 * (j >> 1), hsh = 2 * j;
        int v[8];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const sycl::int4 l = load16_a2(ql + 16 * k), h = load16_a2(qh + 16 * k);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                v[4 * k + i] = sub_bytes(((word_of(l, i) >> lsh) & 0x0f0f0f0f) | (((word_of(h, i) >> hsh) & 0x03030303) << 4), 32);
        }
        r.lo = i4(v); r.hi = i4(v + 4);
        const float d = (float) b->d;
        r.d0 = d * (float) b->scales[8 * n + 2 * j]; r.d1 = d * (float) b->scales[8 * n + 2 * j + 1];
        r.m0 = r.m1 = 0.f;
        return r;
    }
};
// Q2_K: sub-block g = half n (g / 4), shift j (g % 4): qs bytes 32n .. (values 0-15) and 32n + 16 .. (16-31), bits 2j
struct TQ2K {
    using Block = block_q2_K;
    static constexpr int LUT = 0; static constexpr Off OFF = MIN16;
    static uint32_t lut(int) { return 0; }
    static W load(const Block* b, int g, const uint32_t*) {   // 84-byte blocks: 4-byte aligned
        W r;
        const int n = g >> 2, j = g & 3;
        int v[8];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const sycl::int4 q = load16_a4(b->qs + 32 * n + 16 * k);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[4 * k + i] = int((word_of(q, i) >> (2 * j)) & 0x03030303);
        }
        r.lo = i4(v); r.hi = i4(v + 4);
        const sycl::half2 dm = b->dm;
        const float d = (float) dm[0], dmin = (float) dm[1];
        const uint32_t s0 = b->scales[8 * n + 2 * j], s1 = b->scales[8 * n + 2 * j + 1];
        r.d0 = d * (float) (s0 & 0xF); r.d1 = d * (float) (s1 & 0xF);
        r.m0 = dmin * (float) (s0 >> 4); r.m1 = dmin * (float) (s1 >> 4);
        return r;
    }
};

// IQ4_XS: sub-block g's 16 bytes = 32 nibbles (low: values 0-15, high: 16-31), each through the codebook. A byte's two
// codebook bytes come from a 256-entry local table (unsigned, at bits 0 and 16)
inline uint32_t iq4_tab_entry(int b) {
    return uint32_t(uint8_t(kvalues_iq4nl[b & 15])) | uint32_t(uint8_t(kvalues_iq4nl[b >> 4])) << 16;
}
inline void iq4_tab4(const uint32_t* tab, uint32_t v, int& lo, int& hi) {
    const uint32_t a = tab[v & 0xff] | tab[(v >> 8) & 0xff] << 8, c = tab[(v >> 16) & 0xff] | tab[v >> 24] << 8;
    lo = int((a & 0xffff) | c << 16);
    hi = int(a >> 16 | (c & 0xffff0000u));
}
inline void iq4_16(const uint32_t* tab, const sycl::int4 q, W& r) {
    int lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) iq4_tab4(tab, word_of(q, i), lo[i], hi[i]);
    r.lo = i4(lo); r.hi = i4(hi);
}
struct TIQ4XS {
    using Block = block_iq4_xs;
    static constexpr int LUT = 256; static constexpr Off OFF = NONE;
    static uint32_t lut(int i) { return iq4_tab_entry(i); }
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * g);   // 136-byte blocks: 8-byte aligned
        const sycl::int2 qa = q2[0], qb = q2[1];
        iq4_16(tab, sycl::int4(qa.x(), qa.y(), qb.x(), qb.y()), r);
        const sycl::int2 hd = *reinterpret_cast<const sycl::int2*>(b);   // d, scales_h, scales_l
        const uint32_t h0 = uint32_t(hd.x()), sl = uint32_t(hd.y());
        const int ls = int((sl >> (4 * g)) & 0x0f) | int(((h0 >> (16 + 2 * g)) & 0x03) << 4);
        r.d0 = r.d1 = h2f(h0) * (float) (ls - 32);
        r.m0 = r.m1 = 0.f;
        return r;
    }
};

// the grid types: a group of 8 values is a grid entry (unsigned bytes, at most 0x3f) and 8 sign bits, applied
// bytewise: (v ^ m) + (m & 1) is -v in a byte whose mask m is 0xff (two's complement, no carry out of the byte as v > 0)
// Local words: [16] the 4-bit sign masks, [2 x 128] ksigns_iq2xs's 128 sign bytes as their two masks, then the grid.
constexpr int GL_SIGNS = 0, GL_KSIGNS = 16, GL_GRID = 16 + 2 * 128;
inline uint32_t grid_lut_head(int i) {
    if (i < 16) return sign_mask4(uint32_t(i));
    const uint32_t s = ksigns_iq2xs[(i - 16) >> 1];
    return sign_mask4((i - 16) & 1 ? s >> 4 : s & 0xf);
}
inline void signed_group(uint32_t w0, uint32_t w1, uint32_t m0, uint32_t m1, int* v, int l) {
    v[2 * l] = int((w0 ^ m0) + (m0 & 0x01010101u));
    v[2 * l + 1] = int((w1 ^ m1) + (m1 & 0x01010101u));
}
// a group with a sign byte sb
inline void grid_group(const uint32_t* tab, uint32_t w0, uint32_t w1, uint32_t sb, int* v, int l) {
    signed_group(w0, w1, tab[GL_SIGNS + (sb & 0xf)], tab[GL_SIGNS + ((sb >> 4) & 0xf)], v, l);
}
// a group with a 7-bit ksigns index k
inline void grid_group_k(const uint32_t* tab, uint32_t w0, uint32_t w1, uint32_t k, int* v, int l) {
    const sycl::uint2 m = *reinterpret_cast<const sycl::uint2*>(tab + GL_KSIGNS + 2 * k);
    signed_group(w0, w1, m.x(), m.y(), v, l);
}
struct TIQ3S {
    using Block = block_iq3_s;
    static constexpr int LUT = GL_GRID + 512; static constexpr Off OFF = NONE;
    static uint32_t lut(int i) { return i < GL_GRID ? grid_lut_head(i) : iq3s_grid[i - GL_GRID]; }
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::int4 q4 = load16_a2(b->qs + 16 * (g >> 1)), s4 = load16_a2(b->signs + 16 * (g >> 2));
        const uint32_t qa = word_of(q4, 2 * (g & 1)), qb = word_of(q4, 2 * (g & 1) + 1), sg = word_of(s4, g & 3);
        const uint32_t qh = (reinterpret_cast<const uint16_t*>(b->qh)[g >> 1] >> (8 * (g & 1))) & 0xff;
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qw = l < 2 ? qa : qb, sh = 16 * (l & 1);
            const uint32_t i0 = ((qw >> sh) & 0xff) | ((qh << (8 - 2 * l)) & 0x100), i1 = ((qw >> (sh + 8)) & 0xff) | ((qh << (7 - 2 * l)) & 0x100);
            grid_group(tab, tab[GL_GRID + i0], tab[GL_GRID + i1], (sg >> (8 * l)) & 0xff, v, l);
        }
        r.lo = i4(v); r.hi = i4(v + 4);
        r.d0 = r.d1 = (float) b->d * (float) (1 + 2 * ((b->scales[g >> 1] >> (4 * (g & 1))) & 0xf));
        r.m0 = r.m1 = 0.f;
        return r;
    }
};
struct TIQ3XXS {
    using Block = block_iq3_xxs;
    static constexpr int LUT = GL_GRID + 256; static constexpr Off OFF = NONE;
    static uint32_t lut(int i) { return i < GL_GRID ? grid_lut_head(i) : iq3xxs_grid[i - GL_GRID]; }
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::int4 q4 = load16_a2(b->qs + 16 * (g >> 1)), a4 = load16_a2(b->qs + 64 + 16 * (g >> 2));
        const uint32_t qa = word_of(q4, 2 * (g & 1)), qb = word_of(q4, 2 * (g & 1) + 1), aux = word_of(a4, g & 3);
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qw = l < 2 ? qa : qb, sh = 16 * (l & 1);
            grid_group_k(tab, tab[GL_GRID + ((qw >> sh) & 0xff)], tab[GL_GRID + ((qw >> (sh + 8)) & 0xff)], (aux >> (7 * l)) & 127, v, l);
        }
        r.lo = i4(v); r.hi = i4(v + 4);
        r.d0 = r.d1 = (float) b->d * (0.5f + (float) (aux >> 28)) * 0.5f;
        r.m0 = r.m1 = 0.f;
        return r;
    }
};
template <const uint64_t* GRID, int N>
inline uint32_t grid64_lut(int i) {
    if (i < GL_GRID) return grid_lut_head(i);
    const uint64_t e = GRID[(i - GL_GRID) >> 1];
    return uint32_t((i - GL_GRID) & 1 ? e >> 32 : e);
}
struct TIQ2XXS {
    using Block = block_iq2_xxs;
    static constexpr int LUT = GL_GRID + 2 * 256; static constexpr Off OFF = NONE;
    static uint32_t lut(int i) { return grid64_lut<iq2xxs_grid, 256>(i); }
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::int4 q4 = load16_a2(reinterpret_cast<const uint8_t*>(b->qs) + 16 * (g >> 1));
        const uint32_t a0 = word_of(q4, 2 * (g & 1)), a1 = word_of(q4, 2 * (g & 1) + 1);   // grid bytes; signs + scale
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t gi = (a0 >> (8 * l)) & 0xff;
            const sycl::uint2 e = *reinterpret_cast<const sycl::uint2*>(tab + GL_GRID + 2 * gi);
            grid_group_k(tab, e.x(), e.y(), (a1 >> (7 * l)) & 127, v, l);
        }
        r.lo = i4(v); r.hi = i4(v + 4);
        r.d0 = r.d1 = (float) b->d * (0.5f + (float) (a1 >> 28)) * 0.25f;
        r.m0 = r.m1 = 0.f;
        return r;
    }
};
struct TIQ2XS {
    using Block = block_iq2_xs;
    static constexpr int LUT = GL_GRID + 2 * 512; static constexpr Off OFF = NONE;
    static uint32_t lut(int i) { return grid64_lut<iq2xs_grid, 512>(i); }
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::int4 q4 = load16_a2(reinterpret_cast<const uint8_t*>(b->qs) + 16 * (g >> 1));
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t e = (word_of(q4, 2 * (g & 1) + (l >> 1)) >> (16 * (l & 1))) & 0xffff, gi = e & 511;
            const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(tab + GL_GRID + 2 * gi);
            grid_group_k(tab, w.x(), w.y(), e >> 9, v, l);
        }
        r.lo = i4(v); r.hi = i4(v + 4);
        const uint32_t sc = b->scales[g];
        const float d = (float) b->d;
        r.d0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f;
        r.d1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        r.m0 = r.m1 = 0.f;
        return r;
    }
};
struct TIQ2S {
    using Block = block_iq2_s;
    static constexpr int LUT = GL_GRID + 2 * 1024; static constexpr Off OFF = NONE;
    static uint32_t lut(int i) { return grid64_lut<iq2s_grid, 1024>(i); }
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::int4 q4 = load16_a2(b->qs + 16 * (g >> 2)), s4 = load16_a2(b->qs + 32 + 16 * (g >> 2)), hs = load16_a2(b->qh);
        const uint32_t qw = word_of(q4, g & 3), sw = word_of(s4, g & 3);
        const uint32_t qh = (word_of(hs, g >> 2) >> (8 * (g & 3))) & 0xff, sc = (word_of(hs, 2 + (g >> 2)) >> (8 * (g & 3))) & 0xff;
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t gi = ((qw >> (8 * l)) & 0xff) | ((qh << (8 - 2 * l)) & 0x300);
            const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(tab + GL_GRID + 2 * gi);
            grid_group(tab, w.x(), w.y(), (sw >> (8 * l)) & 0xff, v, l);
        }
        r.lo = i4(v); r.hi = i4(v + 4);
        const float d = (float) b->d;
        r.d0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f;
        r.d1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        r.m0 = r.m1 = 0.f;
        return r;
    }
};
// IQ1_M: a group of 8 values is a grid entry of {-1, 0, 1} plus a delta of +-1/8, read in the grid's GPU
// form (iq1s_grid_gpu: an entry's values + 1 as nibbles, values 0-3 low, 4-7 high), so q in 0..2 and o = delta - 1
constexpr int IQ1_LUT = 2048;
inline void iq1_group(const uint32_t* tab, uint32_t idx, int* v, int l) {
    const uint32_t e = tab[idx];
    v[2 * l] = int(e & 0x0f0f0f0f);
    v[2 * l + 1] = int((e >> 4) & 0x0f0f0f0f);
}
struct TIQ1M {
    using Block = block_iq1_m;
    static constexpr int LUT = IQ1_LUT; static constexpr Off OFF = IQ1;
    static uint32_t lut(int i) { return iq1s_grid_gpu[i]; }
    static W load(const Block* b, int g, const uint32_t* tab) {   // 56-byte blocks: 4-byte aligned
        W r;
        const uint32_t qs = reinterpret_cast<const uint32_t*>(b->qs)[g];
        const uint32_t qh = reinterpret_cast<const uint16_t*>(b->qh)[g];   // two bytes: groups 0-1, 2-3
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(b->scales);
        const uint32_t s01 = uint32_t(sc[0]) | uint32_t(sc[1]) << 16, s23 = uint32_t(sc[2]) | uint32_t(sc[3]) << 16;
        // the block scale: the top nibbles of the four scale words, as fp16
        const uint32_t dbits = (s01 >> 12 & 0xf) | (s01 >> 24 & 0xf0) | (s23 >> 4 & 0xf00) | (s23 >> 16 & 0xf000);
        const float d = h2f(dbits);
        const uint32_t sw = sc[g >> 1] >> (6 * (g & 1));
        r.d0 = d * (float) (2 * (sw & 7) + 1);
        r.d1 = d * (float) (2 * ((sw >> 3) & 7) + 1);
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t hb = (qh >> (8 * (l >> 1) + 4 * (l & 1))) & 0xf;   // the group's 3 high index bits, its delta's sign
            iq1_group(tab, ((qs >> (8 * l)) & 0xff) | ((hb & 7) << 8), v, l);
            r.o[l] = (hb & 8 ? -IQ1M_DELTA : IQ1M_DELTA) - 1.0f;
        }
        r.lo = i4(v); r.hi = i4(v + 4);
        r.m0 = r.m1 = 0.f;
        return r;
    }
};

// lane g's dot with the column's sub-block (the column's quants, scale and sum)
template <Off O>
inline float dot(const W& r, const X& x) {
    if constexpr (O == NONE)
        return x.d * (r.d0 * (float) dp4a4(r.lo, x.lo, 0) + r.d1 * (float) dp4a4(r.hi, x.hi, 0));
    else if constexpr (O == MIN32)   // the mins against the quants' exact sum (Q8_1's s is that sum rounded to fp16)
        return x.d * (r.d0 * (float) dp4a4(r.hi, x.hi, dp4a4(r.lo, x.lo, 0)) - r.m0 * (float) (sum4(x.lo) + sum4(x.hi)));
    else if constexpr (O == MIN16)
        return x.d * (r.d0 * (float) dp4a4(r.lo, x.lo, 0) - r.m0 * (float) sum4(x.lo) + r.d1 * (float) dp4a4(r.hi, x.hi, 0) - r.m1 * (float) sum4(x.hi));
    else {   // IQ1: d x (sum q x + o_k x sum of group k's activations), groups of 8 = two words
        auto grp = [&](int a, int b, int xa, int xb, float o) { return (float) dp4a(b, xb, dp4a(a, xa, 0)) + o * (float) dp4a(0x01010101, xb, dp4a(0x01010101, xa, 0)); };
        const float lo = grp(r.lo.x(), r.lo.y(), x.lo.x(), x.lo.y(), r.o[0]) + grp(r.lo.z(), r.lo.w(), x.lo.z(), x.lo.w(), r.o[1]);
        const float hi = grp(r.hi.x(), r.hi.y(), x.hi.x(), x.hi.y(), r.o[2]) + grp(r.hi.z(), r.hi.w(), x.hi.z(), x.hi.w(), r.o[3]);
        return x.d * (r.d0 * lo + r.d1 * hi);
    }
}
// value i (0..31) of lane g's sub-block, as the format defines it
template <Off O>
inline float value(const W& r, int i) {
    const sycl::int4 w = i < 16 ? r.lo : r.hi;
    const int q = int(int8_t(word_of(w, (i & 15) >> 2) >> (8 * (i & 3))));
    const float d = i < 16 ? r.d0 : r.d1;
    if constexpr (O == IQ1) return d * ((float) q + r.o[i >> 3]);
    else if constexpr (O == NONE) return d * (float) q;
    else return d * (float) q - (i < 16 ? r.m0 : r.m1);
}

// ---- 32-value types: a block a lane, F::load32(block, tab) -> W (d0 = d1, no offset)
struct TQ80 {
    using Block = block_q8_0;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    static W load32(const Block* b, const uint32_t*) {
        W r;
        r.lo = load16_a2(b->qs); r.hi = load16_a2(b->qs + 16);
        r.d0 = r.d1 = (float) b->d; r.m0 = r.m1 = 0.f;
        return r;
    }
};
struct TIQ4NL {
    using Block = block_iq4_nl;
    static constexpr int LUT = 256;
    static uint32_t lut(int i) { return iq4_tab_entry(i); }
    static W load32(const Block* b, const uint32_t* tab) {
        W r;
        iq4_16(tab, load16_a2(b->qs), r);
        r.d0 = r.d1 = (float) b->d; r.m0 = r.m1 = 0.f;
        return r;
    }
};

}  // namespace nsq
