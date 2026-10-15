// nhs.cpp: the nemotronh silo (nhs.h) - a MoE layer's decode work on the GPU. The routing in one launch (a
// work-group a row), then all of a layer's chosen experts in one launch per projection: the deepseek4 silo's scheme
// for the up projections (8 lanes a 256-block, a lane a sub-block of 32, dp4a against the split Q8 activations, the
// grids in local memory; copied, with IQ2_XS and IQ4_XS added and the expert's matrix found by its id on the GPU),
// and a 32-value step for the down projections, whose 2,688 inputs no 256-block divides (Q2_0, IQ4_NL, Q8_0).
#include "ns.h"
#include "ns_internal.hpp"
#include "nhs.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>

// ggml's grids (iq2xs_grid, iq2s_grid, iq3xxs_grid, kvalues_iq4nl)
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
};
inline XA xcol(const int8_t* xs, int n_in, int ncols, int j) {
    const sycl::half* d = reinterpret_cast<const sycl::half*>(xs + std::size_t(ncols) * n_in);
    const int16_t* sm = reinterpret_cast<const int16_t*>(d + std::size_t(ncols) * (n_in / 32));
    return XA{xs + std::size_t(j) * n_in, d + std::size_t(j) * (n_in / 32), sm + std::size_t(j) * (n_in / 16)};
}
struct X { sycl::int4 u0, u1; float d; };
inline X xload(const XA& xa, int b) { return X{xa.q16(b, 0), xa.q16(b, 1), xa.dd(b)}; }

inline uint32_t ksign(uint32_t b7) { return b7 | ((sycl::popcount(b7) & 1u) << 7); }
inline uint32_t sign_mask4(uint32_t b) { return ((b * 0x00204081u) & 0x01010101u) * 0xffu; }   // bit i -> 0xff in byte i
inline void grid_group(sycl::uint2 e, uint32_t sb, int* v, int l) {
    const uint32_t m0 = sign_mask4(sb & 0xf), m1 = sign_mask4((sb >> 4) & 0xf);
    v[2 * l] = int((e.x() ^ m0) + (m0 & 0x01010101u));
    v[2 * l + 1] = int((e.y() ^ m1) + (m1 & 0x01010101u));
}

// ---- the 256-value blocks (the up projections, 1,024 inputs): lane g of 8 = sub-block g

// IQ3_XXS (98-byte blocks): eight grid entries of 4 bytes, signs 7 bits a pair of entries, the scale in the top bits
struct WideIQ3XXS {
    using Block = block_iq3_xxs;
    static constexpr int LUT = 256;
    static uint32_t lut(int i) { return iq3xxs_grid[i]; }
    struct W { sycl::int4 lo, hi; float s0, s1; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const sycl::int4 q4 = load16_a2(p + 2 + 16 * (g >> 1));
        const uint32_t i0 = word_of(q4, 2 * (g & 1)), i1 = word_of(q4, 2 * (g & 1) + 1);
        const uint32_t a1 = word_of(load16_a2(p + 2 + 64 + 16 * (g >> 2)), g & 3);
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t iw = l < 2 ? i0 : i1, sh = 16 * (l & 1);
            grid_group(sycl::uint2(tab[(iw >> sh) & 0xff], tab[(iw >> (sh + 8)) & 0xff]), ksign((a1 >> (7 * l)) & 127), v, l);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.s0 = r.s1 = (float) b->d * (0.5f + (float) (a1 >> 28)) * 0.5f;
        return r;
    }
};

// IQ2_S (82-byte blocks): four grid entries of 8 bytes (10-bit indices), sign bytes, a scale nibble a half
struct WideIQ2S {
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
};

// IQ2_XS (74-byte blocks: d, qs [32] uint16 - a 9-bit grid index and 7 sign bits each -, scales [8]): sub-block g =
// qs[4g .. 4g + 4), its two scale nibbles (first 16 values, last 16)
struct WideIQ2XS {
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
        const uint32_t sc = p[2 + 64 + g];
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q = ((l < 2 ? w0 : w1) >> (16 * (l & 1))) & 0xffff;
            grid_group(grid[q & 511], ksign(q >> 9), v, l);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        const float d = (float) b->d;
        r.s0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f;
        r.s1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        return r;
    }
};

// IQ4_XS (136-byte blocks: d, scales_h, scales_l [4], qs [128]): sub-block g = qs[16g .. 16g + 16) (low nibbles the
// first 16 values, high the last), its 6-bit scale; a byte's two codes to two int8 values through a 256-entry table
struct WideIQ4XS {
    using Block = block_iq4_xs;
    static constexpr int LUT = 256;
    static uint32_t lut(int i) { return uint32_t(uint8_t(kvalues_iq4nl[i & 15])) | (uint32_t(uint8_t(kvalues_iq4nl[i >> 4])) << 8); }
    struct W { sycl::int4 lo, hi; float s0, s1; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const sycl::int4 q = load16_a2(p + 8 + 16 * g);
        uint32_t lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t c = word_of(q, i);
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
        const uint32_t sh = *reinterpret_cast<const uint16_t*>(p + 2);
        const uint32_t ls = ((p[4 + g / 2] >> (4 * (g % 2))) & 0xf) | (((sh >> (2 * g)) & 3) << 4);
        r.s0 = r.s1 = (float) b->d * (float) ((int) ls - 32);
        return r;
    }
};

// Q8_0 as 256-blocks (eight 34-byte blocks): lane g block g
struct Q8x8 { block_q8_0 b[8]; };
struct WideQ8_0 {
    using Block = Q8x8;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    struct W { sycl::int4 lo, hi; float s0, s1; };
    static W load(const Block* b, int g, const uint32_t*) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b) + 34 * g;
        const float d = (float) *reinterpret_cast<const sycl::half*>(p);
        return W{load16_a2(p + 2), load16_a2(p + 18), d, d};
    }
};

template <typename F>
inline float wdot(const typename F::W& r, const X& v) { return v.d * (r.s0 * (float) dp4a4(r.lo, v.u0, 0) + r.s1 * (float) dp4a4(r.hi, v.u1, 0)); }

template <typename F>
void fill_tab(uint32_t* tab, const sycl::nd_item<3>& item) {
    if constexpr (F::LUT <= 1) return;
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    for (int i = warp * WARP + lane; i < F::LUT; i += WARPS * WARP) tab[i] = F::lut(i);
    sycl::group_barrier(item.get_group());
}

int multi_groups() {   // NS_NH_MULTI_GROUPS: the work-groups of a layer's experts in one launch (default 1024)
    static const int v = std::getenv("NS_NH_MULTI_GROUPS") ? std::max(1, std::atoi(std::getenv("NS_NH_MULTI_GROUPS"))) : 1024;
    return v;
}

// a sub-group a run of consecutive rows (mostly one entry's: one column), its lanes' NB activation blocks loaded once
// a column, the weights streamed (the deepseek4 silo's multi_cached)
template <typename F, int NB>
void multi256(sycl::queue& q, const uint8_t* base, int64_t stride, const int32_t* ids, const int32_t* cols, int entries, const void* xs, int ncols,
              float* y, int n_in, int n_out) {
    const auto* x = static_cast<const int8_t*>(xs);
    const std::size_t rows = std::size_t(entries) * n_out;
    const unsigned blocks = (unsigned) std::min<std::size_t>((rows + WARPS - 1) / WARPS, std::size_t(multi_groups()));
    const std::size_t nsg = std::size_t(blocks) * WARPS, chunk = (rows + nsg - 1) / nsg;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                           uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
                           fill_tab<F>(t, it);
                           const int lane = int(it.get_local_id(2)), warp = int(it.get_local_id(1));
                           const int g = lane & 7, sub = lane >> 3;
                           const int bpr = n_in / QK;
                           auto sg = it.get_sub_group();
                           const std::size_t r0 = (std::size_t(it.get_group(2)) * WARPS + warp) * chunk, r1 = std::min(rows, r0 + chunk);
                           int cur = -1;
                           X xv[NB];
                           for (std::size_t r = r0; r < r1; ++r) {
                               const int e = int(r / n_out), row = int(r % n_out);
                               const int col = cols[e];
                               if (col != cur) {
                                   const XA xa = xcol(x, n_in, ncols, col);
#pragma unroll
                                   for (int u = 0; u < NB; ++u)
                                       if (sub + 4 * u < bpr) xv[u] = xload(xa, (sub + 4 * u) * (QK / Q8K) + g);
                                   cur = col;
                               }
                               const auto* wr = reinterpret_cast<const typename F::Block*>(base + std::size_t(ids[e]) * stride) + std::size_t(row) * bpr;
                               float v = 0.f;
#pragma unroll
                               for (int u = 0; u < NB; ++u)
                                   if (sub + 4 * u < bpr) v += wdot<F>(F::load(wr + sub + 4 * u, g, t), xv[u]);
#pragma unroll
                               for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
                               if (lane == 0) y[r] = v;
                           }
                       });
    });
}

template <typename F>
void multi_wide(sycl::queue& q, const uint8_t* base, int64_t stride, const int32_t* ids, const int32_t* cols, int entries, const void* xs, int ncols,
                float* y, int n_in, int n_out) {
    const int bpr = n_in / QK;
    if (bpr <= 4) multi256<F, 1>(q, base, stride, ids, cols, entries, xs, ncols, y, n_in, n_out);
    else if (bpr <= 8) multi256<F, 2>(q, base, stride, ids, cols, entries, xs, ncols, y, n_in, n_out);
    else multi256<F, 4>(q, base, stride, ids, cols, entries, xs, ncols, y, n_in, n_out);
}

// ---- the 32-value step (the down projections, 2,688 inputs = 84 sub-blocks): a lane a sub-block, the 32 lanes of a
// sub-group over the row's sub-blocks in turn. The weights in the packed layout (ns_nhs_pack: a row's codes first,
// 16-byte aligned, then its scales), so a lane's codes are one aligned load

inline int packed_qs(int type, int n_in) { return type == 8 ? n_in : type == 20 ? n_in / 2 : n_in / 4; }
inline int packed_scales(int type, int n_in) { return type == 42 ? n_in / 64 : n_in / 32; }
inline int64_t packed_row(int type, int n_in) { return (packed_qs(type, n_in) + 2 * packed_scales(type, n_in) + 15) / 16 * 16; }

// Q2_0 (codes 0..3, value (code - 1) d over 64 values): sub-block s = 8 code bytes at 8 s, the scale of block s / 2;
// the codes multiplied unsigned (a byte's four 2-bit codes spread to four bytes by shifts), less the activations' sum
struct NarrowQ2_0 {
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    static float dot(const uint8_t* row, int qsb, int s, const uint32_t*, const X& x, int xs) {
        const sycl::uint2 c = *reinterpret_cast<const sycl::uint2*>(row + 8 * s);
        const float d = (float) reinterpret_cast<const sycl::half*>(row + qsb)[s >> 1];
        auto spread = [](uint32_t b) { return int((b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303u); };
        const uint32_t a = c.x(), b = c.y();
        const sycl::int4 lo(spread(a & 0xff), spread((a >> 8) & 0xff), spread((a >> 16) & 0xff), spread(a >> 24));
        const sycl::int4 hi(spread(b & 0xff), spread((b >> 8) & 0xff), spread((b >> 16) & 0xff), spread(b >> 24));
        return d * x.d * (float) (dp4a4(hi, x.u1, dp4a4(lo, x.u0, 0)) - xs);
    }
};

// IQ4_NL (16 code bytes a sub-block at 16 s - low nibbles the first 16 values, high the last 16 -, a scale each): a
// byte's two codes to their two int8 values through a 256-entry table
struct NarrowIQ4NL {
    static constexpr int LUT = 256;
    static uint32_t lut(int i) { return uint32_t(uint8_t(kvalues_iq4nl[i & 15])) | (uint32_t(uint8_t(kvalues_iq4nl[i >> 4])) << 8); }
    static float dot(const uint8_t* row, int qsb, int s, const uint32_t* tab, const X& x, int) {
        const sycl::uint4 c = *reinterpret_cast<const sycl::uint4*>(row + 16 * s);
        const float d = (float) reinterpret_cast<const sycl::half*>(row + qsb)[s];
        uint32_t l[4], h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t w = i == 0 ? c.x() : i == 1 ? c.y() : i == 2 ? c.z() : c.w();
            uint32_t a = 0, b = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t e = tab[(w >> (8 * k)) & 0xff];
                a |= (e & 0xff) << (8 * k);
                b |= ((e >> 8) & 0xff) << (8 * k);
            }
            l[i] = a;
            h[i] = b;
        }
        const sycl::int4 lo{int(l[0]), int(l[1]), int(l[2]), int(l[3])}, hi{int(h[0]), int(h[1]), int(h[2]), int(h[3])};
        return d * x.d * (float) dp4a4(hi, x.u1, dp4a4(lo, x.u0, 0));
    }
};

// Q8_0 (32 int8 a sub-block at 32 s, a scale each)
struct NarrowQ8_0 {
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    static float dot(const uint8_t* row, int qsb, int s, const uint32_t*, const X& x, int) {
        const sycl::int4* c = reinterpret_cast<const sycl::int4*>(row + 32 * s);
        const float d = (float) reinterpret_cast<const sycl::half*>(row + qsb)[s];
        return d * x.d * (float) dp4a4(c[1], x.u1, dp4a4(c[0], x.u0, 0));
    }
};

template <typename F>
void multi32(sycl::queue& q, int type, const uint8_t* base, int64_t stride, const int32_t* ids, const int32_t* cols, int entries, const void* xs,
             int ncols, float* y, int n_in, int n_out) {
    const auto* x = static_cast<const int8_t*>(xs);
    const std::size_t rows = std::size_t(entries) * n_out;
    const unsigned blocks = (unsigned) std::min<std::size_t>((rows + WARPS - 1) / WARPS, std::size_t(multi_groups()));
    const std::size_t nsg = std::size_t(blocks) * WARPS, chunk = (rows + nsg - 1) / nsg;
    const int nsb = n_in / 32, qsb = packed_qs(type, n_in);
    const int64_t rb = packed_row(type, n_in);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                           uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
                           fill_tab<F>(t, it);
                           const int lane = int(it.get_local_id(2)), warp = int(it.get_local_id(1));
                           auto sg = it.get_sub_group();
                           const std::size_t r0 = (std::size_t(it.get_group(2)) * WARPS + warp) * chunk, r1 = std::min(rows, r0 + chunk);
                           int cur = -1;
                           X xv[3];
                           int xsum[3] = {0, 0, 0};
                           for (std::size_t r = r0; r < r1; ++r) {
                               const int e = int(r / n_out), row = int(r % n_out);
                               const int col = cols[e];
                               if (col != cur) {
                                   const XA xa = xcol(x, n_in, ncols, col);
#pragma unroll
                                   for (int u = 0; u < 3; ++u) {
                                       const int s = lane + 32 * u;
                                       if (s < nsb) {
                                           xv[u] = xload(xa, s);
                                           xsum[u] = xa.sm[2 * s] + xa.sm[2 * s + 1];
                                       }
                                   }
                                   cur = col;
                               }
                               const uint8_t* wr = base + std::size_t(ids[e]) * stride + std::size_t(row) * rb;
                               float v = 0.f;
#pragma unroll
                               for (int u = 0; u < 3; ++u) {
                                   const int s = lane + 32 * u;
                                   if (s < nsb) v += F::dot(wr, qsb, s, t, xv[u], xsum[u]);
                               }
#pragma unroll
                               for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
                               if (lane == 0) y[r] = v;
                           }
                       });
    });
}

}  // namespace

extern "C" {

int ns_nhs_route(ns_gpu* g, const float* logits, const float* bias, int64_t T, int64_t E, int64_t K, int norm, float scale, int32_t* ids, float* w,
                 int32_t* cols) {
    NS_TRY
    if (T <= 0) return 0;
    if (E > 1024 || K > 64 || K > E) return ns_fail("ns_nhs_route: at most 1024 experts, 64 chosen");
    constexpr int WG = 256;
    // a work-group a row: each work-item its experts' scores; K rounds of an arg max over the group (ties to the lower
    // index), the chosen one's score taken out
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sel(sycl::range<1>(1024), h);
        sycl::local_accessor<float, 1> pr(sycl::range<1>(1024), h);
        sycl::local_accessor<int, 1> pick(sycl::range<1>(64), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t r = it.get_group(0);
            const int l = (int) it.get_local_id(0);
            const float* lg = logits + r * E;
            for (int64_t i = l; i < E; i += WG) {
                const float p = 1.0f / (1.0f + sycl::exp(-lg[i]));
                pr[i] = p;
                sel[i] = p + bias[i];
            }
            sycl::group_barrier(it.get_group());
            for (int k = 0; k < K; ++k) {
                float best = -INFINITY;
                int bi = (int) E;
                for (int64_t i = l; i < E; i += WG) {
                    if (sel[i] > best) {   // the lowest index of the largest among this item's (indices ascending)
                        best = sel[i];
                        bi = (int) i;
                    }
                }
                const float m = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
                const int mi = sycl::reduce_over_group(it.get_group(), best == m ? bi : (int) E, sycl::minimum<int>());
                if (l == 0) {
                    pick[k] = mi;
                    sel[mi] = -INFINITY;
                }
                sycl::group_barrier(it.get_group());
            }
            if (l == 0) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k) s += pr[pick[k]];
                const float f = (norm && s > 0.0f ? 1.0f / s : 1.0f) * scale;
                for (int k = 0; k < K; ++k) {
                    ids[r * K + k] = pick[k];
                    w[r * K + k] = pr[pick[k]] * f;
                    cols[r * K + k] = (int32_t) r;
                }
            }
        });
    });
    return 0;
    NS_CATCH
}

size_t ns_nhs_q8_bytes(int64_t n_in, int64_t ncols) { return (size_t) (ncols * n_in + ncols * (n_in / 32) * 2 + ncols * (n_in / 16) * 2); }

// x [ncols, n_in] float32 -> the split layout: a sub-group of 32 a block (its amax / 127 rounded to fp16 first, as
// Q8_1's scale), each 16 quants' sum
int ns_nhs_quantize_q8(ns_gpu* g, const float* x, void* xs, int64_t n_in, int64_t ncols) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_nhs_quantize_q8: n_in a multiple of 32");
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

int ns_nhs_packed(int type, int64_t n_in) { return n_in % QK != 0 && ns_nhs_multi_supported(type, n_in); }

int64_t ns_nhs_packed_row_bytes(int type, int64_t n_in) { return packed_row(type, (int) n_in); }

// rows of 18 / 34-byte blocks -> the packed layout: a row's code bytes in order, then its scales, padded to 16 bytes
int ns_nhs_pack(ns_gpu* g, int type, const void* src, void* dst, int64_t rows, int64_t n_in) {
    NS_TRY
    if (!ns_nhs_packed(type, n_in)) return ns_fail("ns_nhs_pack: not a packed type at this width");
    const int bs = type == 42 ? 64 : 32, bb = type == 8 ? 34 : 18, cb = bb - 2;
    const int64_t nb = n_in / bs, rb = packed_row(type, (int) n_in), qsb = packed_qs(type, (int) n_in);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(dst);
    g->q.parallel_for(sycl::range<1>((size_t) (rows * nb)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / nb, k = id[0] % nb;
        const uint8_t* b = s + (r * nb + k) * bb;
        uint8_t* o = d + r * rb;
        o[qsb + 2 * k] = b[0];
        o[qsb + 2 * k + 1] = b[1];
        for (int i = 0; i < cb; ++i) o[k * cb + i] = b[2 + i];
        if (k == 0) for (int64_t i = qsb + 2 * nb; i < rb; ++i) o[i] = 0;
    });
    return 0;
    NS_CATCH
}

int ns_nhs_packed_f16(ns_gpu* g, int type, const void* src, uint16_t* dst, int64_t rows, int64_t n_in) {
    NS_TRY
    if (!ns_nhs_packed(type, n_in)) return ns_fail("ns_nhs_packed_f16: not a packed type at this width");
    const int64_t rb = packed_row(type, (int) n_in), qsb = packed_qs(type, (int) n_in), q4 = n_in / 4;
    const uint8_t* s = static_cast<const uint8_t*>(src);
    sycl::half* y = reinterpret_cast<sycl::half*>(dst);
    // a work-item four consecutive values (neighbouring items neighbouring values: the writes coalesce)
    g->q.parallel_for(sycl::range<1>((size_t) (rows * q4)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / q4, j = (id[0] % q4) * 4;
        const uint8_t* row = s + r * rb;
        const sycl::half* sc = reinterpret_cast<const sycl::half*>(row + qsb);
        sycl::half4 o;
        if (type == 42) {
            const float d = (float) sc[j / 64];
            const uint32_t c = row[j / 4];
            o = sycl::half4((sycl::half) (d * (float) ((int) (c & 3) - 1)), (sycl::half) (d * (float) ((int) ((c >> 2) & 3) - 1)),
                            (sycl::half) (d * (float) ((int) ((c >> 4) & 3) - 1)), (sycl::half) (d * (float) ((int) (c >> 6) - 1)));
        } else if (type == 20) {
            const int64_t sb = j / 32, i = j % 32;
            const float d = (float) sc[sb];
            const uint8_t* c = row + 16 * sb;
            const int sh = i < 16 ? 0 : 4, b0 = (int) (i & 15);
            o = sycl::half4((sycl::half) (d * (float) kvalues_iq4nl[(c[b0] >> sh) & 15]), (sycl::half) (d * (float) kvalues_iq4nl[(c[b0 + 1] >> sh) & 15]),
                            (sycl::half) (d * (float) kvalues_iq4nl[(c[b0 + 2] >> sh) & 15]), (sycl::half) (d * (float) kvalues_iq4nl[(c[b0 + 3] >> sh) & 15]));
        } else {
            const float d = (float) sc[j / 32];
            const int8_t* c = reinterpret_cast<const int8_t*>(row) + j;
            o = sycl::half4((sycl::half) (d * (float) c[0]), (sycl::half) (d * (float) c[1]), (sycl::half) (d * (float) c[2]), (sycl::half) (d * (float) c[3]));
        }
        *reinterpret_cast<sycl::half4*>(y + r * n_in + j) = o;
    });
    return 0;
    NS_CATCH
}

int ns_nhs_multi_supported(int type, int64_t n_in) {
    const bool wide = n_in % QK == 0 && (type == 8 || type == 17 || type == 18 || type == 22 || type == 23);
    const bool narrow = n_in <= 96 * 32 && ((type == 42 && n_in % 64 == 0) || ((type == 20 || type == 8) && n_in % 32 == 0));
    return wide || narrow;
}

int ns_nhs_multi(ns_gpu* g, int type, const void* base, int64_t stride, const int32_t* ids, const int32_t* cols, int64_t entries, const void* xs,
                 int64_t ncols, float* y, int64_t n_in, int64_t n_out) {
    NS_TRY
    if (!ns_nhs_multi_supported(type, n_in)) return ns_fail("ns_nhs_multi: ggml type " + std::to_string(type) + " at " + std::to_string(n_in) + " inputs");
    if (entries < 1) return 0;
    const uint8_t* b = static_cast<const uint8_t*>(base);
    const int E = (int) entries, NC = (int) ncols, K = (int) n_in, N = (int) n_out;
    if (n_in % QK == 0 && type != 20 && type != 42) {
        switch (type) {
            case 8: multi_wide<WideQ8_0>(g->q, b, stride, ids, cols, E, xs, NC, y, K, N); break;
            case 17: multi_wide<WideIQ2XS>(g->q, b, stride, ids, cols, E, xs, NC, y, K, N); break;
            case 18: multi_wide<WideIQ3XXS>(g->q, b, stride, ids, cols, E, xs, NC, y, K, N); break;
            case 22: multi_wide<WideIQ2S>(g->q, b, stride, ids, cols, E, xs, NC, y, K, N); break;
            default: multi_wide<WideIQ4XS>(g->q, b, stride, ids, cols, E, xs, NC, y, K, N); break;
        }
        return 0;
    }
    switch (type) {
        case 42: multi32<NarrowQ2_0>(g->q, type, b, stride, ids, cols, E, xs, NC, y, K, N); break;
        case 20: multi32<NarrowIQ4NL>(g->q, type, b, stride, ids, cols, E, xs, NC, y, K, N); break;
        default: multi32<NarrowQ8_0>(g->q, type, b, stride, ids, cols, E, xs, NC, y, K, N); break;
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
