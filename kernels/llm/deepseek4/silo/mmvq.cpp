// mmvq.cpp: the deepseek4 engine's decode products for its experts (ds4.h) - 1..8 columns of Q8 activations against
// a layer's experts at once: IQ2_XXS and IQ2_S (gate | up), IQ3_XXS and MXFP4 (down: no shared grouped kernel takes
// them). The scheme is
// the deepseek2 silo's (copied, not shared): 8 lanes a 256-block, a lane a sub-block of 32 values, the activations in
// the split Q8 layout (a column's quants contiguous and 16-byte aligned), dp4a against them, the grids in local memory.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds4.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>

// ggml's grids (iq2xxs_grid, iq3xxs_grid, ksigns_iq2xs)
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

// ---- IQ2_XXS (66-byte blocks, 2-byte aligned): lane g = sub-block g - four grid entries (two words of unsigned
// bytes each) with their signs applied bytewise. The signs are computed, not looked up: ksigns_iq2xs[b] is b with
// its parity as bit 7, and a nibble of it widens to byte masks by a multiply. Local memory holds only the grid, an
// entry's two words read at once (B70 at the first version's 20 local reads a block: 293 GB/s)
inline uint32_t ksign(uint32_t b7) { return b7 | ((sycl::popcount(b7) & 1u) << 7); }
inline uint32_t sign_mask4(uint32_t b) { return ((b * 0x00204081u) & 0x01010101u) * 0xffu; }   // bit i -> 0xff in byte i
inline void grid_group(sycl::uint2 e, uint32_t sb, int* v, int l) {
    const uint32_t m0 = sign_mask4(sb & 0xf), m1 = sign_mask4((sb >> 4) & 0xf);
    v[2 * l] = int((e.x() ^ m0) + (m0 & 0x01010101u));
    v[2 * l + 1] = int((e.y() ^ m1) + (m1 & 0x01010101u));
}
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


// ---- IQ3_XXS (98-byte blocks, 2-byte aligned): lane g = sub-block g - eight grid entries of 4 unsigned bytes
// (iq3xxs_grid, a byte index each: qs[8g .. 8g + 8)) with their signs (7 bits per pair of entries, from the
// sub-block's word at byte 64 + 4g, whose top 4 bits are its scale; computed as above)
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
        const sycl::int4 q4 = load16_a2(p + 2 + 16 * (g >> 1));
        const uint32_t i0 = word_of(q4, 2 * (g & 1)), i1 = word_of(q4, 2 * (g & 1) + 1);   // the 8 grid indices
        const uint32_t a1 = word_of(load16_a2(p + 2 + 64 + 16 * (g >> 2)), g & 3);           // signs + scale
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

// ---- IQ2_S (82-byte blocks, 2-byte aligned): lane g = sub-block g - four grid entries (iq2s_grid, 10-bit indices:
// qs[4g + l] and two bits of qh[g]) with their sign bytes (qs[32 + 4g + l], no parity), two scales (nibbles of
// scales[g]: the first 16 values', the last 16's)
inline uint32_t load4_a2(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return uint32_t(h[0]) | (uint32_t(h[1]) << 16);
}
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

// ---- MXFP4 (17-byte blocks of 32: an E8M0 scale, 16 bytes of E2M1 codes - low nibbles values 0..15, high 16..31):
// eight blocks make the 256-value step, lane g block g. Its 16 code bytes come from byte-aligned word loads; a byte's
// two codes become their two (doubled) values through a 256-entry table; the scale is ggml's E8M0 at half
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
        // the 16 code bytes at p + 1: whole words from the 4-byte boundary below, shifted (the fifth word only when
        // the bytes reach into it - never past the block)
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

// ---- Q8_0 (34-byte blocks of 32: a half scale, 32 int8, 2-byte aligned): eight blocks make the 256-value step, lane
// g block g (the attention's grouped output o_a in one launch: its 8 groups each against its own slice)
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

// ---- Q5_K (176-byte blocks of 256, 16-byte aligned: d, dmin, 12 bytes of 6-bit scales and mins, qh [32], qs [128]):
// lane g = sub-block g - its 32 values' low nibbles (qs[32 (g / 2) + j], the high nibble for odd g) and bit g of
// qh[j] as unsigned 5-bit codes; value = d sc_g q - dmin m_g, so a dot is d sc_g (q . x) - dmin m_g (sum x) with the
// activations' sums the split layout carries (the shared expert's gate | up in one launch, and q_a)
struct WideQ5K {
    struct XT { sycl::int4 u0, u1; float d; int s; };
    static XT xl(int g, const XA& xa, int b0) { const X v = xload(g, xa, b0); return XT{v.u0, v.u1, v.d, v.s}; }
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
        // ggml's get_scale_min_k4
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
    static float dot(const W& r, const XT& v) { return v.d * (r.dsc * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)) - r.dmn * (float) v.s); }
};

int multi_groups() {   // NS_DS4_MULTI_GROUPS: the work-groups of a layer's experts in one launch (each fills its grid table once; default 1024)
    static const int v = std::getenv("NS_DS4_MULTI_GROUPS") ? std::max(1, std::atoi(std::getenv("NS_DS4_MULTI_GROUPS"))) : 1024;
    return v;
}

// One row's dot over its 256-blocks: 8 lanes a block (lane g its sub-block g), 4 blocks a sub-group step. A lane's
// blocks are loaded UNROLL at a time before any is used (the loads in flight together: one at a time, the first
// version, held the B70 at 290 GB/s)
#ifndef DS4_MULTI_UNROLL
#define DS4_MULTI_UNROLL 1
#endif
template <typename F>
inline float row_dot(const typename F::Block* wr, int blocks_per_row, const XA& xa, int lane, const uint32_t* tab) {
    constexpr int U = DS4_MULTI_UNROLL;
    const int g = lane & 7, sub = lane >> 3;
    float acc = 0.f;
    for (int k0 = sub; k0 < blocks_per_row; k0 += 4 * U) {
        typename F::W wv[U];
        typename F::XT xv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kbx = k0 + 4 * u;
            if (kbx < blocks_per_row) {
                wv[u] = F::load(wr + kbx, g, tab);
                xv[u] = F::xl(g, xa, kbx * (QK / Q8K));
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
            if (k0 + 4 * u < blocks_per_row) acc += F::dot(wv[u], xv[u]);
    }
    return acc;
}

template <typename F>
void fill_tab(uint32_t* tab, const sycl::nd_item<3>& item) {
    if constexpr (F::LUT <= 1) return;
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    for (int i = warp * WARP + lane; i < F::LUT; i += WARPS * WARP) tab[i] = F::lut(i);
    sycl::group_barrier(item.get_group());
}

// The same products with the activations held: a sub-group takes a run of consecutive rows (mostly one entry's, so
// one column), loads its lanes' NB activation blocks once - again only where the column changes - and streams the
// weights. The first scheme loaded a row's activations for every row: 4 KiB of them against its 1 KiB of IQ2_XXS
// weights (NS_DS4_MULTI_CACHED=0 runs it)
template <typename F, int NB>
void multi_cached(sycl::queue& q, const uint64_t* ptrs, const int32_t* cols, int entries, const void* xs, int ncols, float* y, int n_in, int n_out,
                  const int32_t* rstart, int total_rows) {
    const auto* x = static_cast<const int8_t*>(xs);
    const std::size_t rows = rstart ? std::size_t(total_rows) : std::size_t(entries) * n_out;
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
                           int e = 0, cur = -1;
                           typename F::XT xv[NB];
                           for (std::size_t r = r0; r < r1; ++r) {
                               int row;
                               if (rstart) {
                                   while (e + 1 < entries && int(r) >= rstart[e + 1]) ++e;
                                   row = int(r) - rstart[e];
                               } else {
                                   e = int(r / n_out);
                                   row = int(r % n_out);
                               }
                               const int col = cols[e];
                               if (col != cur) {
                                   const XA xa = xcol(x, n_in, ncols, col);
#pragma unroll
                                   for (int u = 0; u < NB; ++u)
                                       if (sub + 4 * u < bpr) xv[u] = F::xl(g, xa, (sub + 4 * u) * (QK / Q8K));
                                   cur = col;
                               }
                               const auto* wr = reinterpret_cast<const typename F::Block*>(ptrs[e]) + std::size_t(row) * bpr;
                               float v = 0.f;
#pragma unroll
                               for (int u = 0; u < NB; ++u)
                                   if (sub + 4 * u < bpr) v += F::dot(F::load(wr + sub + 4 * u, g, t), xv[u]);
#pragma unroll
                               for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
                               if (lane == 0) y[r] = v;
                           }
                       });
    });
}

bool cached_on() {
    static const bool v = !(std::getenv("NS_DS4_MULTI_CACHED") && std::getenv("NS_DS4_MULTI_CACHED")[0] == '0');
    return v;
}

// y [entries, n_out]: entry e = the matrix at ptrs[e] against column cols[e]; rows of all entries in one stride
// rstart (null: every entry n_out rows): entry e's rows are [rstart[e], rstart[e + 1]) of y - matrices of different
// heights in one launch (a row finds its entry by a scan: few entries)
template <typename F>
void multi(sycl::queue& q, const uint64_t* ptrs, const int32_t* cols, int entries, const void* xs, int ncols, float* y, int n_in, int n_out,
           const int32_t* rstart = nullptr, int total_rows = 0) {
    // the IQ types gain (B70: IQ2_XXS 291 -> 312 GB/s, IQ3_XXS 303 -> 317); Q8_0 and Q5_K lose, and keep the first scheme
    if (cached_on() && F::LUT > 0) {
        const int bpr = n_in / QK;
        if (bpr <= 4) return multi_cached<F, 1>(q, ptrs, cols, entries, xs, ncols, y, n_in, n_out, rstart, total_rows);
        if (bpr <= 8) return multi_cached<F, 2>(q, ptrs, cols, entries, xs, ncols, y, n_in, n_out, rstart, total_rows);
        if (bpr <= 16) return multi_cached<F, 4>(q, ptrs, cols, entries, xs, ncols, y, n_in, n_out, rstart, total_rows);
    }
    const auto* x = static_cast<const int8_t*>(xs);
    const std::size_t rows = rstart ? std::size_t(total_rows) : std::size_t(entries) * n_out;
    const unsigned blocks = (unsigned) std::min<std::size_t>((rows + WARPS - 1) / WARPS, std::size_t(multi_groups()));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                           uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
                           fill_tab<F>(t, it);
                           const int lane = int(it.get_local_id(2)), warp = int(it.get_local_id(1));
                           const int bpr = n_in / QK;
                           auto sg = it.get_sub_group();
                           for (std::size_t r = std::size_t(it.get_group(2)) * WARPS + warp; r < rows; r += std::size_t(it.get_group_range(2)) * WARPS) {
                               int e, row;
                               if (rstart) {
                                   e = 0;
                                   while (e + 1 < entries && int(r) >= rstart[e + 1]) ++e;
                                   row = int(r) - rstart[e];
                               } else {
                                   e = int(r / n_out);
                                   row = int(r % n_out);
                               }
                               const auto* wr = reinterpret_cast<const typename F::Block*>(ptrs[e]) + std::size_t(row) * bpr;
                               float v = row_dot<F>(wr, bpr, xcol(x, n_in, ncols, cols[e]), lane, t);
#pragma unroll
                               for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
                               if (lane == 0) y[r] = v;
                           }
                       });
    });
}
}  // namespace

extern "C" {

size_t ns_ds4_q8_bytes(int64_t n_in, int64_t ncols) { return (size_t) (ncols * n_in + ncols * (n_in / 32) * 2 + ncols * (n_in / 16) * 2); }

// x [ncols, n_in] float32 -> the split layout: a sub-group of 32 a block (its amax / 127 rounded to fp16 first, as
// Q8_1's scale), each 16 quants' sum
int ns_ds4_quantize_q8(ns_gpu* g, const float* x, void* xs, int64_t n_in, int64_t ncols) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_ds4_quantize_q8: n_in a multiple of 32");
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

int ns_ds4_multi_supported(int type, int64_t n_in) { return (type == 8 || type == 13 || type == 16 || type == 18 || type == 22 || type == 39) && n_in % QK == 0; }

int ns_ds4_mmvq_multi(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* cols, int64_t entries, const void* xs, int64_t ncols, float* y,
                      int64_t n_in, int64_t n_out) {
    NS_TRY
    if (!ns_ds4_multi_supported(type, n_in)) return ns_fail("ns_ds4_mmvq_multi: ggml type " + std::to_string(type) + " at this width is not its own");
    if (entries < 1) return 0;
    if (type == 16) multi<WideIQ2XXS>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    else if (type == 18) multi<WideIQ3XXS>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    else if (type == 13) multi<WideQ5K>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    else if (type == 8) multi<WideQ8_0>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    else if (type == 22) multi<WideIQ2S>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    else multi<WideMXFP4>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    return 0;
    NS_CATCH
}

int ns_ds4_mmvq_multi_var(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* cols, const int32_t* rstart, int64_t entries, const void* xs,
                          int64_t ncols, float* y, int64_t n_in, int64_t total_rows) {
    NS_TRY
    if (!ns_ds4_multi_supported(type, n_in)) return ns_fail("ns_ds4_mmvq_multi_var: ggml type " + std::to_string(type) + " at this width is not its own");
    if (entries < 1 || total_rows < 1) return 0;
    const int E = (int) entries, NC = (int) ncols, K = (int) n_in, TR = (int) total_rows;
    switch (type) {
        case 8: multi<WideQ8_0>(g->q, ptrs, cols, E, xs, NC, y, K, 0, rstart, TR); break;
        case 13: multi<WideQ5K>(g->q, ptrs, cols, E, xs, NC, y, K, 0, rstart, TR); break;
        case 16: multi<WideIQ2XXS>(g->q, ptrs, cols, E, xs, NC, y, K, 0, rstart, TR); break;
        case 18: multi<WideIQ3XXS>(g->q, ptrs, cols, E, xs, NC, y, K, 0, rstart, TR); break;
        case 22: multi<WideIQ2S>(g->q, ptrs, cols, E, xs, NC, y, K, 0, rstart, TR); break;
        default: multi<WideMXFP4>(g->q, ptrs, cols, E, xs, NC, y, K, 0, rstart, TR); break;
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
