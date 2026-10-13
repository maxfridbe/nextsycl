// mmvq.cpp: the deepseek4 engine's decode products for its experts (ds4.h) - 1..8 columns of Q8 activations against
// a layer's experts at once: IQ2_XXS (gate | up) and IQ3_XXS (down: no shared grouped kernel takes it). The scheme is
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
// bytes each) with their signs (ksigns_iq2xs) applied bytewise. Local words: [16] the sign masks of four bits, [128]
// ksigns_iq2xs, then the grid as words
inline uint32_t sign_mask4(uint32_t b) { return ((b * 0x00204081u) & 0x01010101u) * 0xffu; }   // bit i -> 0xff in byte i
constexpr int GL_SIGNS = 0, GL_KSIGNS = 16, GL_GRID = 144;
inline void grid_group(const uint32_t* tab, uint32_t w0, uint32_t w1, uint32_t sb, int* v, int l) {
    const uint32_t m0 = tab[GL_SIGNS + (sb & 0xf)], m1 = tab[GL_SIGNS + ((sb >> 4) & 0xf)];
    v[2 * l] = int((w0 ^ m0) + (m0 & 0x01010101u));
    v[2 * l + 1] = int((w1 ^ m1) + (m1 & 0x01010101u));
}
struct WideIQ2XXS {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_iq2_xxs;
    static constexpr int LUT = GL_GRID + 2 * 256;
    static uint32_t lut(int i) {
        if (i < 16) return sign_mask4(uint32_t(i));
        if (i < GL_GRID) return uint32_t(ksigns_iq2xs[i - 16]);
        const uint64_t e = iq2xxs_grid[(i - GL_GRID) >> 1];
        return uint32_t((i - GL_GRID) & 1 ? e >> 32 : e);
    }
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::int4 q4 = load16_a2(reinterpret_cast<const uint8_t*>(b->qs) + 16 * (g >> 1));
        const uint32_t a0 = word_of(q4, 2 * (g & 1)), a1 = word_of(q4, 2 * (g & 1) + 1);   // grid bytes; signs + scale
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t gi = (a0 >> (8 * l)) & 0xff, sb = tab[GL_KSIGNS + ((a1 >> (7 * l)) & 127)];
            grid_group(tab, tab[GL_GRID + 2 * gi], tab[GL_GRID + 2 * gi + 1], sb, v, l);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.d = (float) b->d * (0.5f + (float) (a1 >> 28)) * 0.25f;
        return r;
    }
    static float dot(const W& r, const X& v) { return r.d * v.d * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};


// ---- IQ3_XXS (98-byte blocks, 2-byte aligned): lane g = sub-block g - eight grid entries of 4 unsigned bytes
// (iq3xxs_grid, a byte index each: qs[8g .. 8g + 8)) with their signs (ksigns_iq2xs, 7 bits per pair of entries, from
// the sub-block's word at byte 64 + 4g, whose top 4 bits are its scale)
constexpr int GL3_GRID = 144;
struct WideIQ3XXS {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    using Block = block_iq3_xxs;
    static constexpr int LUT = GL3_GRID + 256;
    static uint32_t lut(int i) {
        if (i < 16) return sign_mask4(uint32_t(i));
        if (i < GL3_GRID) return uint32_t(ksigns_iq2xs[i - 16]);
        return iq3xxs_grid[i - GL3_GRID];
    }
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
            const uint32_t e1 = tab[GL3_GRID + ((iw >> sh) & 0xff)], e2 = tab[GL3_GRID + ((iw >> (sh + 8)) & 0xff)];
            grid_group(tab, e1, e2, tab[GL_KSIGNS + ((a1 >> (7 * l)) & 127)], v, l);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.d = (float) b->d * (0.5f + (float) (a1 >> 28)) * 0.5f;
        return r;
    }
    static float dot(const W& r, const X& v) { return r.d * v.d * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};

int multi_groups() {   // NS_DS4_MULTI_GROUPS: the work-groups of a layer's experts in one launch (each fills its grid table once; default 1024)
    static const int v = std::getenv("NS_DS4_MULTI_GROUPS") ? std::max(1, std::atoi(std::getenv("NS_DS4_MULTI_GROUPS"))) : 1024;
    return v;
}

// One row's dot over its 256-blocks: 8 lanes a block (lane g its sub-block g), 4 blocks a sub-group step
template <typename F>
inline float row_dot(const typename F::Block* wr, int blocks_per_row, const XA& xa, int lane, const uint32_t* tab) {
    const int g = lane & 7, sub = lane >> 3;
    float acc = 0.f;
    for (int kbx = sub; kbx < blocks_per_row; kbx += 4) {
        const typename F::W wv = F::load(wr + kbx, g, tab);
        acc += F::dot(wv, F::xl(g, xa, kbx * (QK / Q8K)));
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

// y [entries, n_out]: entry e = the matrix at ptrs[e] against column cols[e]; rows of all entries in one stride
template <typename F>
void multi(sycl::queue& q, const uint64_t* ptrs, const int32_t* cols, int entries, const void* xs, int ncols, float* y, int n_in, int n_out) {
    const auto* x = static_cast<const int8_t*>(xs);
    const std::size_t rows = std::size_t(entries) * n_out;
    const unsigned blocks = (unsigned) std::min<std::size_t>((rows + WARPS - 1) / WARPS, std::size_t(multi_groups()));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                           uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
                           fill_tab<F>(t, it);
                           const int lane = int(it.get_local_id(2)), warp = int(it.get_local_id(1));
                           const int bpr = n_in / QK;
                           auto sg = it.get_sub_group();
                           for (std::size_t r = std::size_t(it.get_group(2)) * WARPS + warp; r < rows; r += std::size_t(it.get_group_range(2)) * WARPS) {
                               const int e = int(r / n_out), row = int(r % n_out);
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

int ns_ds4_multi_supported(int type, int64_t n_in) { return (type == 16 || type == 18) && n_in % QK == 0; }

int ns_ds4_mmvq_multi(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* cols, int64_t entries, const void* xs, int64_t ncols, float* y,
                      int64_t n_in, int64_t n_out) {
    NS_TRY
    if (!ns_ds4_multi_supported(type, n_in)) return ns_fail("ns_ds4_mmvq_multi: ggml type " + std::to_string(type) + " at this width is not its own");
    if (entries < 1) return 0;
    if (type == 16) multi<WideIQ2XXS>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    else multi<WideIQ3XXS>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    return 0;
    NS_CATCH
}

}  // extern "C"
