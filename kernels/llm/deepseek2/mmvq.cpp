// mmvq.cpp: the deepseek2 engine's decode products (ds2.h) - 1..8 columns of Q8 activations against the stored
// blocks of its experts' types: IQ1_S (no shared kernel takes it) and IQ2_XXS (the first layers' down projections),
// one matrix or a layer's experts at once. The scheme is the qwen35 silo's wide kernels (kernels/llm/qwen35/silo/
// mmvq.cpp, copied here, not shared): 8 lanes a 256-block, a lane a sub-block of 32 values, the activations in the
// split Q8 layout (a column's quants contiguous and 16-byte aligned), dp4a against them, the grids in local memory.
//
// IQ1_S: a sub-block is four groups of 8 values, each a grid entry of {-1, 0, 1} (an 11-bit index: a qs byte and 3 bits
// of qh) plus the sub-block's delta (+-1/8), times d (2 x its 3-bit scale + 1). The grid is used as ggml's GPU
// form (iq1s_grid_gpu: the entry + 1 in nibbles, 0..2), so a group is two dp4a of unsigned nibbles; the delta - 1
// comes out against the activations' sum: value = dl x (grid' + delta - 1).
#include "ns.h"
#include "ns_internal.hpp"
#include "ds2.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>

// ggml's grids (iq1s_grid_gpu, iq2xxs_grid, ksigns_iq2xs)
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

// ---- IQ1_S (50-byte blocks, 2-byte aligned): lane g = sub-block g
struct WideIQ1S {
    using XT = X;
    static X xl(int g, const XA& xa, int b0) { return xload(g, xa, b0); }
    struct Block { sycl::half d; uint8_t qs[QK / 8]; uint16_t qh[QK / 32]; };
    static constexpr int LUT = 2048;
    static uint32_t lut(int i) { return iq1s_grid_gpu[i]; }
    struct W { sycl::int4 lo, hi; float d, delta; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        // two 16-byte loads (the 2-byte loads a lane's three fields would take are slow on Xe2): qs [16 (g >> 2) ..]
        // from byte 2, qh from byte 34; d from the first lane's... each lane its own two-byte d (one load)
        const uint8_t* p = reinterpret_cast<const uint8_t*>(b);
        const sycl::int4 q4 = load16_a2(p + 2 + 16 * (g >> 2)), h4 = load16_a2(p + 34);
        const uint32_t qs = word_of(q4, g & 3);   // qs[4g .. 4g + 3]
        const uint32_t qh = (word_of(h4, g >> 1) >> (16 * (g & 1))) & 0xffff;
        const uint16_t* h = reinterpret_cast<const uint16_t*>(b);
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t e = tab[((qs >> (8 * l)) & 0xff) | (((qh >> (3 * l)) & 7) << 8)];
            v[2 * l] = int(e & 0x0f0f0f0fu);
            v[2 * l + 1] = int((e >> 4) & 0x0f0f0f0fu);
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]);
        r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.d = (float) sycl::bit_cast<sycl::half>(h[0]) * (float) (2 * ((qh >> 12) & 7) + 1);
        r.delta = -1.0f + (qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA);
        return r;
    }
    static float dot(const W& r, const X& v) {
        return r.d * v.d * ((float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)) + r.delta * (float) v.s);
    }
};

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

// ---- Q4_K / Q5_K (the qwen35 silo's, copied): lane g = sub-block pair 2 (g >> 1), half g & 1 - 32 quants of two
// sub-blocks; the scales and d from one aligned load of the block's head (small global loads are slow on Xe2)
inline int k4_byte(const uint32_t sc[3], int i) { return int((sc[i >> 2] >> (8 * (i & 3))) & 0xff); }
inline void k4_scale_min(const uint32_t sc[3], int j, float& s, float& m) {
    if (j < 4) { s = (float) (k4_byte(sc, j) & 63); m = (float) (k4_byte(sc, j + 4) & 63); }
    else {
        s = (float) ((k4_byte(sc, j + 4) & 0xF) | ((k4_byte(sc, j - 4) >> 6) << 4));
        m = (float) ((k4_byte(sc, j + 4) >> 4) | ((k4_byte(sc, j) >> 6) << 4));
    }
}
inline void k4_head(const void* b, int sb, float& dl, float& ml, float& dh, float& mh) {
    const sycl::int4 hd = *reinterpret_cast<const sycl::int4*>(b);
    const uint32_t dmw = uint32_t(hd.x()), sc[3] = {uint32_t(hd.y()), uint32_t(hd.z()), uint32_t(hd.w())};
    const float d = (float) sycl::bit_cast<sycl::half>(uint16_t(dmw & 0xffff)), dmin = (float) sycl::bit_cast<sycl::half>(uint16_t(dmw >> 16));
    float s0, m0, s1, m1; k4_scale_min(sc, sb, s0, m0); k4_scale_min(sc, sb + 1, s1, m1);
    dl = d * s0; ml = dmin * m0; dh = d * s1; mh = dmin * m1;
}
struct XK { sycl::int4 ua, uc; float da, dc; int sa, sc; };
inline XK xload_k(int g, const XA& xa, int b0) {
    XK v; const int sb = 2 * (g >> 1), half = g & 1;
    const int a = b0 + sb, c = b0 + sb + 1;
    v.ua = xa.q16(a, half); v.uc = xa.q16(c, half);
    v.da = xa.dd(a); v.dc = xa.dd(c);
    v.sa = xa.sum16(a, half); v.sc = xa.sum16(c, half);
    return v;
}
struct KW { sycl::int4 lo, hi; float dl, dh, ml, mh; };
inline float dot_k(const KW& r, const XK& v) {
    return v.da * (r.dl * (float) dp4a4(r.lo, v.ua, 0) - r.ml * (float) v.sa) + v.dc * (r.dh * (float) dp4a4(r.hi, v.uc, 0) - r.mh * (float) v.sc);
}
struct WideQ4K {
    struct Block { sycl::half2 dm; uint8_t scales[12]; uint8_t qs[128]; };
    using XT = XK;
    using W = KW;
    static constexpr int LUT = 1;
    static uint32_t lut(int) { return 0; }
    static XK xl(int g, const XA& xa, int b0) { return xload_k(g, xa, b0); }
    static W load(const Block* b, int g, const uint32_t*) {
        W r;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];
        r.lo = sycl::int4(q.x() & 0x0f0f0f0f, q.y() & 0x0f0f0f0f, q.z() & 0x0f0f0f0f, q.w() & 0x0f0f0f0f);
        r.hi = sycl::int4((q.x() >> 4) & 0x0f0f0f0f, (q.y() >> 4) & 0x0f0f0f0f, (q.z() >> 4) & 0x0f0f0f0f, (q.w() >> 4) & 0x0f0f0f0f);
        k4_head(b, 2 * (g >> 1), r.dl, r.ml, r.dh, r.mh);
        return r;
    }
    static float dot(const W& r, const XK& v) { return dot_k(r, v); }
};
struct WideQ5K {
    struct Block { sycl::half2 dm; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
    using XT = XK;
    using W = KW;
    static constexpr int LUT = 1;
    static uint32_t lut(int) { return 0; }
    static XK xl(int g, const XA& xa, int b0) { return xload_k(g, xa, b0); }
    static W load(const Block* b, int g, const uint32_t*) {
        W r; const int sb = 2 * (g >> 1), half = g & 1;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];
        const sycl::int4 h = reinterpret_cast<const sycl::int4*>(b->qh)[half];
        auto hb = [](int v, int bit) { return ((v >> bit) & 0x01010101) << 4; };
        r.lo = sycl::int4((q.x() & 0x0f0f0f0f) | hb(h.x(), sb), (q.y() & 0x0f0f0f0f) | hb(h.y(), sb),
                          (q.z() & 0x0f0f0f0f) | hb(h.z(), sb), (q.w() & 0x0f0f0f0f) | hb(h.w(), sb));
        r.hi = sycl::int4(((q.x() >> 4) & 0x0f0f0f0f) | hb(h.x(), sb + 1), ((q.y() >> 4) & 0x0f0f0f0f) | hb(h.y(), sb + 1),
                          ((q.z() >> 4) & 0x0f0f0f0f) | hb(h.z(), sb + 1), ((q.w() >> 4) & 0x0f0f0f0f) | hb(h.w(), sb + 1));
        k4_head(b, sb, r.dl, r.ml, r.dh, r.mh);
        return r;
    }
    static float dot(const W& r, const XK& v) { return dot_k(r, v); }
};

int lut_groups() {   // NS_DS2_MMVQ_GROUPS: the work-groups of a kernel (each fills its grid table once; default 512)
    static const int v = std::getenv("NS_DS2_MMVQ_GROUPS") ? std::max(1, std::atoi(std::getenv("NS_DS2_MMVQ_GROUPS"))) : 512;
    return v;
}
int multi_groups() {   // NS_DS2_MULTI_GROUPS: the same for a layer's experts in one launch (default 1024: B70, 8 IQ1_S experts at once ~225 GB/s, 512 ~180)
    static const int v = std::getenv("NS_DS2_MULTI_GROUPS") ? std::max(1, std::atoi(std::getenv("NS_DS2_MULTI_GROUPS"))) : 1024;
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

// y [ncols, n_out] = W . x: a sub-group a row, every column against it
template <typename F, int NCOLS>
void wide_kernel(const typename F::Block* __restrict__ w, const int8_t* __restrict__ x, float* __restrict__ y, int n_in, int n_out, uint32_t* tab,
                 const sycl::nd_item<3>& item) {
    fill_tab<F>(tab, item);
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int bpr = n_in / QK, g = lane & 7, sub = lane >> 3;
    auto sg = item.get_sub_group();
    XA xa[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) xa[j] = xcol(x, n_in, NCOLS, j);
    for (int row = int(item.get_group(2)) * WARPS + warp; row < n_out; row += int(item.get_group_range(2)) * WARPS) {
        const typename F::Block* wr = w + std::size_t(row) * bpr;
        float acc[NCOLS] = {};
        for (int kbx = sub; kbx < bpr; kbx += 4) {
            const typename F::W wv = F::load(wr + kbx, g, tab);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) acc[j] += F::dot(wv, F::xl(g, xa[j], kbx * (QK / Q8K)));
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[j];
#pragma unroll
            for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (lane == 0) y[std::size_t(j) * n_out + row] = v;
        }
    }
}
template <typename F, int NCOLS>
void launch(sycl::queue& q, const void* weights, const void* xs, float* y, int n_in, int n_out) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const int8_t*>(xs);
    unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
    if (F::LUT > 1) blocks = std::min(blocks, unsigned(lut_groups()));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                           wide_kernel<F, NCOLS>(w, x, y, n_in, n_out, tab.template get_multi_ptr<sycl::access::decorated::no>().get(), it);
                       });
    });
}
template <typename F>
void wide(sycl::queue& q, const void* w, const void* x, float* y, int n_in, int n_out, int ncols) {
    switch (ncols) {
        case 1: launch<F, 1>(q, w, x, y, n_in, n_out); return;
        case 2: launch<F, 2>(q, w, x, y, n_in, n_out); return;
        case 3: launch<F, 3>(q, w, x, y, n_in, n_out); return;
        case 4: launch<F, 4>(q, w, x, y, n_in, n_out); return;
        case 5: launch<F, 5>(q, w, x, y, n_in, n_out); return;
        case 6: launch<F, 6>(q, w, x, y, n_in, n_out); return;
        case 7: launch<F, 7>(q, w, x, y, n_in, n_out); return;
        default: launch<F, 8>(q, w, x, y, n_in, n_out); return;
    }
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

size_t ns_ds2_q8_bytes(int64_t n_in, int64_t ncols) { return (size_t) (ncols * n_in + ncols * (n_in / 32) * 2 + ncols * (n_in / 16) * 2); }

// x [ncols, n_in] float32 -> the split layout: a sub-group of 32 a block (its amax / 127 rounded to fp16 first, as
// Q8_1's scale), each 16 quants' sum
int ns_ds2_quantize_q8(ns_gpu* g, const float* x, void* xs, int64_t n_in, int64_t ncols) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_ds2_quantize_q8: n_in a multiple of 32");
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
        int t = qv;   // the 16-value group's sum (lanes 0-15, 16-31)
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

int ns_ds2_mmvq_supported(int type, int64_t n_in) { return (type == 19 || type == 16 || type == 12 || type == 13) && n_in % QK == 0; }

int ns_ds2_mmvq(ns_gpu* g, int type, const void* w, const void* xs, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (ncols < 1 || ncols > 8) return ns_fail("ns_ds2_mmvq: 1..8 columns");
    if (!ns_ds2_mmvq_supported(type, n_in)) return ns_fail("ns_ds2_mmvq: ggml type " + std::to_string(type) + " at this width is not its own");
    switch (type) {
        case 19: wide<WideIQ1S>(g->q, w, xs, y, (int) n_in, (int) n_out, (int) ncols); break;
        case 16: wide<WideIQ2XXS>(g->q, w, xs, y, (int) n_in, (int) n_out, (int) ncols); break;
        case 12: wide<WideQ4K>(g->q, w, xs, y, (int) n_in, (int) n_out, (int) ncols); break;
        default: wide<WideQ5K>(g->q, w, xs, y, (int) n_in, (int) n_out, (int) ncols); break;
    }
    return 0;
    NS_CATCH
}

int ns_ds2_mmvq_multi(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* cols, int64_t entries, const void* xs, int64_t ncols, float* y,
                      int64_t n_in, int64_t n_out) {
    NS_TRY
    if (!ns_ds2_mmvq_supported(type, n_in)) return ns_fail("ns_ds2_mmvq_multi: ggml type " + std::to_string(type) + " at this width is not its own");
    if (entries < 1) return 0;
    if (type == 12 || type == 13) return ns_fail("ns_ds2_mmvq_multi: the experts' types only (IQ1_S, IQ2_XXS)");
    if (type == 19) multi<WideIQ1S>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    else multi<WideIQ2XXS>(g->q, ptrs, cols, (int) entries, xs, (int) ncols, y, (int) n_in, (int) n_out);
    return 0;
    NS_CATCH
}

}  // extern "C"
