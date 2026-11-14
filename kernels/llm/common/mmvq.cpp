// mmvq.cpp: the llm engines' shared decode-width kernels (glm.h's C ABI: ns_quantize_q8_1, ns_mmvq, ns_dequant_f16,
// ns_moe_grouped) - nextsycl's own, over the block readers of quants.hpp:
//
// - activations quantized once to Q8_1 (32 values: int8 quants, an fp16 scale d = amax / 127 and s = d x their sum)
// - a product: a sub-group (32 lanes) a row; for the 256-value types 8 lanes a block (lane g its sub-block g of 32
//   values) and 4 blocks a step, for the 32-value types a block a lane; each lane's weights read once and dotted with
//   every column (dp4a), the lanes' sums added at the end. A row's sums are in an order fixed by the kernel alone, so
//   a column of an n-column call equals a one-column call bit for bit (verify passes = one-token passes).
// - the routed experts of a layer (ns_moe_grouped): the same rows, a group's entries (its tokens) taken 8 at a time
//   against each block of weights read; the SwiGLU between, quantized to Q8_1 in the scratch.
// - the types with a codebook keep it in local memory (a work-group fills it once; those kernels stride over rows with
//   a bounded number of work-groups).
#include "ns.h"
#include "ns_internal.hpp"
#include "ns_q8_1.hpp"
#include "quants.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

namespace {
using namespace nsq;
constexpr int WARP = 32, WARPS = 4;   // 4 sub-groups (rows) a work-group
constexpr int MAXE = 8;               // entries (columns) a pass of the grouped kernels

int lut_groups() {   // NS_MMVQ_GROUPS: the work-groups of a kernel with a local table (default 512)
    static const int v = std::getenv("NS_MMVQ_GROUPS") ? std::max(1, std::atoi(std::getenv("NS_MMVQ_GROUPS"))) : 512;
    return v;
}

// R rows' dots with NC columns (xs[j] column j's Q8_1 blocks) by L lanes (8, 16 or 32; 4 too for a 32-value type) -
// a sub-group takes 32 / L row sets; acc[r x NC + j] the lane's part. Each lane reads a column's activations once for
// the R rows. 256-value types: lane l takes sub-block l % 8 of blocks l / 8, + L / 8, ...; 32-value types: blocks
// l, l + L, ... A row's sums are in the same order whatever R and NC are.
template <typename F, int NC, int L, int R>
inline void rows_wide(const typename F::Block* const* wr, int nb, const block_q8_1* const* xs, int l, const uint32_t* tab, float* acc) {
    const int g = l & 7;
    for (int kb = l >> 3; kb < nb; kb += L / 8) {
        W w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = F::load(wr[r] + kb, g, tab);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const X x = xload(xs[j] + 8 * kb + g);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r * NC + j] += dot<F::OFF>(w[r], x);
        }
    }
}
template <typename F, int NC, int L, int R>
inline void rows_small(const typename F::Block* const* wr, int nb, const block_q8_1* const* xs, int l, const uint32_t* tab, float* acc) {
    for (int kb = l; kb < nb; kb += L) {
        W w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = F::load32(wr[r] + kb, tab);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const X x = xload(xs[j] + kb);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r * NC + j] += dot<NONE>(w[r], x);
        }
    }
}
template <typename F> constexpr bool is_small() { return sizeof(typename F::Block) <= 34; }
template <typename F> constexpr int per_block() { return is_small<F>() ? 32 : QK; }
template <typename F> std::size_t row_bytes_t(int64_t n) { return std::size_t(n / per_block<F>()) * sizeof(typename F::Block); }

template <typename F, int NC, int L, int R>
inline void rows_dot(const typename F::Block* const* wr, int n_in, const block_q8_1* const* xs, int l, const uint32_t* tab, float* acc) {
    if constexpr (is_small<F>()) rows_small<F, NC, L, R>(wr, n_in / 32, xs, l, tab, acc);
    else rows_wide<F, NC, L, R>(wr, n_in / QK, xs, l, tab, acc);
}
template <typename F, int NC, int L>
inline void row_dot(const typename F::Block* wr, int n_in, const block_q8_1* const* xs, int l, const uint32_t* tab, float* acc) {
    rows_dot<F, NC, L, 1>(&wr, n_in, xs, l, tab, acc);
}
// the L lanes of each row added (every lane of the sub-group takes part)
template <int NC, int L>
inline void reduce(sycl::sub_group sg, float* acc) {
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int o = L / 2; o > 0; o >>= 1) acc[j] += sycl::permute_group_by_xor(sg, acc[j], o);
}
// lanes a row for rows of n_in values: a lane at least two blocks (or sub-blocks) to read where the row allows
template <typename F>
int lanes_for(int n_in) {
    const int tasks = is_small<F>() ? n_in / 32 : n_in / QK * 8;
    const int lo = is_small<F>() ? 4 : 8;
    int l = 32;
    while (l > lo && tasks < 2 * l) l /= 2;
    return l;
}

template <typename F>
inline void fill_tab(uint32_t* tab, const sycl::nd_item<2>& it) {
    if constexpr (F::LUT > 0) {
        for (int i = int(it.get_local_linear_id()); i < F::LUT; i += WARPS * WARP) tab[i] = F::lut(i);
        sycl::group_barrier(it.get_group());
    }
}

// ---- y [ncols, n_out] = W [n_out, n_in] . x: a sub-group 32 / L rows, 4 sub-groups a work-group; a type with a local
// table strides its (bounded) work-groups over the rows
template <typename F, int NC, int L>
void mmvq_launch(sycl::queue& q, const void* w, const void* x, float* y, int n_in, int n_out) {
    constexpr int R = WARP / L;
    const auto* W0 = static_cast<const typename F::Block*>(w);
    const auto* X0 = static_cast<const block_q8_1*>(x);
    unsigned groups = unsigned((std::size_t(n_out) + WARPS * R - 1) / (WARPS * R));
    if (F::LUT > 0) groups = std::min(groups, unsigned(lut_groups()));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(groups * WARPS, WARP), sycl::range<2>(WARPS, WARP)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
            fill_tab<F>(t, it);
            const int lane = int(it.get_local_id(1)), l = lane % L;
            const block_q8_1* xs[NC];
#pragma unroll
            for (int j = 0; j < NC; ++j) xs[j] = X0 + std::size_t(j) * (n_in / 32);
            const std::size_t bpr = std::size_t(n_in / per_block<F>());
            for (int r0 = (int(it.get_group(0)) * WARPS + int(it.get_local_id(0))) * R; r0 < n_out; r0 += int(it.get_group_range(0)) * WARPS * R) {
                const int row = r0 + lane / L;
                float acc[NC] = {};
                row_dot<F, NC, L>(W0 + std::size_t(sycl::min(row, n_out - 1)) * bpr, n_in, xs, l, t, acc);
                reduce<NC, L>(it.get_sub_group(), acc);
                if (l == 0 && row < n_out)
#pragma unroll
                    for (int j = 0; j < NC; ++j) y[std::size_t(j) * n_out + row] = acc[j];
            }
        });
    });
}
template <typename F, int L>
void mmvq_cols(sycl::queue& q, const void* w, const void* x, float* y, int n_in, int n_out, int ncols) {
    switch (ncols) {
        case 1: mmvq_launch<F, 1, L>(q, w, x, y, n_in, n_out); return;
        case 2: mmvq_launch<F, 2, L>(q, w, x, y, n_in, n_out); return;
        case 3: mmvq_launch<F, 3, L>(q, w, x, y, n_in, n_out); return;
        case 4: mmvq_launch<F, 4, L>(q, w, x, y, n_in, n_out); return;
        case 5: mmvq_launch<F, 5, L>(q, w, x, y, n_in, n_out); return;
        case 6: mmvq_launch<F, 6, L>(q, w, x, y, n_in, n_out); return;
        case 7: mmvq_launch<F, 7, L>(q, w, x, y, n_in, n_out); return;
        default: mmvq_launch<F, 8, L>(q, w, x, y, n_in, n_out); return;
    }
}
template <typename F>
void mmvq_any(sycl::queue& q, const void* w, const void* x, float* y, int n_in, int n_out, int ncols) {
    switch (lanes_for<F>(n_in)) {
        case 32: mmvq_cols<F, 32>(q, w, x, y, n_in, n_out, ncols); return;
        case 16: mmvq_cols<F, 16>(q, w, x, y, n_in, n_out, ncols); return;
        case 8: mmvq_cols<F, 8>(q, w, x, y, n_in, n_out, ncols); return;
        default:
            if constexpr (is_small<F>()) mmvq_cols<F, 4>(q, w, x, y, n_in, n_out, ncols);
            else mmvq_cols<F, 8>(q, w, x, y, n_in, n_out, ncols);
            return;
    }
}

// the types and their readers: a call with the reader as a template argument, or false for a type with none
template <typename Fn>
bool with_type(int type, Fn&& fn) {
    switch (type) {
        case 8: fn(TQ80{}); return true;
        case 10: fn(TQ2K{}); return true;
        case 12: fn(TQ4K{}); return true;
        case 13: fn(TQ5K{}); return true;
        case 14: fn(TQ6K{}); return true;
        case 16: fn(TIQ2XXS{}); return true;
        case 17: fn(TIQ2XS{}); return true;
        case 18: fn(TIQ3XXS{}); return true;
        case 20: fn(TIQ4NL{}); return true;
        case 21: fn(TIQ3S{}); return true;
        case 22: fn(TIQ2S{}); return true;
        case 23: fn(TIQ4XS{}); return true;
        case 29: fn(TIQ1M{}); return true;
        default: return false;
    }
}
int block_values(int type) {
    int n = 0;
    with_type(type, [&](auto f) { n = per_block<decltype(f)>(); });
    return n;
}
std::size_t row_bytes(int type, int64_t n) {
    std::size_t b = 0;
    with_type(type, [&](auto f) { using F = decltype(f); b = std::size_t(n / per_block<F>()) * sizeof(typename F::Block); });
    return b;
}

// ---- n values of a stored matrix into half: a work-item a sub-block of 32 (a 256-block's lane g, or a 32-block)
template <typename F>
void dequant_launch(sycl::queue& q, const void* src, int64_t n, sycl::half* dst) {
    const auto* B = static_cast<const typename F::Block*>(src);
    const std::size_t subs = std::size_t(n / 32);
    const unsigned groups = unsigned(std::min<std::size_t>((subs + 127) / 128, F::LUT > 0 ? std::size_t(4096) : std::size_t(1) << 30));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(groups * WARPS, WARP), sycl::range<2>(WARPS, WARP)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
            fill_tab<F>(t, it);
            for (std::size_t s = it.get_global_linear_id(); s < subs; s += it.get_global_range(0) * WARP) {
                W w;
                if constexpr (is_small<F>()) w = F::load32(B + s, t);
                else w = F::load(B + s / 8, int(s % 8), t);
                sycl::vec<sycl::half, 8> v[4];
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    if constexpr (is_small<F>()) v[i >> 3][i & 7] = (sycl::half) value<NONE>(w, i);
                    else v[i >> 3][i & 7] = (sycl::half) value<F::OFF>(w, i);
                }
                auto* o = reinterpret_cast<sycl::vec<sycl::half, 8>*>(dst + s * 32);
#pragma unroll
                for (int k = 0; k < 4; ++k) o[k] = v[k];
            }
        });
    });
}

// ---- the grouped experts. An expert's blob: gate [n_ff rows of n_embd] | up [the same] | down [n_embd rows of n_ff].
// The scratch (as Strata's native_expert_grouped left it, which the glm5next silo's down reads): gate, up, h [entries x
// n_ff] float32, each 256-byte aligned, then h as Q8_1 [entries][n_ff / 32]
inline std::size_t al256(std::size_t b) { return (b + 255) & ~std::size_t(255); }

// R rows' dots with entries [e0, e1) of their group: 8 at a time, then 4, 2, 1 (each entry's sums the same whatever its
// chunk: rows do not depend on how many entries a launch has). xf(e) an entry's activation blocks, sf(e, r, v) its result
template <typename F, int L, int R, int N, typename XF, typename SF>
inline void entry_chunk(const typename F::Block* const* wr, int n_in, int e0, XF xf, SF sf, int l, const uint32_t* t, sycl::sub_group sg) {
    const block_q8_1* xs[N];
#pragma unroll
    for (int k = 0; k < N; ++k) xs[k] = xf(e0 + k);
    float acc[R * N] = {};
    rows_dot<F, N, L, R>(wr, n_in, xs, l, t, acc);
    reduce<R * N, L>(sg, acc);
    if (l == 0)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int k = 0; k < N; ++k) sf(e0 + k, r, acc[r * N + k]);
}
template <typename F, int L, int R, typename XF, typename SF>
inline void entries(const typename F::Block* const* wr, int n_in, int e0, int e1, XF xf, SF sf, int l, const uint32_t* t, sycl::sub_group sg) {
    int e = e0;
    for (; e + MAXE <= e1; e += MAXE) entry_chunk<F, L, R, MAXE>(wr, n_in, e, xf, sf, l, t, sg);
    if (e + 4 <= e1) { entry_chunk<F, L, R, 4>(wr, n_in, e, xf, sf, l, t, sg); e += 4; }
    if (e + 2 <= e1) { entry_chunk<F, L, R, 2>(wr, n_in, e, xf, sf, l, t, sg); e += 2; }
    if (e < e1) entry_chunk<F, L, R, 1>(wr, n_in, e, xf, sf, l, t, sg);
}
// work-groups for `sets` row sets (4 x 32 / L a work-group), at most 4 x NS_MMVQ_GROUPS of them for a type with a local
// table (each fills it once, then strides over the rows)
template <typename F, int L>
unsigned expert_groups(std::size_t sets) {
    constexpr int S = WARP / L;
    const std::size_t g = (sets + WARPS * S - 1) / (WARPS * S);
    return unsigned(F::LUT > 0 ? std::min<std::size_t>(g, std::size_t(lut_groups()) * 4) : g);
}

// gate | up: row r of an expert's gate and row r of its up as one set (they share the activations), the sets of every
// group in turn; 32 / L sets a sub-group (all of one group: n_ff a multiple of them)
template <typename F, int L>
void gu_launch(sycl::queue& q, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_tok,
               int64_t groups, const void* xq, int n_embd, int n_ff, float* gate, float* up) {
    constexpr int S = WARP / L;
    const auto* X0 = static_cast<const block_q8_1*>(xq);
    const std::size_t gu_row = row_bytes_t<F>(n_embd), up_off = std::size_t(n_ff) * gu_row;
    const unsigned wgs = expert_groups<F, L>(std::size_t(groups) * n_ff);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(std::size_t(wgs) * WARPS, WARP), sycl::range<2>(WARPS, WARP)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
            fill_tab<F>(t, it);
            const int lane = int(it.get_local_id(1)), l = lane % L;
            const std::size_t ns = std::size_t(*n_groups) * n_ff;
            for (std::size_t i0 = (std::size_t(it.get_group(0)) * WARPS + it.get_local_id(0)) * S; i0 < ns; i0 += std::size_t(it.get_group_range(0)) * WARPS * S) {
                const std::size_t i = i0 + lane / L;
                const int grp = int(i / n_ff), r = int(i % n_ff);
                const uint8_t* blob = reinterpret_cast<const uint8_t*>(grp_ptr[grp]) + std::size_t(r) * gu_row;
                const typename F::Block* wr[2] = {reinterpret_cast<const typename F::Block*>(blob), reinterpret_cast<const typename F::Block*>(blob + up_off)};
                entries<F, L, 2>(wr, n_embd, grp_start[grp], grp_start[grp + 1],
                                 [&](int e) { return X0 + std::size_t(ent_tok[e]) * (n_embd / 32); },
                                 [&](int e, int k, float v) { (k ? up : gate)[std::size_t(e) * n_ff + r] = v; }, l, t, it.get_sub_group());
            }
        });
    });
}
// down: out [ent_dst[e], n_embd] = down . h_e, two consecutive rows a set
template <typename F, int L>
void down_launch(sycl::queue& q, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                 int64_t groups, const block_q8_1* hq, int n_embd, int n_ff, std::size_t down_off, float* out) {
    constexpr int S = WARP / L;
    const std::size_t d_row = row_bytes_t<F>(n_ff);
    const int sets = n_embd / 2;
    const unsigned wgs = expert_groups<F, L>(std::size_t(groups) * sets);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(std::size_t(wgs) * WARPS, WARP), sycl::range<2>(WARPS, WARP)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            uint32_t* t = tab.template get_multi_ptr<sycl::access::decorated::no>().get();
            fill_tab<F>(t, it);
            const int lane = int(it.get_local_id(1)), l = lane % L;
            const std::size_t ns = std::size_t(*n_groups) * sets;
            for (std::size_t i0 = (std::size_t(it.get_group(0)) * WARPS + it.get_local_id(0)) * S; i0 < ns; i0 += std::size_t(it.get_group_range(0)) * WARPS * S) {
                const std::size_t i = i0 + lane / L;
                const int grp = int(i / sets), row = 2 * int(i % sets);
                const uint8_t* base = reinterpret_cast<const uint8_t*>(grp_ptr[grp]) + down_off + std::size_t(row) * d_row;
                const typename F::Block* wr[2] = {reinterpret_cast<const typename F::Block*>(base), reinterpret_cast<const typename F::Block*>(base + d_row)};
                entries<F, L, 2>(wr, n_ff, grp_start[grp], grp_start[grp + 1],
                                 [&](int e) { return hq + std::size_t(e) * (n_ff / 32); },
                                 [&](int e, int k, float v) { out[std::size_t(ent_dst[e]) * n_embd + row + k] = v; }, l, t, it.get_sub_group());
            }
        });
    });
}
// the lanes a row of the grouped kernels: the caller's (Strata's meaning: 4 / 8 / 16 / 32, 0 = the default) where it
// fits the type and the rows - 8 at the least for a 256-value type, rows of a group a multiple of 32 / L
template <typename F>
int expert_lanes(int lanes, int n_in, int rows) {
    int l = lanes > 0 ? lanes : lanes_for<F>(n_in);
    l = std::max(l, is_small<F>() ? 4 : 8);
    l = std::min(l, 32);
    while (l < 32 && rows % (WARP / l) != 0) l *= 2;
    return l;
}
template <typename F>
void gu_any(int lanes, sycl::queue& q, const uint64_t* p, const int32_t* s, const int32_t* n, const int32_t* tok, int64_t groups, const void* xq,
            int n_embd, int n_ff, float* gate, float* up) {
    switch (expert_lanes<F>(lanes, n_embd, n_ff)) {
        case 4: if constexpr (is_small<F>()) { gu_launch<F, 4>(q, p, s, n, tok, groups, xq, n_embd, n_ff, gate, up); return; } [[fallthrough]];
        case 8: gu_launch<F, 8>(q, p, s, n, tok, groups, xq, n_embd, n_ff, gate, up); return;
        case 16: gu_launch<F, 16>(q, p, s, n, tok, groups, xq, n_embd, n_ff, gate, up); return;
        default: gu_launch<F, 32>(q, p, s, n, tok, groups, xq, n_embd, n_ff, gate, up); return;
    }
}
template <typename F>
void down_any(int lanes, sycl::queue& q, const uint64_t* p, const int32_t* s, const int32_t* n, const int32_t* dst, int64_t groups, const block_q8_1* hq,
              int n_embd, int n_ff, std::size_t down_off, float* out) {
    switch (expert_lanes<F>(lanes, n_ff, n_embd / 2)) {
        case 4: if constexpr (is_small<F>()) { down_launch<F, 4>(q, p, s, n, dst, groups, hq, n_embd, n_ff, down_off, out); return; } [[fallthrough]];
        case 8: down_launch<F, 8>(q, p, s, n, dst, groups, hq, n_embd, n_ff, down_off, out); return;
        case 16: down_launch<F, 16>(q, p, s, n, dst, groups, hq, n_embd, n_ff, down_off, out); return;
        default: down_launch<F, 32>(q, p, s, n, dst, groups, hq, n_embd, n_ff, down_off, out); return;
    }
}
// h = silu(gate) x up (GLM's clamp when limit > 0: silu(min(g, limit)) x clamp(u, -limit, limit)) as float32 and Q8_1:
// a sub-group a block of 32
void swiglu_q8(sycl::queue& q, const float* gate, const float* up, float* hf, block_q8_1* hq, int64_t entries, int n_ff, float limit) {
    const std::size_t n = std::size_t(entries) * n_ff;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((n + 255) / 256 * 256), sycl::range<1>(256)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const std::size_t i = it.get_global_id(0);
        float v = 0.f;
        if (i < n) {
            float g = gate[i], u = up[i];
            if (limit > 0.f) { g = sycl::fmin(g, limit); u = sycl::fmin(sycl::fmax(u, -limit), limit); }
            v = g / (1.0f + sycl::exp(-g)) * u;
            hf[i] = v;
        }
        float amax = sycl::fabs(v), sum = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
            sum += sycl::permute_group_by_xor(sg, sum, o);
        }
        if (i >= n) return;
        const float d = ns::q8_1_finite(amax / 127.0f);
        block_q8_1* b = hq + i / 32;
        b->qs[i % 32] = ns::q8_1_quant(v, d, amax);
        if (i % 32 == 0) b->ds = ns::q8_1_ds(d, sum);
    });
}

// the grouped kernels' gate | up and down types: the pairs the engines were routed by (Strata's lists, of the types read
// here) - a pair taken here keeps an engine off its silo's grouped kernels, so the lists stay as they were
bool moe_gu_type(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 23 || t == 29 || t == 12 || t == 13 || t == 8 || t == 10; }
bool moe_d_type(int t) { return t == 20 || t == 23 || t == 8 || t == 10 || t == 12 || t == 13; }

}  // namespace

// NS_KERNEL_TRACE=FILE: each distinct call of these shared kernels (function, types, shape) appended to FILE once -
// which types and shapes the engines still send here
static void trace(const char* fn, int a, int b, int64_t c, int64_t d, int64_t e) {
    static const char* path = getenv("NS_KERNEL_TRACE");
    if (!path || !*path) return;
    static std::mutex m;
    static std::set<std::string> seen;
    char line[160];
    snprintf(line, sizeof line, "%s %d %d %lld %lld %lld", fn, a, b, (long long) c, (long long) d, (long long) e);
    std::lock_guard<std::mutex> l(m);
    if (!seen.insert(line).second) return;
    if (FILE* f = fopen(path, "a")) { fprintf(f, "%s\n", line); fclose(f); }
}

extern "C" {

size_t ns_q8_1_bytes(int64_t n_in, int64_t ncols) { return (size_t) (n_in / 32) * sizeof(block_q8_1) * (size_t) ncols; }

int ns_mmvq_supported(int type) { return block_values(type) > 0; }

// x [ncols, n_in] float32 -> Q8_1 blocks (column j's at j x n_in / 32): a sub-group a block
int ns_quantize_q8_1(ns_gpu* g, const float* x, void* q8_1, int64_t n_in, int64_t ncols) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_quantize_q8_1: n_in a multiple of 32");
    trace("q8_1", 0, 0, n_in, ncols, 0);
    const std::size_t n = std::size_t(n_in) * std::size_t(ncols);
    auto* q = static_cast<block_q8_1*>(q8_1);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((n + 255) / 256 * 256), sycl::range<1>(256)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const std::size_t i = it.get_global_id(0);
        const float v = i < n ? x[i] : 0.0f;
        float amax = sycl::fabs(v), sum = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
            sum += sycl::permute_group_by_xor(sg, sum, o);
        }
        if (i >= n) return;
        const float d = ns::q8_1_finite(amax / 127.0f);
        block_q8_1* b = q + i / 32;
        b->qs[i % 32] = ns::q8_1_quant(v, d, amax);
        if (i % 32 == 0) b->ds = ns::q8_1_ds(d, sum);
    });
    return 0;
    NS_CATCH
}

// y [ncols, n_out] = W [n_out, n_in] (stored blocks) . x (Q8_1, ncols columns); ncols 1..8
int ns_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (ncols < 1 || ncols > 8) return ns_fail("ns_mmvq: 1..8 columns");
    trace("mmvq", type, 0, n_in, n_out, ncols);
    const int bv = block_values(type);
    if (bv == 0) return ns_fail("ns_mmvq: ggml type " + std::to_string(type) + " has no decode kernel");
    if (n_in % bv != 0) return ns_fail("ns_mmvq: n_in " + std::to_string(n_in) + " not a multiple of the type's block");
    with_type(type, [&](auto f) { mmvq_any<decltype(f)>(g->q, w, x_q8_1, y, (int) n_in, (int) n_out, (int) ncols); });
    return 0;
    NS_CATCH
}

// n values of a stored matrix (n a multiple of 256) expanded to fp16, for the prompt path's half GEMMs
int ns_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst) {
    NS_TRY
    if (block_values(type) == 0 || n % 256 != 0) return ns_fail("ns_dequant_f16: ggml type " + std::to_string(type));
    trace("dequant_f16", type, 0, n, 0, 0);
    with_type(type, [&](auto f) { dequant_launch<decltype(f)>(g->q, src, n, reinterpret_cast<sycl::half*>(dst)); });
    return 0;
    NS_CATCH
}

int ns_moe_grouped_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    return moe_gu_type(gu_type) && moe_d_type(d_type) && n_embd % block_values(gu_type) == 0 && n_ff % block_values(d_type) == 0 && n_ff % 32 == 0 ? 1 : 0;
}

size_t ns_moe_scratch_bytes(int64_t entries, int64_t n_ff) {
    const size_t f = (size_t) entries * (size_t) n_ff * sizeof(float);
    return 3 * al256(f) + al256((size_t) entries * (size_t) (n_ff / 32) * sizeof(block_q8_1));
}

// a layer's routed experts: gate | up for every entry, the SwiGLU as Q8_1, down. grp_ptr [groups] the experts' blobs,
// grp_start [groups + 1] their entries' ranges, n_groups (one int; groups is its upper bound), ent_tok [entries] each
// entry's row of x_q8_1, ent_dst [entries] its row of out [., n_embd]; all device memory. phase 0 both halves, 1 gate |
// up and the activation (into the scratch), 2 down from the scratch. `lanes` (Strata's lanes a row, which set the order
// of its sums) is not read: here the lanes a row follow from the row's length alone, so a verify pass's rows equal
// one-token passes' whatever count the caller gives (B70, GLM-5.3's IQ2_XXS / Q2_K experts: 8 lanes 252 us a token's
// layer, 32 lanes 196)
int ns_moe_grouped(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t groups, int64_t entries, const void* x_q8_1,
                   void* scratch, float* out, float limit, int lanes, int phase) {
    NS_TRY
    (void) lanes;
    if (!ns_moe_grouped_supported(gu_type, d_type, n_embd, n_ff))
        return ns_fail("ns_moe_grouped: types " + std::to_string(gu_type) + "/" + std::to_string(d_type) + " not supported");
    if (groups <= 0 || entries <= 0) return 0;
    trace("moe", gu_type, d_type, n_embd, n_ff, phase);
    const std::size_t f = al256(std::size_t(entries) * std::size_t(n_ff) * sizeof(float));
    auto* base = static_cast<uint8_t*>(scratch);
    float* gate = reinterpret_cast<float*>(base);
    float* up = reinterpret_cast<float*>(base + f);
    float* hf = reinterpret_cast<float*>(base + 2 * f);
    auto* hq = reinterpret_cast<block_q8_1*>(base + 3 * f);
    if (phase != 2) {
        with_type(gu_type, [&](auto t) { gu_any<decltype(t)>(0, g->q, grp_ptr, grp_start, n_groups, ent_tok, groups, x_q8_1, (int) n_embd, (int) n_ff, gate, up); });
        swiglu_q8(g->q, gate, up, hf, hq, entries, (int) n_ff, limit);
    }
    if (phase != 1) {
        const std::size_t down_off = 2 * std::size_t(n_ff) * row_bytes(gu_type, n_embd);
        with_type(d_type, [&](auto t) { down_any<decltype(t)>(0, g->q, grp_ptr, grp_start, n_groups, ent_dst, groups, hq, (int) n_embd, (int) n_ff, down_off, out); });
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
