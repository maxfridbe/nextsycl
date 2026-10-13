// attn.cpp: the deepseek4 engine's attention on XMX (ds4.h: ns_ds4_attend_xmx) - the qwen35 silo's flash kernel
// (llm/qwen35/silo/attn.cpp), copied and turned around for this model's MQA: every head of a row reads the same cells
// (its window of raw keys and its compressed blocks - the indexer's pick is a row's, not a head's), so the DPAS rows
// are 8 heads of one row, not 8 rows of one head. A work-group takes one row and G groups of 8 heads and streams the
// row's cells through local memory in chunks of CH:
//
//   1. the chunk's cells into local memory, packed twice: as K (a word = features 2i, 2i + 1 of a cell) and as V (a
//      word = cells 2i, 2i + 1 of a feature) - the key is the value;
//   2. S = Q K^T on joint_matrix (half x half -> float, 8x16x16; sub-group 16): four sub-groups a head group, each over
//      a quarter of the 512 features, their partial scores summed in step 3;
//   3. the online softmax a head (the running max, its rescale c, the sum), P as half;
//   4. O = O c + P V on joint_matrix, each sub-group its quarter of the features (O in registers: 8 tiles of 8x16).
//
// The learned sink joins each head's sum at the end (no value). Q is scaled by scale log2(e) as it is staged (exp2 in
// the softmax). The old kernel (ds4.cpp: attend) read every cell once for each 4 heads, on the vector units.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds4.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <vector>

namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
using half = sycl::half;
#ifndef DS4_ATTN_CH
#define DS4_ATTN_CH 16
#endif
constexpr int D = 512, QD = 128, NQ = D / QD, SG = 16, M = 8, CH = DS4_ATTN_CH, CHP = CH + 8, KS = CH;
constexpr float LOG2E = 1.4426950408889634f;

template <typename T>
inline auto lp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto gp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

// local memory, in 4-byte words: K packed [D/2][CH], V packed [CH/2][D], the partial scores [G][NQ][M][CH] float,
// P [G][M][CHP] half, the head stats [G][M] x 3; Q staged as half [G][M][QD] (a quarter of the features at a time)
// before the first chunk, over the K / V / score words
template <int G>
struct Slm {
    static constexpr int k = D / 2 * KS, v = CH / 2 * D, s = G * NQ * M * CH, p = G * M * CHP / 2, st = 3 * G * M;
    static constexpr int q = G * M * QD / 2;
    static constexpr int words = (k + v + s > q ? k + v + s : q) + p + st;
};

// SPLIT (decode: a few rows): a work-group per (row, head groups, split of kb cells), each head's partial (values
// unnormalized, max, sum) into `part` [T][H][ns][PS] for the merge (attend_merge); a split past its row's cells exits
constexpr int PS = D + 16;   // a partial's row: D values, then the max and the sum (padded: rows 64-byte aligned)
template <int G, bool SPLIT>
void attend_xmx(sycl::queue& queue, const float* qv, const half* raw, int64_t R, const half* comp, int64_t ratio, const int32_t* sel, int64_t K,
                const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window, float scale, int64_t kb, int64_t ns, float* part) {
    constexpr int NSG = NQ * G, WG = NSG * SG;
    using L = Slm<G>;
    const int64_t hgroups = H / (M * G);
    const float qscale = scale * LOG2E;
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(L::words), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * hgroups * (SPLIT ? ns : 1)) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int hg = sgi / NQ, qd = sgi % NQ;   // this sub-group's head group in the work-group, its quarter
            const int64_t gw = it.get_group(0), spl = SPLIT ? gw % ns : 0, gi = SPLIT ? gw / ns : gw;
            const int64_t t = gi / hgroups, h0 = (gi % hgroups) * (M * G);
            const int64_t p = pos0 + t;
            const int64_t n_raw = sycl::min<int64_t>(window, p + 1);
            const int64_t vis = comp ? (p + 1) / ratio : 0;
            const bool pick = sel && vis > K;
            const int64_t n = n_raw + (pick ? K : vis);
            // this work-group's cells (decode: its split)
            const int64_t jbeg = SPLIT ? spl * kb : 0, jend = SPLIT ? sycl::min(n, (spl + 1) * kb) : n;
            if (jbeg >= jend) return;   // the whole work-group, before any barrier
            auto cell = [&](int64_t i) -> const half* {
                if (i < n_raw) return raw + ((p - n_raw + 1 + i) % R) * D;
                const int64_t k = i - n_raw;
                return comp + (pick ? (int64_t) sel[t * K + k] : k) * D;
            };
            uint32_t* base = slm.get_multi_ptr<sycl::access::decorated::no>().get();
            uint32_t* kp = base;
            uint32_t* vp = kp + L::k;
            float* sp = (float*) (vp + L::v);
            half* pp = (half*) (base + (L::k + L::v + L::s > L::q ? L::k + L::v + L::s : L::q));
            float* mrow = (float*) ((uint32_t*) pp + L::p);
            float* lrow = mrow + G * M;
            float* crow = lrow + G * M;
            half* qh = (half*) base;

            // ---- Q, scaled, as half, a quarter of the features at a time; the head stats
            for (int i = tid; i < G * M; i += WG) {
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> qa[QD / 16];
            for (int qq = 0; qq < NQ; ++qq) {
                for (int i = tid; i < G * M * QD; i += WG) {
                    const int hh = i / QD, d = i % QD;   // head h0 + hh
                    qh[i] = (half) (qv[(t * H + h0 + hh) * D + qq * QD + d] * qscale);
                }
                sycl::group_barrier(it.get_group());
                if (qd == qq) {
#pragma unroll
                    for (int k = 0; k < QD / 16; ++k) jm::joint_matrix_load(sg, qa[k], lp(qh + (hg * M) * QD + k * 16), (size_t) QD);
                }
                sycl::group_barrier(it.get_group());
            }
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> o[QD / 16];
#pragma unroll
            for (int tt = 0; tt < QD / 16; ++tt) jm::joint_matrix_fill(sg, o[tt], 0.0f);

            for (int64_t j0 = jbeg; j0 < jend; j0 += CH) {
                // ---- 1. the chunk's cells, packed as K and as V (cells past n are zeros: masked below)
                for (int i = tid; i < CH * (D / 8); i += WG) {   // K: cell c_, features 8c .. 8c + 7 -> words [4c .. 4c + 3][c_]
                    const int nn = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 w(0, 0, 0, 0);
                    if (j0 + nn < jend) w = *reinterpret_cast<const sycl::uint4*>(cell(j0 + nn) + 8 * c);
                    kp[(4 * c + 0) * KS + nn] = w.x();
                    kp[(4 * c + 1) * KS + nn] = w.y();
                    kp[(4 * c + 2) * KS + nn] = w.z();
                    kp[(4 * c + 3) * KS + nn] = w.w();
                }
                for (int i = tid; i < (CH / 2) * (D / 8); i += WG) {   // V: cells 2p, 2p + 1, features 8c .. -> words [p][8c ..]
                    const int pr = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 a(0, 0, 0, 0), b(0, 0, 0, 0);
                    if (j0 + 2 * pr < jend) a = *reinterpret_cast<const sycl::uint4*>(cell(j0 + 2 * pr) + 8 * c);
                    if (j0 + 2 * pr + 1 < jend) b = *reinterpret_cast<const sycl::uint4*>(cell(j0 + 2 * pr + 1) + 8 * c);
                    const uint32_t av[4] = {a.x(), a.y(), a.z(), a.w()}, bv[4] = {b.x(), b.y(), b.z(), b.w()};
                    uint32_t* dst = vp + pr * D + 8 * c;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        dst[2 * e] = (av[e] & 0xffffu) | (bv[e] << 16);
                        dst[2 * e + 1] = (av[e] >> 16) | (bv[e] & 0xffff0000u);
                    }
                }
                sycl::group_barrier(it.get_group());
                // ---- 2. partial scores over this sub-group's quarter
                const half* kph = (const half*) kp;
#pragma unroll
                for (int nt = 0; nt < CH / 16; ++nt) {
                    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> s;
                    jm::joint_matrix_fill(sg, s, 0.0f);
#pragma unroll
                    for (int k = 0; k < QD / 16; ++k) {
                        jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> kbm;
                        jm::joint_matrix_load(sg, kbm, lp(kph + ((qd * QD + k * 16) / 2) * (KS * 2) + nt * 16 * 2), (size_t) (KS * 2));
                        jm::joint_matrix_mad(sg, s, qa[k], kbm, s);
                    }
                    jm::joint_matrix_store(sg, s, lp(sp + ((hg * NQ + qd) * M) * CH + nt * 16), (size_t) CH, jm::layout::row_major);
                }
                sycl::group_barrier(it.get_group());
                // ---- 3. softmax: sub-group (g, q) takes heads 2q, 2q + 1 of group g, 8 lanes a head, CH / 8 cells a lane
                {
                    const int r = qd * 2 + lane / 8, part = lane % 8;
                    float sv[CH / 8], mx = -INFINITY;
#pragma unroll
                    for (int i = 0; i < CH / 8; ++i) {
                        const int nn = part * (CH / 8) + i;
                        float v = 0.0f;
#pragma unroll
                        for (int q4 = 0; q4 < NQ; ++q4) v += sp[((hg * NQ + q4) * M + r) * CH + nn];
                        sv[i] = j0 + nn < jend ? v : -INFINITY;
                        mx = sycl::fmax(mx, sv[i]);
                    }
                    mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 1));
                    mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 2));
                    mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 4));
                    const int si = hg * M + r;
                    const float mo = mrow[si], mn = sycl::fmax(mo, mx);
                    const float c = mn == -INFINITY ? 1.0f : sycl::exp2(mo - mn);
                    float sum = 0.0f;
                    half* prow = pp + (hg * M + r) * CHP;
#pragma unroll
                    for (int i = 0; i < CH / 8; ++i) {
                        const float pv = sv[i] == -INFINITY ? 0.0f : sycl::exp2(sv[i] - mn);
                        sum += pv;
                        prow[part * (CH / 8) + i] = (half) pv;
                    }
                    sum += sycl::permute_group_by_xor(sg, sum, 1);
                    sum += sycl::permute_group_by_xor(sg, sum, 2);
                    sum += sycl::permute_group_by_xor(sg, sum, 4);
                    if (part == 0) {
                        mrow[si] = mn;
                        lrow[si] = lrow[si] * c + sum;
                        crow[si] = c;
                    }
                }
                sycl::group_barrier(it.get_group());
                // ---- 4. O = O c + P V over this sub-group's quarter
                {
                    const float* cr = crow + hg * M;
#pragma unroll
                    for (int tt = 0; tt < QD / 16; ++tt) {
                        int e = 0;
                        jm::joint_matrix_apply(sg, o[tt], [&](float& x) { x *= cr[e]; ++e; });
                    }
                    const half* vph = (const half*) vp;
#pragma unroll
                    for (int ks = 0; ks < CH / 16; ++ks) {
                        jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> pa;
                        jm::joint_matrix_load(sg, pa, lp(pp + (hg * M) * CHP + ks * 16), (size_t) CHP);
#pragma unroll
                        for (int tt = 0; tt < QD / 16; ++tt) {
                            jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> vbm;
                            jm::joint_matrix_load(sg, vbm, lp(vph + (ks * 16 / 2) * (D * 2) + (qd * QD + tt * 16) * 2), (size_t) (D * 2));
                            jm::joint_matrix_mad(sg, o[tt], pa, vbm, o[tt]);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());   // the chunk's words free for the next
            }
            if constexpr (SPLIT) {
                // ---- the heads' partials: O unnormalized, the max and the sum
                float* pb = part + ((t * H + h0 + hg * M) * ns + spl) * PS + qd * QD;
#pragma unroll
                for (int tt = 0; tt < QD / 16; ++tt) jm::joint_matrix_store(sg, o[tt], gp(pb + tt * 16), (size_t) (ns * PS), jm::layout::row_major);
                for (int i = tid; i < G * M; i += WG) {
                    float* st = part + ((t * H + h0 + i) * ns + spl) * PS;
                    st[D] = mrow[i];
                    st[D + 1] = lrow[i];
                }
                return;
            }
            // ---- the sinks join the sums (in log2 units, as the scores), then O / l
            for (int i = tid; i < G * M; i += WG) lrow[i] += sycl::exp2(sinks[h0 + i] * LOG2E - mrow[i]);
            sycl::group_barrier(it.get_group());
            const float* lr = lrow + hg * M;
            float* ob = out + (t * H + h0 + hg * M) * D + qd * QD;
#pragma unroll
            for (int tt = 0; tt < QD / 16; ++tt) {
                int e = 0;
                jm::joint_matrix_apply(sg, o[tt], [&](float& x) { x /= lr[e]; ++e; });
                jm::joint_matrix_store(sg, o[tt], gp(ob + tt * 16), (size_t) D, jm::layout::row_major);
            }
        });
    });
}
// the splits of each (row, head) merged, the sink in the sum: out = sum_k O_k 2^(m_k - m) / (sum_k l_k 2^(m_k - m) + 2^(sink - m))
void attend_merge(sycl::queue& queue, const float* part, const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window,
                  int64_t ratio, bool has_comp, bool has_sel, int64_t K, int64_t kb, int64_t ns) {
    queue.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
        const int64_t rh = it.get_group(0), t = rh / H, hh = rh % H, d = it.get_local_id(0);
        const int64_t p = pos0 + t;
        const int64_t vis = has_comp ? (p + 1) / ratio : 0;
        const int64_t n = sycl::min<int64_t>(window, p + 1) + (has_sel && vis > K ? K : vis);
        const int64_t nsr = (n + kb - 1) / kb;
        const float* pp = part + rh * ns * PS;
        float mx = sinks[hh] * LOG2E;
        for (int64_t k = 0; k < nsr; ++k) mx = sycl::fmax(mx, pp[k * PS + D]);
        float Ls = sycl::exp2(sinks[hh] * LOG2E - mx), A = 0.0f;
        for (int64_t k = 0; k < nsr; ++k) {
            const float e = sycl::exp2(pp[k * PS + D] - mx);
            Ls += pp[k * PS + D + 1] * e;
            A += pp[k * PS + d] * e;
        }
        out[rh * D + d] = A / Ls;
    });
}

// the partials' memory, a GPU's: kept for the process (sized for the most a decode pass asks: 8 rows, 64 heads, 32 splits)
float* part_buf(ns_gpu* g, size_t floats) {
    static std::mutex mu;
    static std::vector<std::pair<ns_gpu*, std::pair<float*, size_t>>> bufs;
    std::lock_guard<std::mutex> lk(mu);
    for (auto& b : bufs)
        if (b.first == g) return b.second.second >= floats ? b.second.first : nullptr;
    const size_t cap = std::max<size_t>(floats, (size_t) 8 * 64 * 32 * PS);
    float* p = sycl::malloc_device<float>(cap, g->q);
    bufs.push_back({g, {p, cap}});
    return p;
}
}  // namespace

extern "C" {

int ns_ds4_attend_xmx_supported(int64_t H, int64_t D_) {
    return D_ == D && H % M == 0;
}

int ns_ds4_attend_xmx(ns_gpu* g, const float* q, const uint16_t* raw, int64_t R, const uint16_t* comp, int64_t ratio, const int32_t* sel, int64_t K,
                      const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window, float scale) {
    NS_TRY
    if (!ns_ds4_attend_xmx_supported(H, D)) return ns_fail("ns_ds4_attend_xmx: heads of 512, a multiple of 8 of them");
    const half* r16 = reinterpret_cast<const half*>(raw);
    const half* c16 = reinterpret_cast<const half*>(comp);
    // many rows: a work-group all 64 heads (the cells staged once for them); few rows: 2 head groups a work-group,
    // more work-groups to fill the card (DS4_ATTN_G overrides)
    // decode rows (at most 8): 8 heads a work-group, the cells in splits (~256 work-groups, 32..: at most 32 splits)
    static const bool split = [] { const char* v = getenv("NS_DS4_ATTN_SPLIT"); return !v || v[0] != '0'; }();
    if (split && T <= 8) {
        const int64_t last = pos0 + T - 1;
        const int64_t vis = comp ? (last + 1) / ratio : 0;
        const int64_t maxn = sycl::min<int64_t>(window, last + 1) + (sel && vis > K ? K : vis);
        const int64_t want = std::max<int64_t>(1, 256 / (T * (H / M)));
        int64_t kb = (maxn + want - 1) / want;
        kb = std::max<int64_t>({32, (maxn + 31) / 32, kb});
        kb = (kb + CH - 1) / CH * CH;
        const int64_t ns = (maxn + kb - 1) / kb;
        float* part = part_buf(g, (size_t) (T * H * ns) * PS);
        if (!part) return ns_fail("ns_ds4_attend_xmx: the partials' buffer is too small");
        attend_xmx<1, true>(g->q, q, r16, R, c16, ratio, sel, K, sinks, out, T, H, pos0, window, scale, kb, ns, part);
        attend_merge(g->q, part, sinks, out, T, H, pos0, window, ratio, comp != nullptr, sel != nullptr, K, kb, ns);
        return 0;
    }
    static const int force = [] { const char* v = getenv("NS_DS4_ATTN_G"); return v ? atoi(v) : 0; }();
    const int G = force ? force : T >= 64 ? 8 : 2;
    if (G == 8 && H % 64 == 0) attend_xmx<8, false>(g->q, q, r16, R, c16, ratio, sel, K, sinks, out, T, H, pos0, window, scale, 0, 1, nullptr);
    else if (G >= 4 && H % 32 == 0) attend_xmx<4, false>(g->q, q, r16, R, c16, ratio, sel, K, sinks, out, T, H, pos0, window, scale, 0, 1, nullptr);
    else if (G >= 2 && H % 16 == 0) attend_xmx<2, false>(g->q, q, r16, R, c16, ratio, sel, K, sinks, out, T, H, pos0, window, scale, 0, 1, nullptr);
    else attend_xmx<1, false>(g->q, q, r16, R, c16, ratio, sel, K, sinks, out, T, H, pos0, window, scale, 0, 1, nullptr);
    return 0;
    NS_CATCH
}

}  // extern "C"
