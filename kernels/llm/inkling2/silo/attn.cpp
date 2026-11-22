// attn.cpp: the inkling2 silo's decode attention (silo.h: ns_ink2s_attn_part) - the base kernel's split form (ink.cpp)
// with shorter splits (64 keys at absolute positions: more work-groups at a short context), without its merge (the
// merge, the sink, the output gate and the quantization are ns_ink2s_merge_gate_xq). A work-group a (row, key head,
// split): a sub-group a query head of the three, its lanes over the dims; the split's keys a tile in local memory.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16, D = 128, DL = D / SG, KB = 64, G = 3;

void range(int64_t T, int64_t p0, int64_t win, int64_t& s0, int64_t& ns) {
    const int64_t lo = win > 0 ? (p0 + 1 - win > 0 ? p0 + 1 - win : 0) : 0, hi = p0 + T;
    s0 = lo / KB;
    ns = (hi - 1) / KB - s0 + 1;
}
}  // namespace

extern "C" {

int64_t ns_ink2s_attn_splits(int64_t T, int64_t p0, int64_t win) {
    int64_t s0, ns;
    range(T, p0, win, s0, ns);
    return ns;
}

int ns_ink2s_attn_part(ns_gpu* g, const float* qp, const void* kcv, const void* vcv, const float* r, const float* relb, int64_t E, int64_t R, float* part,
                       int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale) {
    NS_TRY
    if (Hq != G * Hkv || R > SG || T < 1 || T > 16) return ns_fail("ns_ink2s_attn_part: 3 query heads a key head, a bias of at most 16, 1..16 rows");
    int64_t s0, ns;
    range(T, p0, win, s0, ns);
    const half* kc = (const half*) kcv;
    const half* vc = (const half*) vcv;
    constexpr int WG = G * SG;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> ks(sycl::range<1>((size_t) KB * D), h);
        sycl::local_accessor<half, 1> vs(sycl::range<1>((size_t) KB * D), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * Hkv * ns) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, rr = gi / Hkv, hq = hk * G + sgi;
            const int64_t pos = p0 + rr, first = win > 0 ? sycl::max<int64_t>(0, pos + 1 - win) : 0, end = pos + 1;
            const int64_t lo = sycl::max<int64_t>(first, (s0 + sp) * KB), hi = sycl::min<int64_t>(end, (s0 + sp + 1) * KB);
            float qv[DL];
#pragma unroll
            for (int i = 0; i < DL; ++i) qv[i] = qp[(rr * Hq + hq) * D + lane * DL + i] * scale;
            const float rv = lane < R ? r[(rr * Hq + hq) * R + lane] : 0.0f;
            float m = -INFINITY, s = 0.0f, acc[DL];
#pragma unroll
            for (int i = 0; i < DL; ++i) acc[i] = 0.0f;
            const int64_t nk = hi - lo;
            for (int i = tid; i < nk * D; i += WG) {
                const int64_t slot = (lo + i / D) % cap;
                ks[i] = kc[(hk * cap + slot) * D + i % D];
                vs[i] = vc[(hk * cap + slot) * D + i % D];
            }
            sycl::group_barrier(it.get_group());
            for (int64_t jj = 0; jj < nk; ++jj) {
                const int64_t d = pos - (lo + jj);
                float pl = 0.0f;
#pragma unroll
                for (int i = 0; i < DL; ++i) pl += qv[i] * (float) ks[jj * D + lane * DL + i];
                if (d < E && lane < R) pl += rv * relb[d * R + lane];
                const float lg = sycl::reduce_over_group(sg, pl, sycl::plus<float>());
                const float mn = sycl::fmax(m, lg);
                const float c = sycl::exp(m - mn), p = sycl::exp(lg - mn);
                s = s * c + p;
#pragma unroll
                for (int i = 0; i < DL; ++i) acc[i] = acc[i] * c + p * (float) vs[jj * D + lane * DL + i];
                m = mn;
            }
            float* pp = part + ((rr * Hq + hq) * ns + sp) * (D + 2);
#pragma unroll
            for (int i = 0; i < DL; ++i) pp[lane * DL + i] = acc[i];
            if (lane == 0) {
                pp[D] = m;
                pp[D + 1] = s;
            }
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
