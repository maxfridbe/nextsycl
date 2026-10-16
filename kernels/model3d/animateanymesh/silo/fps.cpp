// fps.cpp: farthest point sampling of the vertices' embeddings (silo.h). The engine's own kernel (aam.cpp) is one
// work-group walking every point for each of the K picks: 1.2 s for a 20K-vertex mesh's 2,500 tokens. Here a pick is
// one launch over the whole GPU - a work-item a point, the embeddings read transposed so a sub-group's loads are
// one row of them - and the last work-group to finish reduces the groups' best into the pick the next launch reads.
// Every distance is summed in the same order, one rounding a step, as the engine's: the same picks.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <climits>
#include <limits>

extern "C" {

int ns_model3d_aam_silo_fps_supported(int64_t N, int64_t D, int64_t K) {
    return N > 0 && N < INT_MAX && D > 0 && D <= 1024 && K > 0 && K <= N ? 1 : 0;
}

int ns_model3d_aam_silo_fps(ns_gpu* g, const float* x, int64_t N, int64_t D, int64_t K, float* dist, int32_t* idx) {
    NS_TRY
    if (!ns_model3d_aam_silo_fps_supported(N, D, K)) return ns_fail("ns_model3d_aam_silo_fps: 0 < K <= N, 0 < D <= 1024");
    constexpr int WG = 256;
    const int64_t nwg = (N + WG - 1) / WG;
    auto& q = g->q;
    float* xt = sycl::malloc_device<float>((size_t) (N * D), q);
    float* pbest = sycl::malloc_device<float>((size_t) nwg, q);
    int32_t* pidx = sycl::malloc_device<int32_t>((size_t) nwg + 2, q);   // the groups' picks, then the pick, the count
    if (!xt || !pbest || !pidx) {
        sycl::free(xt, q);
        sycl::free(pbest, q);
        sycl::free(pidx, q);
        return ns_fail("ns_model3d_aam_silo_fps: no memory for its scratch");
    }
    int32_t* cur = pidx + nwg;
    int32_t* count = pidx + nwg + 1;
    q.parallel_for(sycl::range<2>((size_t) D, (size_t) N), [=](sycl::id<2> i) { xt[i[0] * N + i[1]] = x[i[1] * D + i[0]]; });
    q.parallel_for(sycl::range<1>((size_t) N), [=](sycl::id<1> p) {
        dist[p] = std::numeric_limits<float>::max();
        if (p[0] == 0) {
            idx[0] = 0;
            *cur = 0;
            *count = 0;
        }
    });
    for (int64_t kk = 1; kk < K; ++kk) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nwg * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
#pragma clang fp contract(off) reassociate(off)
            const int64_t p = it.get_global_id(0);
            const float* sel = x + (int64_t) *cur * D;
            float best = -1.0f;
            int bi = INT_MAX;
            if (p < N) {
                float s = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    const float diff = sel[d] - xt[d * N + p];
                    const float sq = diff * diff;
                    s = s + sq;
                }
                const float dd = sycl::fmin(s, dist[p]);
                dist[p] = dd;
                best = dd;
                bi = (int) p;
            }
            const auto grp = it.get_group();
            const float m = sycl::reduce_over_group(grp, best, sycl::maximum<float>());
            const int c = sycl::reduce_over_group(grp, best == m ? bi : INT_MAX, sycl::minimum<int>());
            const int64_t w = it.get_group(0);
            int last = 0;
            if (it.get_local_id(0) == 0) {
                pbest[w] = m;
                pidx[w] = c;
                sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space> n(*count);
                last = n.fetch_add(1) == (int32_t) nwg - 1;
            }
            last = sycl::group_broadcast(grp, last, 0);
            if (!last) return;
            // the last group: every group's best seen, the pick made
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
            float bm = -1.0f;
            int bc = INT_MAX;
            for (int64_t i = it.get_local_id(0); i < nwg; i += WG) {
                const float v = pbest[i];
                const int ci = pidx[i];
                if (v > bm || (v == bm && ci < bc)) {
                    bm = v;
                    bc = ci;
                }
            }
            const float gm = sycl::reduce_over_group(grp, bm, sycl::maximum<float>());
            const int gc = sycl::reduce_over_group(grp, bm == gm ? bc : INT_MAX, sycl::minimum<int>());
            if (it.get_local_id(0) == 0) {
                idx[kk] = gc;
                *cur = gc;
                *count = 0;
            }
        });
    }
    q.wait();
    sycl::free(xt, q);
    sycl::free(pbest, q);
    sycl::free(pidx, q);
    return 0;
    NS_CATCH
}

}  // extern "C"
