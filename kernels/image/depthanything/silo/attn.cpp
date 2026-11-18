// attn.cpp: the Depth Anything V2 silo's self attention (silo.h). The engine's own kernel gives each (query, head) item
// the whole key and value matrices to read from global memory; here a work-group of 128 queries of one head loads a tile
// of 32 keys and values into shared local memory once and every lane reads them from there (the same key for all
// lanes: a broadcast), and the online softmax is kept a tile at a time: the tile's 32 scores, their maximum, one
// rescale of the running sums, then the 32 weighted values (the engine's kernel rescales the 64 sums and takes two
// exponentials for every key). Float32 throughout, no contracted multiply-adds.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

// precise arithmetic as da.cpp: no contracted multiply-adds
#pragma float_control(precise, on)
#pragma clang fp contract(off)

namespace {
constexpr int WG = 128;   // queries a work-group
constexpr int BK = 32;    // keys a tile
constexpr int D = 64;
}  // namespace

extern "C" {

int ns_image_da_silo_supported(ns_gpu* g, int64_t d) {
    (void) g;
    return d == D ? 1 : 0;
}

int ns_image_da_silo_attention(ns_gpu* g, const float* qkv, int64_t N, int64_t H, int64_t d, float* out) {
    NS_TRY
    if (d != D) return ns_fail("ns_image_da_silo_attention: head size must be 64");
    if (N <= 0 || H <= 0) return 0;
    const int64_t HD = H * D, ld = 3 * HD;
    const float scale = 1.0f / sycl::sqrt((float) D);
    const int64_t blocks = (N + WG - 1) / WG;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ks(sycl::range<1>(BK * D), h);
        sycl::local_accessor<float, 1> vs(sycl::range<1>(BK * D), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (blocks * WG), (size_t) H), sycl::range<2>(WG, 1)), [=](sycl::nd_item<2> it) {
            const int64_t i = (int64_t) it.get_global_id(0);
            const int64_t hd = (int64_t) it.get_group(1);
            const int l = (int) it.get_local_id(0);
            const bool live = i < N;
            float q[D], acc[D];
            for (int e = 0; e < D; ++e) { q[e] = live ? qkv[i * ld + hd * D + e] : 0.0f; acc[e] = 0.0f; }
            float m = -3.0e38f, lsum = 0.0f;
            for (int64_t j0 = 0; j0 < N; j0 += BK) {
                const int64_t nk = sycl::min((int64_t) BK, N - j0);
                // the tile: 32 keys and values of D floats, WG lanes load 2048 floats each
                for (int t = l; t < BK * D; t += WG) {
                    const int64_t j = j0 + t / D, e = t % D;
                    ks[t] = j < N ? qkv[j * ld + HD + hd * D + e] : 0.0f;
                    vs[t] = j < N ? qkv[j * ld + 2 * HD + hd * D + e] : 0.0f;
                }
                sycl::group_barrier(it.get_group());
                if (live) {
                    // the tile's scores first, then one rescale of the running sums for the whole tile (not one a
                    // key: that was 64 multiplies and two exponentials a key - 46 ms a layer at 2,550 tokens on a B65)
                    float s[BK];
                    float mt = m;
                    for (int jj = 0; jj < BK; ++jj) {
                        float d0 = 0.0f;
                        for (int e = 0; e < D; ++e) d0 += q[e] * ks[jj * D + e];
                        s[jj] = jj < nk ? d0 * scale : -3.0e38f;
                        mt = sycl::fmax(mt, s[jj]);
                    }
                    const float a = sycl::exp(m - mt);
                    lsum *= a;
                    for (int e = 0; e < D; ++e) acc[e] *= a;
                    for (int jj = 0; jj < BK; ++jj) {
                        const float p = jj < nk ? sycl::exp(s[jj] - mt) : 0.0f;
                        lsum += p;
                        for (int e = 0; e < D; ++e) acc[e] += p * vs[jj * D + e];
                    }
                    m = mt;
                }
                sycl::group_barrier(it.get_group());
            }
            if (live) for (int e = 0; e < D; ++e) out[i * HD + hd * D + e] = acc[e] / lsum;
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
