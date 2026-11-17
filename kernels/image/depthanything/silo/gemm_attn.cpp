// gemm_attn.cpp: the pieces around the Depth Anything V2 silo's attention as products (silo.h): the engine multiplies
// q k^T and p v on the shared float32 product (oneDNN, the diffusion kernels' linear) a head at a time; these kernels
// lay the heads out for it (q and k [H, N, D], v transposed [H, D, N]), take the scaled softmax of a score row, and put
// the heads' outputs back side by side. Float32, no contracted multiply-adds (the products' own accumulation aside).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

#pragma float_control(precise, on)
#pragma clang fp contract(off)

namespace {
constexpr int SG = 32;
}

extern "C" {

int ns_image_da_silo_split(ns_gpu* g, const float* qkv, int64_t N, int64_t H, int64_t D, float* q, float* k, float* vt) {
    NS_TRY
    if (N <= 0 || H <= 0 || D <= 0) return 0;
    const int64_t HD = H * D, ld = 3 * HD, total = N * HD;
    g->q.parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0];
        const int64_t i = t / HD, hd = t % HD, h = hd / D, e = hd % D;
        q[(h * N + i) * D + e] = qkv[i * ld + hd];
        k[(h * N + i) * D + e] = qkv[i * ld + HD + hd];
        vt[(h * D + e) * N + i] = qkv[i * ld + 2 * HD + hd];
    });
    return 0;
    NS_CATCH
}

int ns_image_da_silo_softmax(ns_gpu* g, float* s, int64_t rows, int64_t cols, float scale) {
    NS_TRY
    if (rows <= 0 || cols <= 0) return 0;
    // a sub-group a row: the maximum, the sum of exponentials, then each value over the sum
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (rows * SG)), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int64_t r = (int64_t) it.get_group(0);
        const int l = (int) it.get_local_id(0);
        auto sg = it.get_sub_group();
        float* row = s + r * cols;
        float m = -3.0e38f;
        for (int64_t j = l; j < cols; j += SG) m = sycl::fmax(m, row[j] * scale);
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        float sum = 0.0f;
        for (int64_t j = l; j < cols; j += SG) {
            const float p = sycl::exp(row[j] * scale - m);
            row[j] = p;
            sum += p;
        }
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        const float inv = 1.0f / sum;
        for (int64_t j = l; j < cols; j += SG) row[j] *= inv;
    });
    return 0;
    NS_CATCH
}

int ns_image_da_silo_merge(ns_gpu* g, const float* o, int64_t N, int64_t H, int64_t D, float* out) {
    NS_TRY
    if (N <= 0 || H <= 0 || D <= 0) return 0;
    const int64_t HD = H * D, total = N * HD;
    g->q.parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> id) {
        const int64_t t = (int64_t) id[0];
        const int64_t i = t / HD, hd = t % HD, h = hd / D, e = hd % D;
        out[t] = o[(h * N + i) * D + e];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
