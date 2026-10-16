// rignet.cpp: RigNet's kernels (rignet.h): the MLPs' ReLU + folded BatchNorm, EdgeConv's messages and their max per
// node, concatenation by columns. SYCL only, on the GPU's own queue; the linear layers are the shared diffusion
// kernels' (oneDNN, float32). GPLv3, as RigNet (github.com/zhan-xu/RigNet).
#include "ns.h"
#include "ns_internal.hpp"
#include "rignet.h"

#include <sycl/sycl.hpp>
#include <limits>

extern "C" {

int ns_model3d_rignet_relu_affine(ns_gpu* g, float* x, int64_t M, int64_t C, const float* scale, const float* shift) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t i = id[0] * C + id[1];
        float v = sycl::fmax(x[i], 0.0f);
        if (scale) v = v * scale[id[1]] + shift[id[1]];
        x[i] = v;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_rignet_act(ns_gpu* g, float* x, int64_t n, int kind) {
    NS_TRY
    if (kind < 0 || kind > 1) return ns_fail("ns_model3d_rignet_act: kind 0 (tanh) or 1 (sigmoid)");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        x[i] = kind == 0 ? sycl::tanh(v) : 1.0f / (1.0f + sycl::exp(-v));
    });
    return 0;
    NS_CATCH
}

int ns_model3d_rignet_edge_feat(ns_gpu* g, const float* x, int64_t C, const int32_t* src, const int32_t* dst, int64_t E, float* out) {
    NS_TRY
    if (E <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) E, (size_t) C), [=](sycl::id<2> id) {
        const int64_t e = id[0], c = id[1];
        const float xi = x[(int64_t) dst[e] * C + c];
        out[e * 2 * C + c] = xi;
        out[e * 2 * C + C + c] = x[(int64_t) src[e] * C + c] - xi;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_rignet_seg_max(ns_gpu* g, const float* x, int64_t C, const int32_t* rowptr, int64_t N, float* out, int64_t out_stride,
                              int64_t out_off) {
    NS_TRY
    if (N <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) N, (size_t) C), [=](sycl::id<2> id) {
        const int64_t i = id[0], c = id[1];
        const int64_t a = rowptr[i], b = rowptr[i + 1];
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t r = a; r < b; ++r) m = sycl::fmax(m, x[r * C + c]);
        out[i * out_stride + out_off + c] = b > a ? m : 0.0f;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_rignet_copy_cols(ns_gpu* g, const float* src, int64_t src_stride, int64_t M, int64_t C, float* dst, int64_t dst_stride,
                                int64_t dst_off) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        dst[id[0] * dst_stride + dst_off + id[1]] = src[id[0] * src_stride + id[1]];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
