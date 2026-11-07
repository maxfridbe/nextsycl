// rb.cpp: the background removers' own kernels (rb.h): IS-Net's ceil-mode max-pool, its half-pixel bilinear
// upsampling, a bias with ReLU, the sigmoid. Elementwise, on the GPU's queue. After nextcuda's rb.cu.
#include "ns.h"
#include "ns_internal.hpp"
#include "rb.h"

#include <sycl/sycl.hpp>
#include <cfloat>

// precise arithmetic here whatever the library's flags (the image kinds build without -fp-model=precise): no contracted
// multiply-adds in the upsampling, exp and the division of the sigmoid as onnxruntime's CPU computes them
#pragma float_control(precise, on)
#pragma clang fp contract(off)

namespace {
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
}  // namespace

extern "C" {

int ns_image_rb_maxpool2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out) {
    NS_TRY
    const int64_t Ho = (H + 1) / 2, Wo = (W + 1) / 2;
    const int64_t n = Ho * Wo * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % C, p = id / C, xo = p % Wo, yo = p / Wo;
        float m = -FLT_MAX;
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const int64_t y = 2 * yo + dy, xx = 2 * xo + dx;
                if (y < H && xx < W) m = sycl::fmax(m, x[(y * W + xx) * C + c]);
            }
        }
        out[id] = m;
    });
    return 0;
    NS_CATCH
}

int ns_image_rb_resize(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t os) {
    NS_TRY
    const int64_t n = Ho * Wo * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % C, p = id / C, xo = p % Wo, yo = p / Wo;
        const float sy = sycl::fmax(((float) yo + 0.5f) * (float) Hi / (float) Ho - 0.5f, 0.0f);
        const float sx = sycl::fmax(((float) xo + 0.5f) * (float) Wi / (float) Wo - 0.5f, 0.0f);
        const int64_t y0 = sycl::min((int64_t) sy, Hi - 1), x0 = sycl::min((int64_t) sx, Wi - 1);
        const int64_t y1 = sycl::min(y0 + 1, Hi - 1), x1 = sycl::min(x0 + 1, Wi - 1);
        const float fy = sy - (float) y0, fx = sx - (float) x0;
        const float a = x[(y0 * Wi + x0) * C + c], b = x[(y0 * Wi + x1) * C + c];
        const float d = x[(y1 * Wi + x0) * C + c], e = x[(y1 * Wi + x1) * C + c];
        out[(yo * Wo + xo) * os + c] = (a * (1.0f - fx) + b * fx) * (1.0f - fy) + (d * (1.0f - fx) + e * fx) * fy;
    });
    return 0;
    NS_CATCH
}

int ns_image_rb_bias_act(ns_gpu* g, float* x, int64_t M, int64_t C, const float* bias, int relu) {
    NS_TRY
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const float v = x[id] + (bias ? bias[id % C] : 0.0f);
        x[id] = relu ? sycl::fmax(v, 0.0f) : v;
    });
    return 0;
    NS_CATCH
}

int ns_image_rb_sigmoid(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { x[i] = 1.0f / (1.0f + sycl::exp(-x[i])); });
    return 0;
    NS_CATCH
}

}  // extern "C"
