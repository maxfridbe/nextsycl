// mo.cpp: the MOSS-Image engine's own small kernels - what the shared diffusion kernels (nsd) do not have: the DiT's
// soft-clipped gated activation, conversions, the sampler's step and guidance. Elementwise, on the GPU's queue.
#include "ns.h"
#include "ns_internal.hpp"
#include "mo.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
}  // namespace

extern "C" {

int ns_image_mo_situ(ns_gpu* g, const float* hid, int64_t ldh, const float* gate, int64_t ldg, int64_t M, int64_t C, void* out, int64_t ldo) {
    NS_TRY
    half* o = (half*) out;
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t r = i / C, c = i % C;
        const float h = hid[r * ldh + c], gt = gate[r * ldg + c];
        const float gain = 4.0f * sycl::tanh(gt / 4.0f) * (1.0f / (1.0f + sycl::exp(-gt)));
        o[r * ldo + c] = (half) (gain * (25.0f * sycl::tanh(h / 25.0f)));
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { o[i] = (half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_image_mo_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    const half* xs = (const half*) x;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { out[i] = (float) xs[i]; });
    return 0;
    NS_CATCH
}

int ns_image_mo_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { x[i] += a * y[i]; });
    return 0;
    NS_CATCH
}

int ns_image_mo_cfg(ns_gpu* g, float* out, const float* cond, const float* uncond, int64_t n, float s) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { out[i] = uncond[i] + s * (cond[i] - uncond[i]); });
    return 0;
    NS_CATCH
}

}  // extern "C"
