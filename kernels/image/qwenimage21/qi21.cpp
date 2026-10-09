// qi21.cpp: the Qwen-Image 2.1 engine's own small kernels - what the shared diffusion kernels (nsd) do not have:
// the norm with the modulation's (1 + scale), the activations in place, the sampler's step, the VAE's upsampling and shortcut, the
// picture's bytes. Elementwise, on the GPU's queue.
#include "ns.h"
#include "ns_internal.hpp"
#include "qi21.h"

#include <sycl/sycl.hpp>

namespace {
using half = sycl::half;
}

extern "C" {

int ns_image_qi21_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out) {
    NS_TRY
    constexpr int WG = 256, PER = 32;            // a row of up to 8,192 values held in registers
    if (C > (int64_t) WG * PER) {
        return ns_fail("ns_image_qi21_ln_mod: rows longer than 8192");
    }
    half* o = (half*) out;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const float* row = x + r * C;
        float v[PER];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            v[j] = i < C ? row[i] : 0.0f;
            s += v[j];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float s2 = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            const float d = i < C ? v[j] - mean : 0.0f;
            s2 += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) o[r * C + i] = (half) ((v[j] - mean) * inv * (scale ? 1.0f + scale[i] : 1.0f));
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_qi21_gelu_tanh(ns_gpu* g, void* x, int64_t n) {
    NS_TRY
    half* p = (half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = (float) p[i];
        p[i] = (half) (0.5f * v * (1.0f + sycl::tanh(0.7978845608f * (v + 0.044715f * v * v * v))));
    });
    return 0;
    NS_CATCH
}

int ns_image_qi21_silu(ns_gpu* g, void* x, int64_t n) {
    NS_TRY
    half* p = (half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = (float) p[i];
        p[i] = (half) (v / (1.0f + sycl::exp(-v)));
    });
    return 0;
    NS_CATCH
}

int ns_image_qi21_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += a * y[i]; });
    return 0;
    NS_CATCH
}

int ns_image_qi21_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_image_qi21_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    const half* p = (const half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = (float) p[i]; });
    return 0;
    NS_CATCH
}

int ns_image_qi21_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    NS_TRY
    const half* s = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<3>((size_t) (2 * H), (size_t) (2 * W), (size_t) C), [=](sycl::id<3> id) {
        const size_t y = id[0], xx = id[1], c = id[2];
        o[(y * 2 * W + xx) * C + c] = s[((y / 2) * W + xx / 2) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_image_qi21_dupup_add(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t Ci, int64_t Co, int ft, void* out) {
    NS_TRY
    if (ft != 1 && ft != 2) return ns_fail("ns_image_qi21_dupup_add: 1 or 2 time duplicates");
    const int64_t F = (int64_t) ft * 4;
    if ((Co * F) % Ci != 0) return ns_fail("ns_image_qi21_dupup_add: out channels x factor not a multiple of in channels");
    const int64_t R = Co * F / Ci;
    const int64_t t = ft - 1;
    const half* s = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<3>((size_t) (2 * H), (size_t) (2 * W), (size_t) Co), [=](sycl::id<3> id) {
        const int64_t y = id[0], xx = id[1], oc = id[2];
        const int64_t ci = (oc * F + t * 4 + (y % 2) * 2 + (xx % 2)) / R;
        const size_t d = ((size_t) y * 2 * W + xx) * Co + oc;
        o[d] = (half) ((float) o[d] + (float) s[((size_t) (y / 2) * W + xx / 2) * Ci + ci]);
    });
    return 0;
    NS_CATCH
}

int ns_image_qi21_to_rgba8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgba) {
    NS_TRY
    const half* s = (const half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) (H * W * 4)), [=](sycl::id<1> i) {
        const float v = sycl::clamp((float) s[i], -1.0f, 1.0f) * 0.5f + 0.5f;
        rgba[i] = (uint8_t) sycl::clamp(sycl::round(v * 255.0f), 0.0f, 255.0f);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
