// mi.cpp: the Ming-Image engine's own small kernels (SYCL; nextcuda's mi.cu with the same names and arithmetic) - what
// the shared diffusion kernels (nsd) do not have: the DiT's modulated norms and tanh-gated residuals, the text
// encoder's mixture-of-experts rows (gathered by expert, summed back by token), the grouped heads made full, the
// sampler's step, the VAE's upsampling, softmax and the picture's bytes, and a strided copy (a product's slice of
// columns put in its place). Elementwise or one work-group a row, on the GPU's queue; every loop bounded by its arguments.
#include "ns.h"
#include "ns_internal.hpp"
#include "mi.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int WG = 256;
constexpr int PER = 32;  // a row of up to 8,192 values held in registers
}  // namespace

extern "C" {

int ns_image_mi_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out) {
    NS_TRY
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_mi_ln_mod: rows longer than 8192");
    if (M <= 0) return 0;
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

int ns_image_mi_rms_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, void* out) {
    NS_TRY
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_mi_rms_mod: rows longer than 8192");
    if (M <= 0) return 0;
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
            s += v[j] * v[j];
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C + eps);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) o[r * C + i] = (half) (v[j] * inv * (w ? w[i] : 1.0f) * (scale ? 1.0f + scale[i] : 1.0f));
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out) {
    NS_TRY
    if (M <= 0 || N <= 0) return 0;
    half* o = (half*) out;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_local_id(0);
        const float* row = x + r * N;
        float m = -INFINITY;
        for (int64_t i = l; i < N; i += WG) m = sycl::fmax(m, row[i] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t i = l; i < N; i += WG) s += sycl::exp(row[i] * scale - m);
        const float inv = 1.0f / sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        for (int64_t i = l; i < N; i += WG) o[r * N + i] = (half) (sycl::exp(row[i] * scale - m) * inv);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_silu(ns_gpu* g, void* x, int64_t n) {
    NS_TRY
    half* p = (half*) x;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = (float) p[i];
        p[i] = (half) (v / (1.0f + sycl::exp(-v)));
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += a * y[i]; });
    return 0;
    NS_CATCH
}

int ns_image_mi_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_image_mi_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    const half* p = (const half*) x;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = (float) p[i]; });
    return 0;
    NS_CATCH
}

int ns_image_mi_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    NS_TRY
    const half* s = (const half*) x;
    half* o = (half*) out;
    const int64_t n = 4 * H * W * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t c = i % C, xx = (i / C) % (2 * W), y = i / (C * 2 * W);
        o[i] = s[((y / 2) * W + xx / 2) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_to_rgba8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgba) {
    NS_TRY
    const half* s = (const half*) x;
    const int64_t n = H * W * 4;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = sycl::fmin(sycl::fmax((float) s[i], -1.0f), 1.0f) * 0.5f + 0.5f;
        rgba[i] = (uint8_t) sycl::fmin(sycl::fmax(sycl::round(v * 255.0f), 0.0f), 255.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_gather_rows(ns_gpu* g, const void* x, int64_t C, const int32_t* idx, int64_t n, void* out) {
    NS_TRY
    if (n <= 0 || C <= 0) return 0;
    const half* s = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) C), [=](sycl::id<2> id) {
        o[id[0] * C + id[1]] = s[(int64_t) idx[id[0]] * C + id[1]];
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_combine(ns_gpu* g, float* x, int64_t M, int64_t C, const void* y, const int32_t* slot, const float* w, int64_t K) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    const float* yf = (const float*) y;
    // each (token, feature) sums its K experts in order: the same sum every run
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const int64_t t = id[0], i = id[1];
        float s = 0.0f;
        for (int64_t j = 0; j < K; ++j) {
            const int32_t sl = slot[t * K + j];
            if (sl >= 0) s += w[t * K + j] * yf[(int64_t) sl * C + i];
        }
        x[t * C + i] += s;
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_swiglu_scaled(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out) {
    NS_TRY
    const half* p = (const half*) x;
    half* o = (half*) out;
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / C, c = i % C;
        const float a = (float) p[r * 2 * C + c], b = (float) p[r * 2 * C + C + c];
        o[i] = (half) (a / (1.0f + sycl::exp(-a)) * b * s);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out, int64_t ld) {
    NS_TRY
    half* o = (half*) out;
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        o[(i / C) * ld + i % C] = (half) (x[i] * s);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* other, const float* gate) {
    NS_TRY
    const half* o = (const half*) other;
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        x[i] += (float) o[i] * (gate ? gate[i % C] : 1.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_repeat_heads(ns_gpu* g, const void* x, int64_t rows, int64_t stride, int64_t off, int64_t Hkv, int64_t Hq, int64_t D, void* out) {
    NS_TRY
    if (Hkv <= 0 || Hq % Hkv != 0) return ns_fail("ns_image_mi_repeat_heads: query heads not a multiple of the key / value heads");
    if (rows <= 0) return 0;
    const half* s = (const half*) x;
    half* o = (half*) out;
    const int64_t rep = Hq / Hkv;
    g->q.parallel_for(sycl::range<2>((size_t) rows, (size_t) (Hq * D)), [=](sycl::id<2> id) {
        const int64_t r = id[0], i = id[1], h = i / D, d = i % D;
        o[r * Hq * D + i] = s[r * stride + off + (h / rep) * D + d];
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_copy_2d(ns_gpu* g, void* dst, int64_t dst_pitch, const void* src, int64_t src_pitch, int64_t width, int64_t height) {
    NS_TRY
    if (width <= 0 || height <= 0) return 0;
    uint8_t* d = (uint8_t*) dst;
    const uint8_t* s = (const uint8_t*) src;
    if (width % 4 == 0 && dst_pitch % 4 == 0 && src_pitch % 4 == 0 && ((uintptr_t) d | (uintptr_t) s) % 4 == 0) {
        // four bytes a work-item
        const int64_t w4 = width / 4;
        g->q.parallel_for(sycl::range<2>((size_t) height, (size_t) w4), [=](sycl::id<2> id) {
            ((uint32_t*) (d + id[0] * dst_pitch))[id[1]] = ((const uint32_t*) (s + id[0] * src_pitch))[id[1]];
        });
    } else {
        g->q.parallel_for(sycl::range<2>((size_t) height, (size_t) width), [=](sycl::id<2> id) {
            d[id[0] * dst_pitch + id[1]] = s[id[0] * src_pitch + id[1]];
        });
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
