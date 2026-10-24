// mi.cpp: the Ming-Image engine's own small kernels - what the shared diffusion kernels (nsd) do not have: the DiT's
// modulated norms and tanh-gated residuals, the text encoder's mixture-of-experts rows (gathered by expert, summed back
// by token), the grouped heads made full, the sampler's step, the VAE's upsampling, softmax and the picture's bytes.
// Elementwise or one work-group a row, on the GPU's queue. After nextcuda's mi.cu.
#include "ns.h"
#include "ns_internal.hpp"
#include "mi.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;

constexpr int WG = 256;
constexpr int PER = 32;  // a row of up to 8,192 values held in registers

inline float h2f(half h) { return (float) h; }
inline half f2h(float f) { return (half) f; }
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }

// one work-group of WG a row; every work-item of the group takes part in the reductions
inline sycl::nd_range<1> rows(int64_t M) { return sycl::nd_range<1>(sz(M) * WG, WG); }
inline sycl::nd_range<1> flat(int64_t n) { return sycl::nd_range<1>(((sz(n) + 255) / 256) * 256, 256); }
}  // namespace

extern "C" {

int ns_image_mi_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out) {
    NS_TRY
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_mi_ln_mod: rows longer than 8192");
    half* o = (half*) out;
    if (M > 0) g->q.parallel_for(rows(M), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const float* row = x + r * C;
        float v[PER];
        float s = 0.0f;
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            v[j] = i < C ? row[i] : 0.0f;
            s += v[j];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float s2 = 0.0f;
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            const float d = i < C ? v[j] - mean : 0.0f;
            s2 += d * d;
        }
        const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) o[r * C + i] = f2h((v[j] - mean) * inv * (scale ? 1.0f + scale[i] : 1.0f));
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_rms_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, void* out) {
    NS_TRY
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_mi_rms_mod: rows longer than 8192");
    half* o = (half*) out;
    if (M > 0) g->q.parallel_for(rows(M), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const float* row = x + r * C;
        float v[PER];
        float s = 0.0f;
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            v[j] = i < C ? row[i] : 0.0f;
            s += v[j] * v[j];
        }
        const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C + eps);
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) o[r * C + i] = f2h(v[j] * inv * (w ? w[i] : 1.0f) * (scale ? 1.0f + scale[i] : 1.0f));
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out) {
    NS_TRY
    half* o = (half*) out;
    if (M > 0) g->q.parallel_for(rows(M), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_local_id(0);
        const float* row = x + r * N;
        float m = -INFINITY;
        for (int64_t i = l; i < N; i += WG) m = sycl::fmax(m, row[i] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t i = l; i < N; i += WG) s += sycl::exp(row[i] * scale - m);
        const float inv = 1.0f / sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        for (int64_t i = l; i < N; i += WG) o[r * N + i] = f2h(sycl::exp(row[i] * scale - m) * inv);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_silu(ns_gpu* g, void* xv, int64_t n) {
    NS_TRY
    half* p = (half*) xv;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> id) {
        const float v = h2f(p[id]);
        p[id] = f2h(v / (1.0f + sycl::exp(-v)));
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { x[i] += a * y[i]; });
    return 0;
    NS_CATCH
}

int ns_image_mi_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { o[i] = f2h(x[i]); });
    return 0;
    NS_CATCH
}

int ns_image_mi_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    const half* p = (const half*) x;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { out[i] = h2f(p[i]); });
    return 0;
    NS_CATCH
}

int ns_image_mi_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    NS_TRY
    const half* s = (const half*) x;
    half* o = (half*) out;
    const int64_t n = 4 * H * W * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> id) {
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
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) {
        const float v = sycl::fmin(sycl::fmax(h2f(s[i]), -1.0f), 1.0f) * 0.5f + 0.5f;
        rgba[i] = (uint8_t) sycl::fmin(sycl::fmax(sycl::round(v * 255.0f), 0.0f), 255.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_gather_rows(ns_gpu* g, const void* x, int64_t C, const int32_t* idx, int64_t n, void* out) {
    NS_TRY
    if (n <= 0) return 0;
    const half* xs = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(rows(n), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int64_t src = idx[r];
        for (int64_t i = it.get_local_id(0); i < C; i += WG) o[r * C + i] = xs[src * C + i];
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_combine(ns_gpu* g, float* x, int64_t M, int64_t C, const void* y, const int32_t* slot, const float* w, int64_t K) {
    NS_TRY
    if (M <= 0) return 0;
    const float* ys = (const float*) y;
    g->q.parallel_for(rows(M), [=](sycl::nd_item<1> it) {
        const int64_t t = it.get_group(0);
        for (int64_t i = it.get_local_id(0); i < C; i += WG) {
            float s = 0.0f;
            for (int64_t j = 0; j < K; ++j) {
                const int32_t sl = slot[t * K + j];
                if (sl >= 0) s += w[t * K + j] * ys[(int64_t) sl * C + i];
            }
            x[t * C + i] += s;
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_swiglu_scaled(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out) {
    NS_TRY
    const half* xs = (const half*) x;
    half* o = (half*) out;
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t r = i / C, c = i % C;
        const float a = h2f(xs[r * 2 * C + c]), b = h2f(xs[r * 2 * C + C + c]);
        o[i] = f2h(a / (1.0f + sycl::exp(-a)) * b * s);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out, int64_t ld) {
    NS_TRY
    half* o = (half*) out;
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        o[(i / C) * ld + i % C] = f2h(x[i] * s);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* other, const float* gate) {
    NS_TRY
    const half* o = (const half*) other;
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        x[i] += h2f(o[i]) * (gate ? gate[i % C] : 1.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_mi_repeat_heads(ns_gpu* g, const void* x, int64_t rws, int64_t stride, int64_t off, int64_t Hkv, int64_t Hq, int64_t D, void* out) {
    NS_TRY
    if (Hkv <= 0 || Hq % Hkv != 0) return ns_fail("ns_image_mi_repeat_heads: query heads not a multiple of the key / value heads");
    if (rws <= 0) return 0;
    const half* xs = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(rows(rws), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int64_t rep = Hq / Hkv;
        for (int64_t i = it.get_local_id(0); i < Hq * D; i += WG) {
            const int64_t h = i / D, d = i % D;
            o[r * Hq * D + i] = xs[r * stride + off + (h / rep) * D + d];
        }
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
