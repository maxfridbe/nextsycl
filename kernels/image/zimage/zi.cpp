// zi.cpp: the Z-Image engine's own small kernels - what the shared diffusion kernels (nsd) do not have. The DiT's
// modulated norms, tanh-gated residuals and scaled half conversions began as Ming-Image's (mi.cpp: the same
// transformer), the VAE's group norm, softmax, upsample and picture bytes as MOSS-Image's (vae.cpp: the same decoder
// blocks); copies, so that tuning either engine leaves the other as it was checked. Elementwise or one work-group a
// row, on the GPU's queue.
#include "ns.h"
#include "ns_internal.hpp"
#include "zi.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;

constexpr int WG = 256;
constexpr int PER = 32;  // a row of up to 8,192 values held in registers
constexpr int NB = 64;   // the blocks of pixels a group norm's statistics are split in

inline float h2f(half h) { return (float) h; }
inline half f2h(float f) { return (half) f; }
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }

// one work-group of WG a row; every work-item of the group takes part in the reductions
inline sycl::nd_range<1> rows(int64_t M) { return sycl::nd_range<1>(sz(M) * WG, WG); }
}  // namespace

extern "C" {

int ns_image_zi_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out) {
    NS_TRY
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_zi_ln_mod: rows longer than 8192");
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

int ns_image_zi_rms_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, void* out) {
    NS_TRY
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_zi_rms_mod: rows longer than 8192");
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

int ns_image_zi_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* other, const float* gate) {
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

int ns_image_zi_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out, int64_t ld) {
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

int ns_image_zi_swiglu_scaled(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out) {
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

int ns_image_zi_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { x[i] += a * y[i]; });
    return 0;
    NS_CATCH
}

int ns_image_zi_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { o[i] = f2h(x[i]); });
    return 0;
    NS_CATCH
}

int ns_image_zi_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    const half* p = (const half*) x;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { out[i] = h2f(p[i]); });
    return 0;
    NS_CATCH
}

int ns_image_zi_swiglu_f(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out) {
    NS_TRY
    half* o = (half*) out;
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t r = i / C, c = i % C;
        const float a = x[r * 2 * C + c], b = x[r * 2 * C + C + c];
        o[i] = f2h(a / (1.0f + sycl::exp(-a)) * b * s);
    });
    return 0;
    NS_CATCH
}

int64_t ns_image_zi_group_norm_scratch(int64_t groups) { return groups * (NB + 2); }

int ns_image_zi_group_norm(ns_gpu* g, const void* x, int64_t px, int64_t C, int64_t groups, float eps, const float* gamma, const float* beta, int silu, void* out,
                           float* scratch) {
    NS_TRY
    if (px <= 0) return 0;
    if (C % groups != 0) return ns_fail("ns_image_zi_group_norm: C a multiple of the groups");
    const half* xs = (const half*) x;
    half* o = (half*) out;
    const int64_t cg = C / groups;
    float* part = scratch;                      // [groups][NB]
    float* stat = scratch + groups * NB;        // [groups][2]: mean, then rstd
    const float n = (float) (px * cg);
    // pass 1: the partial sums, then the means; pass 2: the partial sums of squared deviations, then the rstd
    for (int pass = 0; pass < 2; ++pass) {
        g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>(sz(groups * NB) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0) / NB, b = it.get_group(0) % NB, l = it.get_local_id(0);
            const int64_t p0 = px * b / NB, p1 = px * (b + 1) / NB;
            const float mean = pass == 0 ? 0.0f : stat[gi * 2];
            float s = 0.0f;
            for (int64_t i = l; i < (p1 - p0) * cg; i += WG) {
                const float v = (float) xs[(p0 + i / cg) * C + gi * cg + i % cg] - mean;
                s += pass == 0 ? v : v * v;
            }
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            if (l == 0) part[gi * NB + b] = s;
        });
        g->q.parallel_for(sycl::range<1>(sz(groups)), [=](sycl::id<1> id) {
            const int64_t gi = id[0];
            float s = 0.0f;
            for (int b = 0; b < NB; ++b) s += part[gi * NB + b];
            if (pass == 0) stat[gi * 2] = s / n;
            else stat[gi * 2 + 1] = 1.0f / sycl::sqrt(s / n + eps);
        });
    }
    const int64_t total = px * C;
    g->q.parallel_for(sycl::range<1>(sz(total)), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % C, gi = c / cg;
        float v = ((float) xs[i] - stat[gi * 2]) * stat[gi * 2 + 1] * gamma[c] + beta[c];
        if (silu) v = v / (1.0f + sycl::exp(-v));
        o[i] = (half) v;
    });
    return 0;
    NS_CATCH
}

int ns_image_zi_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out) {
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
        for (int64_t i = l; i < N; i += WG) o[r * N + i] = (half) (sycl::exp(row[i] * scale - m) * inv);
    });
    return 0;
    NS_CATCH
}

int ns_image_zi_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
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

int ns_image_zi_to_rgb8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgb) {
    NS_TRY
    const half* s = (const half*) x;
    const int64_t n = H * W * 3;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) {
        const float v = sycl::fmin(sycl::fmax((float) s[i], -1.0f), 1.0f) * 0.5f + 0.5f;
        rgb[i] = (uint8_t) sycl::fmin(sycl::fmax(sycl::round(v * 255.0f), 0.0f), 255.0f);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
