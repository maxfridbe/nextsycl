// q3t.cpp: the Qwen3-TTS engine's own kernels (q3t.h). The codec decoder's sequences are short (a chunk: 325 frames at
// 12.5 Hz, then the upsampled waveform's channels) so these are plain element-per-item kernels; the language models'
// decode runs on the MiniMax Music engine's mm3 kernels.
#include "ns.h"
#include "ns_internal.hpp"
#include "q3t.h"

#include <sycl/sycl.hpp>
#include <cmath>

extern "C" {

int ns_audio_q3t_snake_beta(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* log_alpha, const float* log_beta, float* out) {
    NS_TRY
    if (C > 0 && L > 0) g->q.parallel_for(sycl::range<1>((size_t) (C * L)), [=](sycl::id<1> id) {
        const int64_t c = id[0] / L;
        const float v = x[id[0]], s = sycl::sin(v * sycl::exp(log_alpha[c]));
        out[id[0]] = v + s * s / (sycl::exp(log_beta[c]) + 1e-9f);
    });
    return 0;
    NS_CATCH
}

int ns_audio_q3t_dwconv(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* bias, int64_t K, float* out) {
    NS_TRY
    if (C > 0 && L > 0) g->q.parallel_for(sycl::range<1>((size_t) (C * L)), [=](sycl::id<1> id) {
        const int64_t c = id[0] / L, t = id[0] % L;
        float a = bias ? bias[c] : 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            const int64_t s = t - K + 1 + k;
            if (s >= 0) a += w[c * K + k] * x[c * L + s];
        }
        out[id[0]] = a;
    });
    return 0;
    NS_CATCH
}

int ns_audio_q3t_window_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t L, int64_t H, int64_t D, int64_t qs, int64_t kvs,
                             int64_t W, float* out) {
    NS_TRY
    if (L <= 0 || H <= 0) return 0;
    const float scale = 1.0f / sycl::sqrt((float) D);
    // one work-item a (query, head): the window's scores, then the values (D <= 128)
    g->q.parallel_for(sycl::range<1>((size_t) (L * H)), [=](sycl::id<1> id) {
        const int64_t i = id[0] / H, h = id[0] % H;
        const float* qi = q + i * qs + h * D;
        const int64_t j0 = i - W + 1 > 0 ? i - W + 1 : 0;
        float mx = -INFINITY;
        for (int64_t j = j0; j <= i; ++j) {
            const float* kj = k + j * kvs + h * D;
            float s = 0.0f;
            for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
            mx = sycl::fmax(mx, s * scale);
        }
        float acc[128];
        for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
        float sum = 0.0f;
        for (int64_t j = j0; j <= i; ++j) {
            const float* kj = k + j * kvs + h * D;
            float s = 0.0f;
            for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
            const float p = sycl::exp(s * scale - mx);
            sum += p;
            const float* vj = v + j * kvs + h * D;
            for (int64_t d = 0; d < D; ++d) acc[d] += p * vj[d];
        }
        float* o = out + i * (H * D) + h * D;
        for (int64_t d = 0; d < D; ++d) o[d] = acc[d] / sum;
    });
    return 0;
    NS_CATCH
}

int ns_audio_q3t_rope(ns_gpu* g, float* x, int64_t xs, int64_t L, int64_t H, int64_t D, float theta, int64_t p0) {
    NS_TRY
    const int64_t half = D / 2;
    if (L > 0) g->q.parallel_for(sycl::range<1>((size_t) (L * H * half)), [=](sycl::id<1> id) {
        const int64_t i = id[0] / (H * half), h = (id[0] / half) % H, j = id[0] % half;
        const float inv = sycl::pow(theta, -2.0f * (float) j / (float) D);
        const float a = (float) (p0 + i) * inv, c = sycl::cos(a), s = sycl::sin(a);
        float* r = x + i * xs + h * D;
        const float x1 = r[j], x2 = r[j + half];
        r[j] = x1 * c - x2 * s;
        r[j + half] = x2 * c + x1 * s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_q3t_scale_add(ns_gpu* g, float* x, const float* y, const float* scale, int64_t R, int64_t C) {
    NS_TRY
    if (R > 0 && C > 0) g->q.parallel_for(sycl::range<1>((size_t) (R * C)), [=](sycl::id<1> id) { x[id[0]] += scale[id[0] % C] * y[id[0]]; });
    return 0;
    NS_CATCH
}

int ns_audio_q3t_silu(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { const float v = x[i]; x[i] = v / (1.0f + sycl::exp(-v)); });
    return 0;
    NS_CATCH
}

int ns_audio_q3t_clamp(ns_gpu* g, float* x, int64_t n, float lo, float hi) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] = sycl::clamp(x[i], lo, hi); });
    return 0;
    NS_CATCH
}

}  // extern "C"
