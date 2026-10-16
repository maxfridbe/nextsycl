// bz2.cpp: the Breeze TTS 2 engine's own kernels (bz2.h). The text encoder's sequences are a prompt's text segments
// (tens to a few hundred tokens) and the depth decoder's rows two at most, so these are plain item-per-row kernels;
// the cached decode is the MiniMax Music engine's mm3 kernels, the linear layers oneDNN's (nsd).
#include "ns.h"
#include "ns_internal.hpp"
#include "bz2.h"

#include <sycl/sycl.hpp>
#include <cmath>

extern "C" {

int ns_audio_bz2_head_norm_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* weight, float eps,
                                const float* inv_freq, int64_t p0) {
    NS_TRY
    if (D > 256 || D % 2 != 0 || S <= 0) return ns_fail("ns_audio_bz2_head_norm_rope: D even, at most 256");
    if (rows <= 0 || H <= 0) return 0;
    const int64_t half = D / 2;
    // one work-item a (row, head): the norm over its D values, then each pair rotated
    g->q.parallel_for(sycl::range<1>((size_t) (rows * H)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / H, h = id[0] % H;
        float* v = x + r * xs + h * D;
        if (weight) {
            float ss = 0.0f;
            for (int64_t d = 0; d < D; ++d) ss += v[d] * v[d];
            const float inv = sycl::rsqrt(ss / (float) D + eps);
            for (int64_t d = 0; d < D; ++d) v[d] = v[d] * inv * weight[d];
        }
        const float pos = (float) (p0 + r % S);
        for (int64_t j = 0; j < half; ++j) {
            const float a = pos * inv_freq[j], c = sycl::cos(a), s = sycl::sin(a);
            const float x1 = v[j], x2 = v[j + half];
            v[j] = x1 * c - x2 * s;
            v[j + half] = x2 * c + x1 * s;
        }
    });
    return 0;
    NS_CATCH
}

int ns_audio_bz2_attn_window(ns_gpu* g, const float* q, const float* k, const float* v, int64_t L, int64_t Hq, int64_t Hkv, int64_t D,
                             int64_t qs, int64_t kvs, int64_t left, int64_t right, float scale, float* out) {
    NS_TRY
    if (D > 256 || Hkv <= 0 || Hq % Hkv != 0 || left < 1 || right < 1) return ns_fail("ns_audio_bz2_attn_window: D at most 256, Hq a multiple of Hkv, windows of 1 or more");
    if (L <= 0 || Hq <= 0) return 0;
    const int64_t G = Hq / Hkv;
    // one work-item a (query, head): the window's scores (max first, then the weighted values)
    g->q.parallel_for(sycl::range<1>((size_t) (L * Hq)), [=](sycl::id<1> id) {
        const int64_t i = id[0] / Hq, h = id[0] % Hq, kh = h / G;
        const float* qi = q + i * qs + h * D;
        const int64_t j0 = i - left + 1 > 0 ? i - left + 1 : 0;
        const int64_t j1 = i + right - 1 < L - 1 ? i + right - 1 : L - 1;
        float mx = -INFINITY;
        for (int64_t j = j0; j <= j1; ++j) {
            const float* kj = k + j * kvs + kh * D;
            float s = 0.0f;
            for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
            mx = sycl::fmax(mx, s * scale);
        }
        float acc[256];
        for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
        float sum = 0.0f;
        for (int64_t j = j0; j <= j1; ++j) {
            const float* kj = k + j * kvs + kh * D;
            float s = 0.0f;
            for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
            const float p = sycl::exp(s * scale - mx);
            sum += p;
            const float* vj = v + j * kvs + kh * D;
            for (int64_t d = 0; d < D; ++d) acc[d] += p * vj[d];
        }
        float* o = out + i * (Hq * D) + h * D;
        for (int64_t d = 0; d < D; ++d) o[d] = acc[d] / sum;
    });
    return 0;
    NS_CATCH
}

int ns_audio_bz2_gelu_mul(ns_gpu* g, const float* gu, int64_t R, int64_t F, float* out) {
    NS_TRY
    if (R <= 0 || F <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (R * F)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / F, f = id[0] % F;
        const float a = gu[r * 2 * F + f], u = gu[r * 2 * F + F + f];
        // gelu_pytorch_tanh: 0.5 a (1 + tanh(sqrt(2 / pi) (a + 0.044715 a^3)))
        const float t = sycl::tanh(0.7978845608028654f * (a + 0.044715f * a * a * a));
        out[id[0]] = 0.5f * a * (1.0f + t) * u;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
