// vcp.cpp: the VoxCPM2 engine's own kernels (vcp.h). Everything here is small - a local sequence is 5 or 11 rows, the
// VAE's channels a few hundred thousand samples - so these are plain element-per-item kernels.
#include "ns.h"
#include "ns_internal.hpp"
#include "vcp.h"

#include <sycl/sycl.hpp>
#include <cmath>

extern "C" {

int ns_audio_vcp_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0) {
    NS_TRY
    const int64_t half = D / 2;
    if (rows > 0 && S > 0) g->q.parallel_for(sycl::range<1>((size_t) (rows * H * half)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / (H * half), h = (id[0] / half) % H, i = id[0] % half;
        float* p = x + r * xs + h * D;
        const float a = (float) (p0 + r % S) * inv_freq[i], c = sycl::cos(a), s = sycl::sin(a);
        const float x1 = p[i], x2 = p[i + half];
        p[i] = x1 * c - x2 * s;
        p[i + half] = x2 * c + x1 * s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_vcp_local_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t B, int64_t S, int64_t Hq, int64_t Hkv, int64_t D,
                            int64_t qs, int64_t kvs, float* out) {
    NS_TRY
    if (B <= 0 || S <= 0) return 0;
    if (D > 128 || Hq % Hkv != 0) return ns_fail("ns_audio_vcp_local_attn: D <= 128, query heads a multiple of the key heads");
    const float scale = 1.0f / sycl::sqrt((float) D);
    const int64_t G = Hq / Hkv;
    g->q.parallel_for(sycl::range<1>((size_t) (B * S * Hq)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / Hq, h = id[0] % Hq, b = r / S, kh = h / G;
        const float* qi = q + r * qs + h * D;
        float sc[64];
        float mx = -INFINITY;
        for (int64_t j = 0; j < S && j < 64; ++j) {
            const float* kj = k + (b * S + j) * kvs + kh * D;
            float s = 0.0f;
            for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
            sc[j] = s * scale;
            mx = sycl::fmax(mx, sc[j]);
        }
        float acc[128];
        for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
        float sum = 0.0f;
        for (int64_t j = 0; j < S && j < 64; ++j) {
            const float p = sycl::exp(sc[j] - mx);
            sum += p;
            const float* vj = v + (b * S + j) * kvs + kh * D;
            for (int64_t d = 0; d < D; ++d) acc[d] += p * vj[d];
        }
        float* o = out + r * (Hq * D) + h * D;
        for (int64_t d = 0; d < D; ++d) o[d] = acc[d] / sum;
    });
    return 0;
    NS_CATCH
}

int ns_audio_vcp_snake(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, float* out) {
    NS_TRY
    if (C > 0 && L > 0) g->q.parallel_for(sycl::range<1>((size_t) (C * L)), [=](sycl::id<1> id) {
        const float a = alpha[id[0] / L], v = x[id[0]], s = sycl::sin(a * v);
        out[id[0]] = v + s * s / (a + 1e-9f);
    });
    return 0;
    NS_CATCH
}

int ns_audio_vcp_dwconv(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* bias, int64_t K, int64_t dil, float* out) {
    NS_TRY
    if (C > 0 && L > 0) g->q.parallel_for(sycl::range<1>((size_t) (C * L)), [=](sycl::id<1> id) {
        const int64_t c = id[0] / L, t = id[0] % L;
        float a = bias ? bias[c] : 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            const int64_t s = t - (K - 1 - k) * dil;
            if (s >= 0) a += w[c * K + k] * x[c * L + s];
        }
        out[id[0]] = a;
    });
    return 0;
    NS_CATCH
}

int ns_audio_vcp_affine(ns_gpu* g, float* x, int64_t C, int64_t L, const float* scale, const float* bias) {
    NS_TRY
    if (C > 0 && L > 0) g->q.parallel_for(sycl::range<1>((size_t) (C * L)), [=](sycl::id<1> id) {
        const int64_t c = id[0] / L;
        x[id[0]] = x[id[0]] * scale[c] + bias[c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_vcp_fsq(ns_gpu* g, float* x, int64_t n, float s) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] = sycl::rint(sycl::tanh(x[i]) * s) / s; });
    return 0;
    NS_CATCH
}

}  // extern "C"
