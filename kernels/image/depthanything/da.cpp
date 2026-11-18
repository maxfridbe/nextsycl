// da.cpp: Depth Anything V2's own kernels (da.h): the embeddings, LayerNorm, bias + GELU / ReLU, the LayerScale residual,
// self attention, the transposed convolution's shuffle and the half-pixel bilinear resize. Elementwise or one row an
// item, on the GPU's queue, in float32.
#include "ns.h"
#include "ns_internal.hpp"
#include "da.h"

#include <sycl/sycl.hpp>

// precise arithmetic whatever the library's flags: no contracted multiply-adds, exp and erf as torch's CPU computes them
#pragma float_control(precise, on)
#pragma clang fp contract(off)

namespace {
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
}  // namespace

extern "C" {

int ns_image_da_embed(ns_gpu* g, const float* patches, const float* bias, const float* cls, const float* pos, int64_t N, int64_t C, float* out) {
    NS_TRY
    const int64_t n = (N + 1) * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % C, r = id / C;
        out[id] = (r == 0 ? cls[c] : patches[(r - 1) * C + c] + bias[c]) + pos[id];
    });
    return 0;
    NS_CATCH
}

int ns_image_da_layernorm(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, const float* b, float eps, float* out) {
    NS_TRY
    if (M > 0) g->q.parallel_for(sycl::range<1>(sz(M)), [=](sycl::id<1> idx) {
        const float* xr = x + (int64_t) idx[0] * C;
        float* o = out + (int64_t) idx[0] * C;
        float s = 0.0f;
        for (int64_t c = 0; c < C; ++c) s += xr[c];
        const float mean = s / (float) C;
        float v = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            const float d = xr[c] - mean;
            v += d * d;
        }
        const float inv = 1.0f / sycl::sqrt(v / (float) C + eps);
        for (int64_t c = 0; c < C; ++c) o[c] = (xr[c] - mean) * inv * w[c] + b[c];
    });
    return 0;
    NS_CATCH
}

static inline float act_of(float v, int act) {
    if (act == 1) return sycl::fmax(v, 0.0f);
    if (act == 2) return 0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752440f));
    return v;
}

int ns_image_da_bias_act(ns_gpu* g, float* x, int64_t M, int64_t C, const float* bias, int act) {
    NS_TRY
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        x[id] = act_of(x[id] + (bias ? bias[id % C] : 0.0f), act);
    });
    return 0;
    NS_CATCH
}

int ns_image_da_act(ns_gpu* g, const float* x, int64_t n, int act, float* out) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { out[i] = act_of(x[i], act); });
    return 0;
    NS_CATCH
}

int ns_image_da_ls_add(ns_gpu* g, float* x, const float* y, const float* bias, const float* ls, int64_t M, int64_t C) {
    NS_TRY
    const int64_t n = M * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0], c = id % C;
        x[id] = (y[id] + (bias ? bias[c] : 0.0f)) * ls[c] + x[id];
    });
    return 0;
    NS_CATCH
}

int ns_image_da_attention(ns_gpu* g, const float* qkv, int64_t N, int64_t H, int64_t D, float* out) {
    NS_TRY
    if (D < 1 || D > 128) return ns_fail("depth anything attention: head size must be 1..128");
    const int64_t HD = H * D, ld = 3 * HD;
    const float scale = 1.0f / sycl::sqrt((float) D);
    if (N > 0 && H > 0) g->q.parallel_for(sycl::range<1>(sz(N * H)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0], h = id % H, i = id / H;
        float q[128], acc[128];
        for (int64_t d = 0; d < D; ++d) { q[d] = qkv[i * ld + h * D + d]; acc[d] = 0.0f; }
        float m = -3.0e38f, l = 0.0f;
        for (int64_t j = 0; j < N; ++j) {
            const float* k = qkv + j * ld + HD + h * D;
            float s = 0.0f;
            for (int64_t d = 0; d < D; ++d) s += q[d] * k[d];
            s *= scale;
            const float mn = sycl::fmax(m, s);
            const float a = sycl::exp(m - mn), p = sycl::exp(s - mn);
            l = l * a + p;
            const float* v = qkv + j * ld + 2 * HD + h * D;
            for (int64_t d = 0; d < D; ++d) acc[d] = acc[d] * a + p * v[d];
            m = mn;
        }
        for (int64_t d = 0; d < D; ++d) out[i * HD + h * D + d] = acc[d] / l;
    });
    return 0;
    NS_CATCH
}

int ns_image_da_shuffle(ns_gpu* g, const float* y, int64_t h, int64_t w, int64_t k, int64_t Co, const float* bias, float* out) {
    NS_TRY
    const int64_t Wo = w * k, n = h * k * Wo * Co;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t co = id % Co, p = id / Co, xo = p % Wo, yo = p / Wo;
        const int64_t py = yo / k, ky = yo % k, px = xo / k, kx = xo % k;
        out[id] = y[(py * w + px) * (k * k * Co) + (ky * k + kx) * Co + co] + (bias ? bias[co] : 0.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_da_resize(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t os) {
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

}  // extern "C"
