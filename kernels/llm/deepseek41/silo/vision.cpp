// vision.cpp: the vision sidecar's small kernels (ds41.h) - the ViT's 2D rotation of q and k, a row softmax, a bias
// with GELU, SiLU(gate) x up. Its products go through oneMKL (float32); these are the steps between them.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>

extern "C" {

// The ViT's 2D RoPE (vision.py: get_vision_cos_sin + apply_rotary) on q and k of qkv [N][3][H][64]: patch n at row
// n / n_w, column n % n_w; angle j < 16 of a head the row times theta^(-2j/32), j >= 16 the column times
// theta^(-2(j-16)/32); pairs (j, j + 32). q rotated in place, k written rotated to kc [H][N][64] (a head's keys in a row
// for the score products).
int ns_ds41_vit_rope(ns_gpu* g, float* qkv, float* kc, int64_t N, int64_t H, int64_t n_w, float theta) {
    NS_TRY
    constexpr int HD = 64, HALF = 32;
    const int64_t C = H * HD;
    g->q.parallel_for(sycl::range<3>(N, H, HALF), [=](sycl::id<3> id) {
        const int64_t n = id[0], h = id[1];
        const int j = (int) id[2];
        const float pos = j < 16 ? (float) (n / n_w) : (float) (n % n_w);
        const int i = j % 16;
        const float inv = 1.0f / sycl::pow(theta, (float) (2 * i) / (float) HALF);
        const float f = pos * inv;
        const float c = sycl::cos(f), s = sycl::sin(f);
        float* q = qkv + n * 3 * C + h * HD;
        const float q1 = q[j], q2 = q[j + HALF];
        q[j] = q1 * c - q2 * s;
        q[j + HALF] = q2 * c + q1 * s;
        const float* k = qkv + n * 3 * C + C + h * HD;
        const float k1 = k[j], k2 = k[j + HALF];
        float* ko = kc + (h * N + n) * HD;
        ko[j] = k1 * c - k2 * s;
        ko[j + HALF] = k2 * c + k1 * s;
    });
    return 0;
    NS_CATCH
}

// s [rows][n] = softmax(s * scale) by row: a work-group a row
int ns_ds41_softmax_rows(ns_gpu* g, float* s, int64_t rows, int64_t n, float scale) {
    NS_TRY
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        float* r = s + it.get_group(0) * n;
        const int lid = (int) it.get_local_id(0);
        float mx = -INFINITY;
        for (int64_t c = lid; c < n; c += WG) mx = sycl::fmax(mx, r[c] * scale);
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sum = 0.f;
        for (int64_t c = lid; c < n; c += WG) {
            const float e = sycl::exp(r[c] * scale - mx);
            r[c] = e;
            sum += e;
        }
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.f / sum;
        for (int64_t c = lid; c < n; c += WG) r[c] *= inv;
    });
    return 0;
    NS_CATCH
}

// y [rows][c] += b (nullable); act 1: then GELU (erf, torch's default)
int ns_ds41_bias_act(ns_gpu* g, float* y, const float* b, int64_t rows, int64_t c, int act) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(rows, c), [=](sycl::id<2> id) {
        float v = y[id[0] * c + id[1]] + (b ? b[id[1]] : 0.f);
        if (act == 1) v = 0.5f * v * (1.f + sycl::erf(v * 0.70710678118654752f));
        y[id[0] * c + id[1]] = v;
    });
    return 0;
    NS_CATCH
}

// h [rows][f] = silu(gu[r][j]) * gu[r][f + j] (the ViT MLP's w1 output: gate, then up)
int ns_ds41_silu_mul(ns_gpu* g, const float* gu, float* h, int64_t rows, int64_t f) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(rows, f), [=](sycl::id<2> id) {
        const float a = gu[id[0] * 2 * f + id[1]], u = gu[id[0] * 2 * f + f + id[1]];
        h[id[0] * f + id[1]] = a / (1.f + sycl::exp(-a)) * u;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
