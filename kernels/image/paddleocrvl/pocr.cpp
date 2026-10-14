// pocr.cpp: the PaddleOCR-VL engine's kernels (pocr.h) - the projector's 2 x 2 merge, the decoder's cache copies and
// its one-row attention over grouped key / value heads. The linears, norms, RoPE, GELU and the prompt's attention are
// the shared diffusion kernels' (nsd.h).
#include "ns.h"
#include "ns_internal.hpp"
#include "pocr.h"

#include <sycl/sycl.hpp>
#include <cmath>

using half = sycl::half;

extern "C" {

int ns_image_pocr_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_image_pocr_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols) {
    NS_TRY
    if (rows > 0 && cols > 0)
        g->q.parallel_for(sycl::range<2>((size_t) rows, (size_t) cols), [=](sycl::id<2> id) { dst[id[0] * ds + id[1]] = src[id[0] * ss + id[1]]; });
    return 0;
    NS_CATCH
}

int ns_image_pocr_merge2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out) {
    NS_TRY
    if (H % 2 || W % 2) return ns_fail("ns_image_pocr_merge2: even sides");
    const int64_t w2 = W / 2, n = H * W * C;
    if (n > 0)
        g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const int64_t i = id[0], c = i % (4 * C), r = i / (4 * C);
            const int64_t q = c / C, cc = c % C, y = (r / w2) * 2 + q / 2, xx = (r % w2) * 2 + q % 2;
            out[i] = x[(y * W + xx) * C + cc];
        });
    return 0;
    NS_CATCH
}

static constexpr int64_t CHUNK = 256;

int64_t ns_image_pocr_attn1_scratch(int64_t n, int64_t Hq, int64_t D) {
    const int64_t chunks = (n + CHUNK - 1) / CHUNK;
    return Hq * n + Hq * chunks * (D + 2);
}

int ns_image_pocr_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                        int64_t D, float* scratch, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    if (Hq % Hkv || D % 16 || D > 256 || Hq / Hkv > 16) return ns_fail("ns_image_pocr_attn1: Hq a multiple of Hkv (at most 16 a kv head), D a multiple of 16 up to 256");
    const int64_t rep = Hq / Hkv, chunks = (n + CHUNK - 1) / CHUNK;
    const float scale = 1.0f / std::sqrt((float) D);
    float* sc = scratch;                // [Hq][n]
    float* part = scratch + Hq * n;     // [Hq][chunks][D + 2]: the chunk's max, sum, weighted values
    // the scores: a sub-group of 16 a (kv head, key), each lane D / 16 features, every query head of the group at once
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (Hkv * n * 16)), sycl::range<1>(16)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t w = it.get_group(0), gh = w / n, j = w % n, l = it.get_local_id(0);
        const int64_t per = D / 16;
        const float* kj = k + j * kvs + gh * D + l * per;
        float acc[16];
        for (int64_t r = 0; r < rep; ++r) acc[r] = 0.0f;
        for (int64_t e = 0; e < per; ++e) {
            const float kv = kj[e];
            for (int64_t r = 0; r < rep; ++r) acc[r] += q[(gh * rep + r) * D + l * per + e] * kv;
        }
        auto sg = it.get_sub_group();
        for (int64_t r = 0; r < rep; ++r) {
            const float s = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
            if (l == 0) sc[(gh * rep + r) * n + j] = s * scale;
        }
    });
    // each (head, chunk): its max, its sum of exp and its exp-weighted values, a lane a feature
    const size_t wg = D <= 64 ? 64 : (D <= 128 ? 128 : 256);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (Hq * chunks) * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
        const int64_t w = it.get_group(0), h = w / chunks, c = w % chunks, l = it.get_local_id(0);
        const int64_t j0 = c * CHUNK, j1 = sycl::min(n, j0 + CHUNK);
        const float* s = sc + h * n;
        float m = -INFINITY;
        for (int64_t j = j0 + l; j < j1; j += wg) m = sycl::fmax(m, s[j]);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float sum = 0.0f;
        for (int64_t j = j0 + l; j < j1; j += wg) sum += sycl::exp(s[j] - m);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        float* p = part + (h * chunks + c) * (D + 2);
        if (l < D) {
            const float* vh = v + (h / rep) * D + l;
            float acc = 0.0f;
            for (int64_t j = j0; j < j1; ++j) acc += sycl::exp(s[j] - m) * vh[j * kvs];
            p[2 + l] = acc;
        }
        if (l == 0) {
            p[0] = m;
            p[1] = sum;
        }
    });
    // the chunks combined, a lane a feature
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) Hq * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
        const int64_t h = it.get_group(0), l = it.get_local_id(0);
        if (l >= D) return;
        const float* p = part + h * chunks * (D + 2);
        float m = -INFINITY;
        for (int64_t c = 0; c < chunks; ++c) m = sycl::fmax(m, p[c * (D + 2)]);
        float sum = 0.0f, acc = 0.0f;
        for (int64_t c = 0; c < chunks; ++c) {
            const float f = sycl::exp(p[c * (D + 2)] - m);
            sum += p[c * (D + 2) + 1] * f;
            acc += p[c * (D + 2) + 2 + l] * f;
        }
        out[h * D + l] = acc / sum;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
