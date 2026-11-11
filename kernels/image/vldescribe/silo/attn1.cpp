// attn1.cpp: the native describer's one-row attention (silo.h) - flash decoding over the key / value cache, so a written
// token's attention spreads over the card instead of one work-group a head walking every key (the shared
// nsd_attention_causal_rows: tg fell from 31 to 5 tok/s between 440 and 4,300 cached rows on Qwen3-VL-4B, B70).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
constexpr int64_t CHUNK = 256;   // keys a (head, chunk) work-group
}

extern "C" {

int ns_image_vld_silo_attn1_supported(int64_t Hq, int64_t Hkv, int64_t D) {
    return Hkv > 0 && Hq % Hkv == 0 && Hq / Hkv <= 16 && D % 16 == 0 && D > 0 && D <= 256;
}

int64_t ns_image_vld_silo_attn1_scratch(int64_t n, int64_t Hq, int64_t D) {
    const int64_t chunks = (n + CHUNK - 1) / CHUNK;
    return Hq * n + Hq * chunks * (D + 2);
}

int ns_image_vld_silo_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                            int64_t D, float* scratch, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    if (!ns_image_vld_silo_attn1_supported(Hq, Hkv, D)) return ns_fail("ns_image_vld_silo_attn1: Hq a multiple of Hkv (at most 16 a kv head), D a multiple of 16 up to 256");
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
