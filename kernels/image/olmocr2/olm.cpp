// olm.cpp: the olmOCR-2 engine's kernels (olm.h) - the vision tower's window attention, the decoder's Q8_0
// products for one row, its cache copies and one-row attention over grouped heads. The linears for many rows,
// norms, RoPE and the prompt's attention are the shared diffusion kernels' (nsd.h).
#include "ns.h"
#include "ns_internal.hpp"
#include "olm.h"

#include <sycl/sycl.hpp>
#include <cmath>

using half = sycl::half;

extern "C" {

int ns_image_olm_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_image_olm_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols) {
    NS_TRY
    if (rows > 0 && cols > 0)
        g->q.parallel_for(sycl::range<2>((size_t) rows, (size_t) cols), [=](sycl::id<2> id) { dst[id[0] * ds + id[1]] = src[id[0] * ss + id[1]]; });
    return 0;
    NS_CATCH
}

int ns_image_olm_gather(ns_gpu* g, const float* src, int64_t C, const int32_t* idx, int64_t n, float* out) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) C), [=](sycl::id<2> id) { out[id[0] * C + id[1]] = src[(int64_t) idx[id[0]] * C + id[1]]; });
    return 0;
    NS_CATCH
}

int ns_image_olm_seg_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t stride, int64_t N, int64_t H, int64_t D,
                          const int32_t* lo, const int32_t* hi, float* out) {
    NS_TRY
    if (N <= 0) return 0;
    if (D % 16 || D > 256) return ns_fail("ns_image_olm_seg_attn: D a multiple of 16, at most 256");
    const float scale = 1.0f / std::sqrt((float) D);
    // a sub-group of 16 a (token, head), each lane D / 16 features; online softmax over the segment's keys
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N * H * 16)), sycl::range<1>(16)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t w = it.get_group(0), i = w / H, h = w % H, l = it.get_local_id(0);
        const int64_t per = D / 16;
        auto sg = it.get_sub_group();
        float qd[16], acc[16];
        for (int64_t e = 0; e < per; ++e) {
            qd[e] = q[i * stride + h * D + l * per + e] * scale;
            acc[e] = 0.0f;
        }
        float m = -INFINITY, s = 0.0f;
        for (int64_t j = lo[i]; j < hi[i]; ++j) {
            const float* kj = k + j * stride + h * D + l * per;
            float p = 0.0f;
            for (int64_t e = 0; e < per; ++e) p += qd[e] * kj[e];
            p = sycl::reduce_over_group(sg, p, sycl::plus<float>());
            const float mn = sycl::fmax(m, p), a = sycl::exp(m - mn), b = sycl::exp(p - mn);
            const float* vj = v + j * stride + h * D + l * per;
            for (int64_t e = 0; e < per; ++e) acc[e] = acc[e] * a + b * vj[e];
            s = s * a + b;
            m = mn;
        }
        for (int64_t e = 0; e < per; ++e) out[i * H * D + h * D + l * per + e] = acc[e] / s;
    });
    return 0;
    NS_CATCH
}

int ns_image_olm_q8r_gemv(ns_gpu* g, const int8_t* qs, const void* d, const float* x, int64_t n, int64_t k, const float* bias, float* y) {
    NS_TRY
    if (n <= 0) return 0;
    if (k % 256) return ns_fail("ns_image_olm_q8r_gemv: k a multiple of 256");
    const int64_t rows = 4, groups = (n + rows - 1) / rows, nb = k / 32;
    const half* dh = (const half*) d;
    // a sub-group of 16 a row: each step the sub-group reads 256 values (16 bytes a lane, 8 blocks), coalesced
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * rows * 16)), sycl::range<1>((size_t) (rows * 16))),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t r = it.get_group(0) * rows + it.get_local_id(0) / 16, l = it.get_local_id(0) % 16;
        auto sg = it.get_sub_group();
        float acc = 0.0f;
        if (r < n) {
            const int8_t* row = qs + r * k;
            const half* rd = dh + r * nb;
            for (int64_t at = l * 16; at < k; at += 256) {
                const sycl::vec<int8_t, 16> q = *(const sycl::vec<int8_t, 16>*) (row + at);
                const sycl::float4* xv = (const sycl::float4*) (x + at);
                float s = 0.0f;
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const sycl::float4 xx = xv[v];
                    s += (float) q[4 * v] * xx.x() + (float) q[4 * v + 1] * xx.y() + (float) q[4 * v + 2] * xx.z() + (float) q[4 * v + 3] * xx.w();
                }
                acc += s * (float) rd[at / 32];
            }
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (r < n && l == 0) y[r] = acc + (bias ? bias[r] : 0.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_olm_q8r_half(ns_gpu* g, const int8_t* qs, const void* d, int64_t n, int64_t k, void* out) {
    NS_TRY
    const half* dh = (const half*) d;
    half* o = (half*) out;
    const int64_t total = n * k;
    if (total > 0)
        g->q.parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> id) {
            const int64_t i = id[0];
            o[i] = (half) ((float) qs[i] * (float) dh[i / 32]);
        });
    return 0;
    NS_CATCH
}

static constexpr int64_t CHUNK = 256;

int64_t ns_image_olm_attn1_scratch(int64_t n, int64_t Hq, int64_t D) {
    const int64_t chunks = (n + CHUNK - 1) / CHUNK;
    return Hq * n + Hq * chunks * (D + 2);
}

int ns_image_olm_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                        int64_t D, float* scratch, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    if (Hq % Hkv || D % 16 || D > 256 || Hq / Hkv > 16) return ns_fail("ns_image_olm_attn1: Hq a multiple of Hkv (at most 16 a kv head), D a multiple of 16 up to 256");
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
