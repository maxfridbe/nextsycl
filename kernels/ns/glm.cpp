// glm.cpp: the bring-up kernels - exact and plain, float32 activations, one kernel per step of docs/glm5next.md.
// Correctness first: every one is checked against llama.cpp's dumps; the imported optimized kernels replace them
// where speed matters, checked against these.
#include "ns.h"
#include "ns_internal.hpp"

#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <cmath>
#include <cstdlib>
#include <string>

namespace {

inline float h2f(ggml_half h) { return (float) h; }

// ---- dequantization: one work-item per block, the order and arithmetic of ggml-quants.c's dequantize_row_*

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void deq_q8_0(const block_q8_0* x, float* y) {
    const float d = h2f(x->d);
    for (int j = 0; j < QK8_0; ++j) y[j] = x->qs[j] * d;
}

void deq_q2_K(const block_q2_K* x, float* y) {
    const float d = (float) x->dm[0], mn = (float) x->dm[1];   // the SYCL ggml-common.h's (d, dmin) pair
    const uint8_t* q = x->qs;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = x->scales[is++];
            float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l] >> shift) & 3)) - ml;
            sc = x->scales[is++];
            dl = d * (sc & 0xF); ml = mn * (sc >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3)) - ml;
            shift += 2;
        }
        q += 32;
    }
}

void deq_q3_K(const block_q3_K* x, float* y) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*) aux;
    const float d_all = h2f(x->d);
    const uint8_t* q = x->qs;
    const uint8_t* hm = x->hmask;
    uint8_t m = 1;
    for (int i = 0; i < 3; ++i) {
        aux[i] = (uint32_t) x->scales[4 * i] | ((uint32_t) x->scales[4 * i + 1] << 8) | ((uint32_t) x->scales[4 * i + 2] << 16) |
                 ((uint32_t) x->scales[4 * i + 3] << 24);
    }
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

void deq_q4_K(const block_q4_K* x, float* y) {
    const uint8_t* q = x->qs;
    const float d = (float) x->dm[0], mn = (float) x->dm[1];   // the SYCL ggml-common.h's (d, dmin) pair
    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, x->scales, &sc, &m);
        const float d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m);
        const float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

void deq_q5_K(const block_q5_K* x, float* y) {
    const uint8_t* ql = x->qs;
    const uint8_t* qh = x->qh;
    const float d = (float) x->dm[0], mn = (float) x->dm[1];   // the SYCL ggml-common.h's (d, dmin) pair
    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, x->scales, &sc, &m);
        const float d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m);
        const float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

void deq_q6_K(const block_q6_K* x, float* y) {
    const float d = h2f(x->d);
    const uint8_t* ql = x->ql;
    const uint8_t* qh = x->qh;
    const int8_t* sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

void deq_iq2_xxs(const block_iq2_xxs* x, float* y) {
    const float d = h2f(x->d);
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        const uint16_t* q2 = x->qs + 4 * ib32;
        const uint32_t a0 = (uint32_t) q2[0] | ((uint32_t) q2[1] << 16);
        const uint32_t a1 = (uint32_t) q2[2] | ((uint32_t) q2[3] << 16);
        const float db = d * (0.5f + (a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t idx = (uint8_t) (a0 >> (8 * l));
            const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + idx);
            const uint8_t signs = ksigns_iq2xs[(a1 >> 7 * l) & 127];
            for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}

template <class Block, int QK, void (*F)(const Block*, float*)>
void dequant_blocks(sycl::queue& q, const void* src, size_t n, float* dst) {
    const size_t nb = n / QK;
    const Block* b = (const Block*) src;
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> i) { F(b + i[0], dst + i[0] * QK); });
}

}  // namespace

extern "C" {

int ns_dequant(ns_gpu* g, int type, const void* src, size_t n, float* dst) {
    NS_TRY
    auto& q = g->q;
    switch (type) {
        case 0:  // F32
            q.memcpy(dst, src, n * 4);
            return 0;
        case 1: {  // F16
            const sycl::half* s = (const sycl::half*) src;
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { dst[i] = (float) s[i]; });
            return 0;
        }
        case 30: {  // BF16: the top half of a float32
            const uint16_t* s = (const uint16_t*) src;
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { dst[i] = sycl::bit_cast<float>((uint32_t) s[i] << 16); });
            return 0;
        }
        case 8: dequant_blocks<block_q8_0, QK8_0, deq_q8_0>(q, src, n, dst); return 0;
        case 10: dequant_blocks<block_q2_K, QK_K, deq_q2_K>(q, src, n, dst); return 0;
        case 11: dequant_blocks<block_q3_K, QK_K, deq_q3_K>(q, src, n, dst); return 0;
        case 12: dequant_blocks<block_q4_K, QK_K, deq_q4_K>(q, src, n, dst); return 0;
        case 13: dequant_blocks<block_q5_K, QK_K, deq_q5_K>(q, src, n, dst); return 0;
        case 14: dequant_blocks<block_q6_K, QK_K, deq_q6_K>(q, src, n, dst); return 0;
        case 16: dequant_blocks<block_iq2_xxs, QK_K, deq_iq2_xxs>(q, src, n, dst); return 0;
        default: return ns_fail("ns_dequant: ggml type " + std::to_string(type) + " is not supported yet");
    }
    NS_CATCH
}

// x float32 -> fp16 (rows of the prompt path's half GEMMs)
int ns_to_f16(ns_gpu* g, const float* x, uint16_t* y, int64_t n) {
    NS_TRY
    sycl::half* h = (sycl::half*) y;
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { h[i] = (sycl::half) x[i]; });
    return 0;
    NS_CATCH
}

// y [T, N] (+)= x [T, K] . w [N, K]^T with fp16 x and w, float32 y and accumulation (oneMKL: the XMX units)
int ns_gemm_f16(ns_gpu* g, int64_t T, int64_t N, int64_t K, const uint16_t* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                int accumulate) {
    NS_TRY
    using oneapi::mkl::transpose;
    oneapi::mkl::blas::row_major::gemm(g->q, transpose::nontrans, transpose::trans, T, N, K, 1.0f, (const sycl::half*) x, ldx,
                                       (const sycl::half*) w, K, accumulate ? 1.0f : 0.0f, y, ldy);
    return 0;
    NS_CATCH
}

// y [T, N] (+)= x [T, K] . w [N, K]^T, float32 row-major; rows of x are ldx apart, rows of y ldy apart (column
// slices: one head's columns of a wider activation, one row chunk of a matrix into its columns of the output)
int ns_gemm(ns_gpu* g, int64_t T, int64_t N, int64_t K, const float* x, int64_t ldx, const float* w, float* y, int64_t ldy,
            int accumulate) {
    NS_TRY
    using oneapi::mkl::transpose;
    oneapi::mkl::blas::row_major::gemm(g->q, transpose::nontrans, transpose::trans, T, N, K, 1.0f, x, ldx, w, K,
                                       accumulate ? 1.0f : 0.0f, y, ldy);
    return 0;
    NS_CATCH
}

// batch independent products: y_b [T, N] (+)= x_b [T, K] . w_b [N, K]^T, x_b = x + b * sx (rows ldx apart),
// w_b = w + b * sw, y_b = y + b * sy (rows ldy apart) - one call for MLA's 64 heads
int ns_gemm_batch(ns_gpu* g, int64_t batch, int64_t T, int64_t N, int64_t K, const float* x, int64_t ldx, int64_t sx, const float* w, int64_t sw,
                  float* y, int64_t ldy, int64_t sy, int accumulate) {
    NS_TRY
    using oneapi::mkl::transpose;
    oneapi::mkl::blas::row_major::gemm_batch(g->q, transpose::nontrans, transpose::trans, T, N, K, 1.0f, x, ldx, sx, w, K, sw,
                                             accumulate ? 1.0f : 0.0f, y, ldy, sy, batch);
    return 0;
    NS_CATCH
}

// batch products without the transpose: y + b*sy [T, N] (+)= (x + b*sx) [T, K] . (w + b*sw) [K, N]
int ns_gemm_batch_nn(ns_gpu* g, int64_t batch, int64_t T, int64_t N, int64_t K, const float* x, int64_t ldx, int64_t sx, const float* w,
                     int64_t ldw, int64_t sw, float* y, int64_t ldy, int64_t sy, int accumulate) {
    NS_TRY
    using oneapi::mkl::transpose;
    oneapi::mkl::blas::row_major::gemm_batch(g->q, transpose::nontrans, transpose::nontrans, T, N, K, 1.0f, x, ldx, sx, w, ldw, sw,
                                             accumulate ? 1.0f : 0.0f, y, ldy, sy, batch);
    return 0;
    NS_CATCH
}

// batch products in fp16 on the XMX units, float32 out: y_b [T, N] (+)= x_b [T, K] . op(w_b), op(w) = w^T
// ([N, K], rows ldw apart) with trans_w, else w ([K, N]); x_b = x + b * sx, and so on (strides in elements)
int ns_gemm_batch_h(ns_gpu* g, int64_t batch, int trans_w, int64_t T, int64_t N, int64_t K, const uint16_t* x, int64_t ldx, int64_t sx,
                    const uint16_t* w, int64_t ldw, int64_t sw, float* y, int64_t ldy, int64_t sy, int accumulate) {
    NS_TRY
    using oneapi::mkl::transpose;
    oneapi::mkl::blas::row_major::gemm_batch(g->q, transpose::nontrans, trans_w ? transpose::trans : transpose::nontrans, T, N, K, 1.0f,
                                             (const sycl::half*) x, ldx, sx, (const sycl::half*) w, ldw, sw, accumulate ? 1.0f : 0.0f, y,
                                             ldy, sy, batch);
    return 0;
    NS_CATCH
}

// mHC's mixes: m [T, 24] = fn [24, n] . rms(x [T, n]) (no weight). Two passes so one token still spreads over
// the GPU: NB work-groups per token reduce a column block each to 24 partial dots and a partial sum of squares;
// then one work-item per (token, mix) adds the NB partials. (One work-group per token ran the whole 1.5 MB
// reduction on one core: 4.6x slower than oneMKL.)
int ns_hc_mix(ns_gpu* g, const float* x, const float* fn, float* m, float* part, int64_t T, int64_t n, float eps) {
    NS_TRY
    constexpr int WG = 256, M = 24, NB = 32;
    auto& q = g->q;
    if (T > 8) {
        // a prompt chunk: the 24 products as one oneMKL GEMM (the kernel below reads all of fn again for every
        // token: 6 GB a call at 4,096), then each row scaled by its rms. Decode widths keep the kernel below, whose
        // rows do not depend on T (a verify pass's rows equal one-token passes).
        using oneapi::mkl::transpose;
        oneapi::mkl::blas::row_major::gemm(q, transpose::nontrans, transpose::trans, T, M, n, 1.0f, x, n, fn, n, 0.0f, m, M);
        q.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t t = it.get_group(0);
            const float* xr = x + t * n;
            float ss = 0.f;
            for (int64_t c = it.get_local_id(0); c < n; c += WG) ss += xr[c] * xr[c];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const int lid = it.get_local_id(0);
            if (lid < M) m[t * M + lid] /= sycl::sqrt(ss / (float) n + eps);
        });
        return 0;
    }
    const int64_t chunk = (n + NB - 1) / NB;
    q.parallel_for(sycl::nd_range<1>(T * NB * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t grp = it.get_group(0), t = grp / NB, b = grp % NB;
        const float* xr = x + t * n;
        const int lid = it.get_local_id(0);
        const int64_t c0 = b * chunk, c1 = sycl::min(n, c0 + chunk);
        float ss = 0.f, acc[M];
        for (int o = 0; o < M; ++o) acc[o] = 0.f;
        for (int64_t c = c0 + lid; c < c1; c += WG) {
            const float v = xr[c];
            ss += v * v;
            for (int o = 0; o < M; ++o) acc[o] += fn[o * n + c] * v;
        }
        float* pr = part + grp * (M + 1);
        for (int o = 0; o < M; ++o) {
            const float r = sycl::reduce_over_group(it.get_group(), acc[o], sycl::plus<float>());
            if (lid == 0) pr[o] = r;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        if (lid == 0) pr[M] = ss;
    });
    q.parallel_for(sycl::range<1>(T * M), [=](sycl::id<1> id) {
        const int64_t t = id[0] / M, o = id[0] % M;
        float d = 0.f, ss = 0.f;
        for (int b = 0; b < NB; ++b) {
            d += part[(t * NB + b) * (M + 1) + o];
            ss += part[(t * NB + b) * (M + 1) + M];
        }
        m[t * M + o] = d / sycl::sqrt(ss / (float) n + eps);
    });
    return 0;
    NS_CATCH
}

// y = x / sqrt(mean(x^2) + eps) (* w when given), rows of C; y may be x
int ns_rms_norm(ns_gpu* g, const float* x, const float* w, float* y, int64_t rows, int64_t C, float eps) {
    NS_TRY
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const float* xr = x + r * C;
        float s = 0.f;
        for (int64_t c = it.get_local_id(0); c < C; c += WG) s += xr[c] * xr[c];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = 1.f / sycl::sqrt(s / (float) C + eps);
        for (int64_t c = it.get_local_id(0); c < C; c += WG) y[r * C + c] = xr[c] * inv * (w ? w[c] : 1.f);
    });
    return 0;
    NS_CATCH
}

// y = (x - mean) / sqrt(var + eps) * w + b, rows of C
int ns_layer_norm(ns_gpu* g, const float* x, const float* w, const float* b, float* y, int64_t rows, int64_t C, float eps) {
    NS_TRY
    constexpr int WG = 128;
    g->q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const float* xr = x + r * C;
        float s = 0.f;
        for (int64_t c = it.get_local_id(0); c < C; c += WG) s += xr[c];
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float v = 0.f;
        for (int64_t c = it.get_local_id(0); c < C; c += WG) v += (xr[c] - mean) * (xr[c] - mean);
        const float var = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / (float) C;
        const float inv = 1.f / sycl::sqrt(var + eps);
        for (int64_t c = it.get_local_id(0); c < C; c += WG) y[r * C + c] = (xr[c] - mean) * inv * (w ? w[c] : 1.f) + (b ? b[c] : 0.f);
    });
    return 0;
    NS_CATCH
}

// mHC, before a half: from the mixes m [T, 24] (fn . rms(flatten X)) and X [T, 4, C]:
//   h [T, C] = sum_s pre[s] X[s];  post [T, 4];  comb [T, 16] (comb[dst + 4 src]), Sinkhorn-normalized
int ns_hc_pre(ns_gpu* g, const float* m, const float* scale, const float* base, const float* X, float* h, float* post, float* comb,
              float* pre, int64_t T, int64_t C, float eps, int iters) {
    NS_TRY
    auto& q = g->q;
    q.parallel_for(sycl::range<1>(T), [=](sycl::id<1> id) {
        const int64_t t = id[0];
        const float* mt = m + t * 24;
        for (int i = 0; i < 4; ++i) {
            pre[t * 4 + i] = 1.f / (1.f + sycl::exp(-(mt[i] * scale[0] + base[i]))) + eps;
            post[t * 4 + i] = 2.f / (1.f + sycl::exp(-(mt[4 + i] * scale[1] + base[4 + i])));
        }
        float c[16];
        for (int i = 0; i < 16; ++i) c[i] = mt[8 + i] * scale[2] + base[8 + i];
        // softmax over dst for each src (ggml_soft_max over ne0 = dst), then + eps
        for (int src = 0; src < 4; ++src) {
            float mx = c[4 * src];
            for (int d = 1; d < 4; ++d) mx = sycl::fmax(mx, c[d + 4 * src]);
            float sum = 0.f;
            for (int d = 0; d < 4; ++d) { c[d + 4 * src] = sycl::exp(c[d + 4 * src] - mx); sum += c[d + 4 * src]; }
            for (int d = 0; d < 4; ++d) c[d + 4 * src] = c[d + 4 * src] / sum + eps;
        }
        auto norm_cols = [&]() {   // for each dst: divide by (eps + sum over src)
            for (int d = 0; d < 4; ++d) {
                float s = eps;
                for (int src = 0; src < 4; ++src) s += c[d + 4 * src];
                for (int src = 0; src < 4; ++src) c[d + 4 * src] /= s;
            }
        };
        auto norm_rows = [&]() {   // for each src: divide by (eps + sum over dst)
            for (int src = 0; src < 4; ++src) {
                float s = eps;
                for (int d = 0; d < 4; ++d) s += c[d + 4 * src];
                for (int d = 0; d < 4; ++d) c[d + 4 * src] /= s;
            }
        };
        norm_cols();
        for (int i = 1; i < iters; ++i) { norm_rows(); norm_cols(); }
        for (int i = 0; i < 16; ++i) comb[t * 16 + i] = c[i];
    });
    q.parallel_for(sycl::range<2>(T, C), [=](sycl::id<2> id) {
        const int64_t t = id[0], k = id[1];
        float s = 0.f;
        for (int i = 0; i < 4; ++i) s += pre[t * 4 + i] * X[(t * 4 + i) * C + k];
        h[t * C + k] = s;
    });
    return 0;
    NS_CATCH
}

// mHC, after a half: Xo[t][d] = post[t][d] * y[t] + sum_s comb[t][d + 4 s] * X[t][s]   (Xo may not be X)
int ns_hc_post(ns_gpu* g, const float* y, const float* X, const float* post, const float* comb, float* Xo, int64_t T, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<3>(T, 4, C), [=](sycl::id<3> id) {
        const int64_t t = id[0], d = id[1], k = id[2];
        float s = post[t * 4 + d] * y[t * C + k];
        for (int src = 0; src < 4; ++src) s += comb[t * 16 + d + 4 * src] * X[(t * 4 + src) * C + k];
        Xo[(t * 4 + d) * C + k] = s;
    });
    return 0;
    NS_CATCH
}

// y [T, C] = mean over the 4 streams of X [T, 4, C]
int ns_hc_mean(ns_gpu* g, const float* X, float* y, int64_t T, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, C), [=](sycl::id<2> id) {
        const int64_t t = id[0], k = id[1];
        float s = 0.f;
        for (int i = 0; i < 4; ++i) s += X[(t * 4 + i) * C + k];
        y[t * C + k] = s * 0.25f;
    });
    return 0;
    NS_CATCH
}

// KDA's short convolution, then silu: x [T, D] (this ubatch's projections), state [k-1, D] (the previous tokens'
// projections, oldest first; updated in place to the last k-1 inputs), w [D, k]; out [T, D]
// causal depthwise conv over T rows + the k-1 earlier inputs in `state`, then SiLU; state becomes the last k-1
// inputs. snap (nullable): [T-1][k-1][D], the state as it is after each row but the last (speculative rollback)
int ns_conv_silu(ns_gpu* g, const float* x, float* state, const float* w, float* out, int64_t T, int64_t D, int k, float* snap) {
    NS_TRY
    auto& q = g->q;
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
        const int64_t c = id[0];
        // input src (relative to row 0): x for src >= 0, the earlier inputs in state before
        auto in = [&](int64_t src) { return src >= 0 ? x[src * D + c] : state[(k - 1 + src) * D + c]; };
        for (int64_t t = 0; t < T; ++t) {
            float s = 0.f;
            for (int j = 0; j < k; ++j) s += in(t - (k - 1) + j) * w[c * k + j];   // inputs t-k+1 .. t
            out[t * D + c] = s / (1.f + sycl::exp(-s));
        }
        if (snap)
            for (int64_t r = 0; r + 1 < T; ++r)
                for (int j = 0; j < k - 1; ++j) snap[(r * (k - 1) + j) * D + c] = in(r + 1 - (k - 1) + j);
        // the new state: the last k-1 inputs
        float last[8];
        for (int j = 0; j < k - 1; ++j) last[j] = in(T - (k - 1) + j);
        for (int j = 0; j < k - 1; ++j) state[j * D + c] = last[j];
    });
    return 0;
    NS_CATCH
}

// per row of n: x / sqrt(sum x^2 + eps), in place (KDA's q and k per head)
int ns_l2_norm(ns_gpu* g, float* x, int64_t rows, int64_t n, float eps) {
    NS_TRY
    constexpr int WG = 128;
    g->q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        float* xr = x + it.get_group(0) * n;
        float s = 0.f;
        for (int64_t c = it.get_local_id(0); c < n; c += WG) s += xr[c] * xr[c];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = 1.f / sycl::sqrt(s + eps);
        for (int64_t c = it.get_local_id(0); c < n; c += WG) xr[c] *= inv;
    });
    return 0;
    NS_CATCH
}

// KDA decay gate, in place over g [T, H, dh]: g = low * sigmoid(-((g + dt_bias) * A[h]))
int ns_kda_gate(ns_gpu* g, float* gate, const float* dt_bias, const float* A, int64_t T, int64_t H, int64_t dh, float low) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>(T * H * dh), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % (H * dh), h = c / dh;
        const float v = (gate[i] + dt_bias[c]) * A[h];
        gate[i] = low / (1.f + sycl::exp(v));
    });
    return 0;
    NS_CATCH
}

// in place: x = exp(x)
int ns_exp(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] = sycl::exp(x[i]); });
    return 0;
    NS_CATCH
}

// in place: x = sigmoid(x)
int ns_sigmoid(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] = 1.f / (1.f + sycl::exp(-x[i])); });
    return 0;
    NS_CATCH
}

// The KDA recurrence (the fused op of docs/glm5next.md), tokens in order. q, k, v [T, H, d]; eg [T, H, d] the decay
// factors exp(g); beta [T, H]; S [H, d key, d value], updated; o [T, H, d]. The value columns of a head are
// independent: a sub-group of 16 per column, each lane 8 of its 128 keys (registers), the two sums over the keys
// as sub-group reductions - 8,192 sub-groups at work instead of 64 work-groups, no barriers. Decode and verify
// passes run it too, so their rows stay equal.
int ns_kda_scan(ns_gpu* g, const float* qv, const float* kv, const float* vv, const float* eg, const float* beta, float* S, float* o,
                int64_t T, int64_t H, int64_t d, float* snap) {
    NS_TRY
    if (d != 128) return ns_fail("ns_kda_scan: head size 128 only");
    constexpr int SG = 16, KPL = 128 / SG, COLS = 8;   // lanes a column, keys a lane, columns a work-group
    g->q.parallel_for(sycl::nd_range<1>(H * 128 * SG, COLS * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t c = it.get_global_id(0) / SG;   // the column: head hd, value j
        const int64_t hd = c / 128;
        const int j = (int) (c % 128), lane = (int) sg.get_local_id()[0], i0 = lane * KPL;
        float* Sh = S + hd * 128 * 128;
        float col[KPL];
#pragma unroll
        for (int i = 0; i < KPL; ++i) col[i] = Sh[(i0 + i) * 128 + j];
        const float scale = 1.f / sycl::sqrt(128.f);
        for (int64_t t = 0; t < T; ++t) {
            const int64_t base = (t * H + hd) * 128;
            const sycl::vec<float, KPL> kq = *reinterpret_cast<const sycl::vec<float, KPL>*>(kv + base + i0);
            const sycl::vec<float, KPL> gq = *reinterpret_cast<const sycl::vec<float, KPL>*>(eg + base + i0);
            const sycl::vec<float, KPL> qq = *reinterpret_cast<const sycl::vec<float, KPL>*>(qv + base + i0);
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < KPL; ++i) { col[i] *= gq[i]; dot += col[i] * kq[i]; }
            dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
            const float delta = (vv[base + j] - dot) * beta[t * H + hd];
            float out = 0.f;
#pragma unroll
            for (int i = 0; i < KPL; ++i) { col[i] += kq[i] * delta; out += col[i] * qq[i]; }
            out = sycl::reduce_over_group(sg, out, sycl::plus<float>());
            if (lane == 0) o[base + j] = out * scale;
            if (snap && t + 1 < T) {
                float* Sn = snap + (t * H + hd) * 128 * 128;
#pragma unroll
                for (int i = 0; i < KPL; ++i) Sn[(i0 + i) * 128 + j] = col[i];
            }
        }
#pragma unroll
        for (int i = 0; i < KPL; ++i) Sh[(i0 + i) * 128 + j] = col[i];
    });
    return 0;
    NS_CATCH
}

// KDA output: per head, y = rms_norm(o) * w * sigmoid(gate); o, gate [T, H, d] -> y [T, H, d]
int ns_kda_out(ns_gpu* g, const float* o, const float* gate, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps) {
    NS_TRY
    constexpr int WG = 128;
    g->q.parallel_for(sycl::nd_range<1>(T * H * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const float* orow = o + r * d;
        float s = 0.f;
        for (int64_t c = it.get_local_id(0); c < d; c += WG) s += orow[c] * orow[c];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = 1.f / sycl::sqrt(s / (float) d + eps);
        for (int64_t c = it.get_local_id(0); c < d; c += WG) {
            const float gt = gate[r * d + c];
            y[r * d + c] = orow[c] * inv * w[c] / (1.f + sycl::exp(-gt));
        }
    });
    return 0;
    NS_CATCH
}

// out [T, F] (fp16) = silu(min(g, limit)) * clamp(u, -limit, limit), gu [T, 2F] each row's gate then up (one
// GEMM's output for a gate | up pair)
int ns_swiglu_gu_f16(ns_gpu* g, const float* gu, uint16_t* out, int64_t T, int64_t F, float limit) {
    NS_TRY
    sycl::half* h = (sycl::half*) out;
    g->q.parallel_for(sycl::range<2>(T, F), [=](sycl::id<2> id) {
        const int64_t t = id[0], c = id[1];
        const float a = sycl::fmin(gu[t * 2 * F + c], limit);
        const float u = sycl::fmin(sycl::fmax(gu[t * 2 * F + F + c], -limit), limit);
        h[t * F + c] = (sycl::half) (a / (1.f + sycl::exp(-a)) * u);
    });
    return 0;
    NS_CATCH
}

// out = silu(min(gate, limit)) * clamp(up, -limit, limit); gate, up, out [n]
int ns_swiglu_clamp(ns_gpu* g, const float* gate, const float* up, float* out, int64_t n, float limit) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float a = sycl::fmin(gate[i], limit);
        const float u = sycl::fmin(sycl::fmax(up[i], -limit), limit);
        out[i] = a / (1.f + sycl::exp(-a)) * u;
    });
    return 0;
    NS_CATCH
}

// MLA attention over the cached latents, causal, every cached token (no indexer: exact up to ~2048 tokens).
// qa [T, H, L] (the absorbed queries of this ubatch, tokens pos0 .. pos0+T-1), c [n_cached, L] (all latents so far,
// including this ubatch's); u [T, H, L] = sum_t' softmax(qa . c(t') * scale) c(t'), t' <= pos
int ns_mla_attend(ns_gpu* g, const float* qa, const float* c, float* u, int64_t T, int64_t H, int64_t L, int64_t pos0, float scale) {
    NS_TRY
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(T * H * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t th = it.get_group(0), t = th / H;
        const float* qr = qa + th * L;
        const int64_t n = pos0 + t + 1;
        const int lid = it.get_local_id(0);
        // pass 1: the max score; pass 2: the sum and the weighted latents (two passes: exact, n is small here)
        float mx = -INFINITY;
        for (int64_t s = 0; s < n; ++s) {
            float dot = 0.f;
            for (int64_t r = lid; r < L; r += WG) dot += qr[r] * c[s * L + r];
            dot = sycl::reduce_over_group(it.get_group(), dot, sycl::plus<float>()) * scale;
            mx = sycl::fmax(mx, dot);
        }
        float sum = 0.f;
        float acc[2] = {0.f, 0.f};   // L = 512, WG = 256: two latent features a work-item
        for (int64_t s = 0; s < n; ++s) {
            float dot = 0.f;
            for (int64_t r = lid; r < L; r += WG) dot += qr[r] * c[s * L + r];
            dot = sycl::reduce_over_group(it.get_group(), dot, sycl::plus<float>()) * scale;
            const float p = sycl::exp(dot - mx);
            sum += p;
            for (int k = 0; k < 2 && lid + k * WG < L; ++k) acc[k] += p * c[s * L + lid + k * WG];
        }
        for (int k = 0; k < 2 && lid + k * WG < L; ++k) u[th * L + lid + k * WG] = acc[k] / sum;
    });
    return 0;
    NS_CATCH
}

// y[idx[i]] += w[i] * src[i]; rows of C (the experts' outputs back into their tokens); idx distinct per call
int ns_scatter_add(ns_gpu* g, float* y, const float* src, const int32_t* idx, const float* w, int64_t n, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(n, C), [=](sycl::id<2> id) {
        const int64_t i = id[0], k = id[1];
        y[idx[i] * C + k] += w[i] * src[i * C + k];
    });
    return 0;
    NS_CATCH
}

// out[i] = src[idx[i]] rows of C (the tokens an expert serves)
int ns_gather(ns_gpu* g, const float* src, const int32_t* idx, float* out, int64_t n, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(n, C), [=](sycl::id<2> id) { out[id[0] * C + id[1]] = src[idx[id[0]] * C + id[1]]; });
    return 0;
    NS_CATCH
}

// out[i] = (fp16) src[idx[i]] rows of C (an expert's tokens, as the half GEMMs take them)
int ns_gather_f16(ns_gpu* g, const float* src, const int32_t* idx, uint16_t* out, int64_t n, int64_t C) {
    NS_TRY
    sycl::half* h = (sycl::half*) out;
    g->q.parallel_for(sycl::range<2>(n, C), [=](sycl::id<2> id) { h[id[0] * C + id[1]] = (sycl::half) src[idx[id[0]] * C + id[1]]; });
    return 0;
    NS_CATCH
}

// out[i] = src[idx[i]], fp16 rows of C (the MLA cache's cells; 8 halves a work-item)
int ns_gather_h(ns_gpu* g, const uint16_t* src, const int32_t* idx, uint16_t* out, int64_t n, int64_t C) {
    NS_TRY
    if (C % 8 != 0) return ns_fail("ns_gather_h: row width not a multiple of 8");
    using V = sycl::vec<uint16_t, 8>;
    const int64_t cv = C / 8;
    const V* s = (const V*) src;
    V* o = (V*) out;
    g->q.parallel_for(sycl::range<2>(n, cv), [=](sycl::id<2> id) { o[id[0] * cv + id[1]] = s[idx[id[0]] * cv + id[1]]; });
    return 0;
    NS_CATCH
}

// The MoE combine, per token (no two entries write one row): y[t] += sum_j w[j] * rows[ent[j]] for j in
// [t_ptr[t], t_ptr[t+1]); rows [entries, C]
int ns_moe_combine(ns_gpu* g, float* y, const float* rows, const int32_t* t_ptr, const int32_t* ent, const float* w, int64_t T, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, C), [=](sycl::id<2> id) {
        const int64_t t = id[0], k = id[1];
        float s = 0.f;
        for (int32_t j = t_ptr[t]; j < t_ptr[t + 1]; ++j) s += w[j] * rows[(int64_t) ent[j] * C + k];
        y[t * C + k] += s;
    });
    return 0;
    NS_CATCH
}

// y += x (n values)
int ns_add(ns_gpu* g, float* y, const float* x, int64_t n) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { y[i] += x[i]; });
    return 0;
    NS_CATCH
}


// ---- DSA lightning indexer (docs/glm5next.md): pools of 4 consecutive tokens, a pooled key per completed pool ----

// The pools completed by tokens [pos0, pos0 + T): pooled[j][c] = sum_m softmax_m(ig_m[c] + ape[m][c]) * ik_m[c] over
// the 4 members m (positions 4j + m). A member before pos0 comes from `ring` ([4][2 * D]: ik | ig at slot pos % 4,
// the last tokens of the previous pass); then the ring takes this pass's last tokens.
int ns_idx_pool(ns_gpu* g, float* ring, const float* ik, const float* ig, const float* ape, float* pooled, int64_t pos0, int64_t T,
                int64_t D) {
    NS_TRY
    const int64_t j_lo = pos0 / 4, j_hi = (pos0 + T) / 4;
    auto& q = g->q;
    if (j_hi > j_lo) {
        q.parallel_for(sycl::range<2>(j_hi - j_lo, D), [=](sycl::id<2> id) {
            const int64_t j = j_lo + id[0], c = id[1];
            float lg[4], kv[4], mx = -INFINITY;
            for (int m = 0; m < 4; ++m) {
                const int64_t p = 4 * j + m;
                const float* k = p >= pos0 ? ik + (p - pos0) * D : ring + (p % 4) * 2 * D;
                const float* gg = p >= pos0 ? ig + (p - pos0) * D : ring + (p % 4) * 2 * D + D;
                kv[m] = k[c];
                lg[m] = gg[c] + ape[m * D + c];
                mx = sycl::fmax(mx, lg[m]);
            }
            float s = 0.f, acc = 0.f;
            for (int m = 0; m < 4; ++m) { const float e = sycl::exp(lg[m] - mx); s += e; acc += e * kv[m]; }
            pooled[j * D + c] = acc / s;
        });
    }
    const int64_t first = sycl::max<int64_t>(pos0, pos0 + T - 3);
    q.parallel_for(sycl::range<2>(pos0 + T - first, D), [=](sycl::id<2> id) {
        const int64_t p = first + id[0], c = id[1];
        ring[(p % 4) * 2 * D + c] = ik[(p - pos0) * D + c];
        ring[(p % 4) * 2 * D + D + c] = ig[(p - pos0) * D + c];
    });
    return 0;
    NS_CATCH
}

// Pool scores of rows [T] against pools [j0, j0 + n): S [T * H, n] = iq . pooled^T (a GEMM, heads as rows), w [T, H]:
// score[r][j0 + jj] = sum_h relu(S[r * H + h][jj]) * w[r][h] for the pools row r sees (j < (pos0 + r + 1) / 4), else
// -inf. score rows are ld floats apart.
int ns_idx_score(ns_gpu* g, const float* S, const float* w, float* score, int64_t T, int64_t H, int64_t j0, int64_t n, int64_t ld,
                 int64_t pos0) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, n), [=](sycl::id<2> id) {
        const int64_t r = id[0], jj = id[1], j = j0 + jj;
        const int64_t nv = (pos0 + r + 1) / 4;
        float s = -INFINITY;
        if (j < nv) {
            s = 0.f;
            for (int64_t h = 0; h < H; ++h) s += sycl::fmax(S[(r * H + h) * n + jj], 0.f) * w[r * H + h];
        }
        score[r * ld + j] = s;
    });
    return 0;
    NS_CATCH
}

// Per row r: the K highest of score[r][0, n) (rows ld apart) into sel[r][0, K), in ascending position (ties at the
// K-th value broken by position). A radix select over the float's order-preserving key, 8 bits at a time, then one compaction.
int ns_topk(ns_gpu* g, const float* score, int32_t* sel, int64_t T, int64_t n, int64_t ld, int64_t K) {
    NS_TRY
    constexpr int WG = 256;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(256), h);
        sycl::local_accessor<uint32_t, 1> st(sycl::range<1>(4), h);   // prefix, mask, need, ties taken
        h.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t r = it.get_group(0);
            const int lid = it.get_local_id(0);
            const float* row = score + r * ld;
            auto key = [](float f) {
                uint32_t u = sycl::bit_cast<uint32_t>(f);
                return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // larger float -> larger key
            };
            if (lid == 0) { st[0] = 0; st[1] = 0; st[2] = (uint32_t) K; st[3] = 0; }
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (int b = lid; b < 256; b += WG) hist[b] = 0;
                sycl::group_barrier(it.get_group());
                const uint32_t prefix = st[0], mask = st[1];
                for (int64_t j = lid; j < n; j += WG) {
                    const uint32_t k = key(row[j]);
                    if ((k & mask) == prefix) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space> a(hist[(k >> shift) & 255]);
                        a.fetch_add(1u);
                    }
                }
                sycl::group_barrier(it.get_group());
                if (lid == 0) {
                    // the digit, from the top, where the count of keys at or above it reaches `need`
                    uint32_t need = st[2], above = 0;
                    int d = 255;
                    for (; d > 0; --d) {
                        if (above + hist[d] >= need) break;
                        above += hist[d];
                    }
                    st[0] = prefix | ((uint32_t) d << shift);
                    st[1] = mask | (255u << shift);
                    st[2] = need - above;   // still needed among the keys equal to the prefix so far
                }
                sycl::group_barrier(it.get_group());
            }
            // st[0] is the K-th key; every key above it goes, then the first st[2] equal ones by position. A prefix
            // scan per tile places them in ascending position order: the same selection, in the same order, every
            // run (atomics would order it by timing, and the attention's sums with it)
            const uint32_t kth = st[0], ties = st[2];
            uint32_t out = 0, eq_seen = 0;
            for (int64_t j0 = 0; j0 < n; j0 += WG) {
                const int64_t j = j0 + lid;
                const uint32_t k = j < n ? key(row[j]) : 0u;
                const uint32_t eq = j < n && k == kth ? 1u : 0u;
                const uint32_t eq_before = sycl::exclusive_scan_over_group(it.get_group(), eq, sycl::plus<uint32_t>()) + eq_seen;
                const uint32_t take = (j < n && k > kth) || (eq && eq_before < ties) ? 1u : 0u;
                const uint32_t at = sycl::exclusive_scan_over_group(it.get_group(), take, sycl::plus<uint32_t>()) + out;
                if (take) sel[r * K + at] = (int32_t) j;
                out += sycl::reduce_over_group(it.get_group(), take, sycl::plus<uint32_t>());
                eq_seen += sycl::reduce_over_group(it.get_group(), eq, sycl::plus<uint32_t>());
            }
        });
    });
    return 0;
    NS_CATCH
}

// The prompt path's MLA as GEMMs (rows of a chunk): per row t, the cells it attends to (as ns_mla_attend_sel: every
// earlier one, or its selected pools' and its incomplete pool's) as indices idx [T][NC] (padding: 0) and their
// count n [T].
int ns_mla_cells(ns_gpu* g, const int32_t* sel, const int32_t* sel_cnt, int64_t T, int64_t K, int64_t pos0, int32_t* idx, int32_t* n,
                 int64_t NC) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, NC), [=](sycl::id<2> id) {
        const int64_t t = id[0], i = id[1];
        const int64_t p = pos0 + t;
        const int32_t ns = sel_cnt ? sel_cnt[t] : -1;
        const int64_t tail0 = 4 * ((p + 1) / 4);
        const int64_t cnt = ns < 0 ? p + 1 : 4 * (int64_t) ns + (p + 1 - tail0);
        int64_t c = 0;
        if (i < cnt) c = ns < 0 ? i : i < 4 * (int64_t) ns ? 4 * (int64_t) sel[t * K + i / 4] + i % 4 : tail0 + (i - 4 * (int64_t) ns);
        idx[t * NC + i] = (int32_t) c;
        if (i == 0) n[t] = (int32_t) sycl::min<int64_t>(cnt, NC);
    });
    return 0;
    NS_CATCH
}

// Row softmax of scores S [R * H rows of NC] (row r's heads valid over its first n[r] cells, scaled): the rest 0
int ns_softmax_masked(ns_gpu* g, float* S, int64_t R, int64_t H, int64_t NC, const int32_t* n, float scale) {
    NS_TRY
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(R * H * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t rh = it.get_group(0), r = rh / H;
        float* row = S + rh * NC;
        const int64_t m = n[r];
        const int lid = it.get_local_id(0);
        float mx = -INFINITY;
        for (int64_t i = lid; i < m; i += WG) mx = sycl::fmax(mx, row[i] * scale);
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sum = 0.f;
        for (int64_t i = lid; i < m; i += WG) sum += sycl::exp(row[i] * scale - mx);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        for (int64_t i = lid; i < NC; i += WG) row[i] = i < m ? sycl::exp(row[i] * scale - mx) / sum : 0.f;
    });
    return 0;
    NS_CATCH
}

// MLA over a row's selection: sel_cnt[t] < 0 - every earlier token (dense causal, as ns_mla_attend); else the tokens
// of pools sel[t][0, sel_cnt[t]) (4 each) and the row's incomplete pool (positions 4 * ((p + 1) / 4) .. p). A row
// reads at most K * 4 + 3 cells (the indexer switches on past K pools), so its scores live in local memory: every
// work-item scores whole cells, one softmax, then the weighted latents with the cells' rows read coalesced. A
// work-group takes HG heads of a row: each cell's latent is read once for all of them (prompt chunks: HG = 4; one
// row - decode - keeps HG = 1 for the parallelism). The per-head arithmetic is the same either way.
}  // extern "C"

template <int HG>
static void mla_sel(sycl::queue& q, const float* qa, const sycl::half* c, float* u, int64_t T, int64_t H, int64_t L, int64_t pos0, float scale,
                    const int32_t* sel, const int32_t* sel_cnt, int64_t K) {
    constexpr int WG = 256, MAXN = 2560, MAXL = 512;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(HG * MAXN), h), qs(sycl::range<1>(HG * MAXL), h);
        h.parallel_for(sycl::nd_range<1>(T * (H / HG) * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t grp = it.get_group(0), t = grp / (H / HG), h0 = (grp % (H / HG)) * HG;
            const int lid = it.get_local_id(0);
            const int64_t p = pos0 + t;
            const int32_t ns = sel_cnt ? sel_cnt[t] : -1;
            const int64_t tail0 = 4 * ((p + 1) / 4);
            const int64_t n = ns < 0 ? p + 1 : 4 * (int64_t) ns + (p + 1 - tail0);
            auto cell = [&](int64_t i) -> int64_t {
                if (ns < 0) return i;
                if (i < 4 * (int64_t) ns) return 4 * (int64_t) sel[t * K + i / 4] + i % 4;
                return tail0 + (i - 4 * (int64_t) ns);
            };
            for (int64_t r = lid; r < HG * L; r += WG) qs[r] = qa[(t * H + h0) * L + r];
            sycl::group_barrier(it.get_group());
            float mx[HG];
            for (int k = 0; k < HG; ++k) mx[k] = -INFINITY;
            for (int64_t i = lid; i < n; i += WG) {
                const sycl::vec<sycl::half, 4>* cs = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(c + cell(i) * L);
                float dot[HG] = {};
                for (int64_t r = 0; r < L / 4; ++r) {
                    const sycl::float4 v = cs[r].convert<float>();
                    for (int k = 0; k < HG; ++k)
                        dot[k] += qs[k * L + 4 * r] * v.x() + qs[k * L + 4 * r + 1] * v.y() + qs[k * L + 4 * r + 2] * v.z() + qs[k * L + 4 * r + 3] * v.w();
                }
                for (int k = 0; k < HG; ++k) {
                    dot[k] *= scale;
                    sc[k * MAXN + i] = dot[k];
                    mx[k] = sycl::fmax(mx[k], dot[k]);
                }
            }
            float sum[HG];
            for (int k = 0; k < HG; ++k) {
                mx[k] = sycl::reduce_over_group(it.get_group(), mx[k], sycl::maximum<float>());
                float sk = 0.f;
                for (int64_t i = lid; i < n; i += WG) {
                    const float e = sycl::exp(sc[k * MAXN + i] - mx[k]);
                    sc[k * MAXN + i] = e;
                    sk += e;
                }
                sum[k] = sycl::reduce_over_group(it.get_group(), sk, sycl::plus<float>());
            }
            sycl::group_barrier(it.get_group());
            float acc[HG][MAXL / WG] = {};
            for (int64_t i = 0; i < n; ++i) {
                const sycl::half* cs = c + cell(i) * L;
                float cv[MAXL / WG];
                for (int f = 0; f < L / WG; ++f) cv[f] = (float) cs[lid + f * WG];
                for (int k = 0; k < HG; ++k) {
                    const float w = sc[k * MAXN + i];
                    for (int f = 0; f < L / WG; ++f) acc[k][f] += w * cv[f];
                }
            }
            for (int k = 0; k < HG; ++k)
                for (int f = 0; f < L / WG; ++f) u[(t * H + h0 + k) * L + lid + f * WG] = acc[k][f] / sum[k];
        });
    });
}

extern "C" {

int ns_mla_attend_sel(ns_gpu* g, const float* qa, const uint16_t* c16, float* u, int64_t T, int64_t H, int64_t L, int64_t pos0, float scale,
                      const int32_t* sel, const int32_t* sel_cnt, int64_t K) {
    NS_TRY
    if (L > 512 || L % 256 != 0) return ns_fail("ns_mla_attend_sel: latent width 256 or 512");
    if (pos0 + T > 2560 && !sel_cnt) return ns_fail("ns_mla_attend_sel: a dense row past 2,560 tokens (the indexer selects there)");
    // NS_MLA_HG=4: 4 heads a work-group in prompt chunks (measured slower at 3K: 13.7 vs 12.1 s - occupancy)
    static const bool hg4 = getenv("NS_MLA_HG") && getenv("NS_MLA_HG")[0] == '4';
    const sycl::half* c = (const sycl::half*) c16;   // the cache's latents, fp16
    if (hg4 && T > 8 && H % 4 == 0) mla_sel<4>(g->q, qa, c, u, T, H, L, pos0, scale, sel, sel_cnt, K);
    else mla_sel<1>(g->q, qa, c, u, T, H, L, pos0, scale, sel, sel_cnt, K);
    return 0;
    NS_CATCH
}
}  // extern "C"
