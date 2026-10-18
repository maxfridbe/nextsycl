// k41.cpp: DeepSeek-V4.1's own steps (ds41.h): the model's low-precision key simulations, the mHC collapse by the
// previous half's mix, the Engram's rows and gate, the indexer at ratio 1 and 2 with its candidate blocks.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>

namespace {

// 2^ceil(log2 x), x > 0 (kernel.py's fast_round_scale)
inline float pow2_ceil(float x) {
    int e;
    const float m = sycl::frexp(x, &e);
    if (m == 0.5f) e -= 1;
    return sycl::ldexp(1.0f, e);
}

// the nearest E4M3FN value (ties to even), |x| <= 448
inline float e4m3_round(float x) {
    const float a = sycl::fabs(x);
    if (a == 0.f) return x;
    int e;
    sycl::frexp(a, &e);   // a = m 2^e, m in [0.5, 1): the binade's exponent is e - 1
    int be = e - 1;
    if (be < -6) be = -6;   // the subnormals' step
    const float step = sycl::ldexp(1.0f, be - 3);
    float q = sycl::rint(a / step) * step;
    q = sycl::fmin(q, 448.f);
    return x < 0 ? -q : q;
}

// the nearest E2M1 value (ties to the even code), |y| <= 6
inline float e2m1_round(float y) {
    const float a = sycl::fabs(y);
    float q;
    if (a <= 0.25f) q = 0.f;
    else if (a < 0.75f) q = 0.5f;
    else if (a <= 1.25f) q = 1.f;
    else if (a < 1.75f) q = 1.5f;
    else if (a <= 2.5f) q = 2.f;
    else if (a < 3.5f) q = 3.f;
    else if (a <= 5.f) q = 4.f;
    else q = 6.f;
    return y < 0 ? -q : q;
}

// E4M3 byte -> float (exp 0: subnormal m 2^-9; 0x7f NaN - absent from the tables here)
inline float e4m3_f(uint8_t b) {
    const int a = b & 0x7f, ex = a >> 3, m = a & 7;
    const float v = ex == 0 ? (float) m * 0.001953125f : sycl::ldexp(1.0f + (float) m * 0.125f, ex - 7);
    return (b & 0x80) ? -v : v;
}
inline float e8m0_f(uint8_t b) {
    return sycl::bit_cast<float>(b == 0 ? 0x00400000u : ((uint32_t) b << 23));
}
inline float bf16_round(float x) {
    uint32_t u = sycl::bit_cast<uint32_t>(x);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return sycl::bit_cast<float>(u);
}

}  // namespace

extern "C" {

int ns_ds41_quant_sim(ns_gpu* g, float* x, int64_t rows, int64_t cols, int64_t block, int kind) {
    NS_TRY
    if (block <= 0 || block > 64 || cols % block != 0 || kind < 0 || kind > 2) return ns_fail("ns_ds41_quant_sim: a block of 1..64 dividing the row");
    const int64_t nb = rows * (cols / block);
    g->q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
        float* v = x + id[0] * block;
        float amax = 0.f;
        for (int64_t i = 0; i < block; ++i) amax = sycl::fmax(amax, sycl::fabs(v[i]));
        if (kind == 0) {
            const float s = pow2_ceil(sycl::fmax(amax, 1e-4f) / 448.f);
            for (int64_t i = 0; i < block; ++i) v[i] = e4m3_round(sycl::fmin(sycl::fmax(v[i] / s, -448.f), 448.f)) * s;
        } else {
            const float s = kind == 1 ? pow2_ceil(sycl::fmax(amax, 6.f * 1.1754944e-38f) / 6.f) : e4m3_round(sycl::fmax(amax, 6.f * 0.001953125f) / 6.f);
            for (int64_t i = 0; i < block; ++i) v[i] = e2m1_round(sycl::fmin(sycl::fmax(v[i] / s, -6.f), 6.f)) * s;
        }
    });
    return 0;
    NS_CATCH
}

int ns_ds41_hc_collapse(ns_gpu* g, const float* X, const float* pre, float* h, int64_t T, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, C), [=](sycl::id<2> id) {
        const int64_t t = id[0], k = id[1];
        float s = 0.f;
        for (int i = 0; i < 4; ++i) s += pre[t * 4 + i] * X[(t * 4 + i) * C + k];
        h[t * C + k] = s;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_engram_rows(ns_gpu* g, const uint8_t* raw, float* out, int64_t n, int64_t hd) {
    NS_TRY
    if (hd % 32 != 0) return ns_fail("ns_ds41_engram_rows: a row a multiple of 32");
    const int64_t rb = hd + hd / 32;
    g->q.parallel_for(sycl::range<2>(n, hd), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        const uint8_t* row = raw + r * rb;
        out[r * hd + c] = bf16_round(e4m3_f(row[c]) * e8m0_f(row[hd + c / 32]));
    });
    return 0;
    NS_CATCH
}

int ns_ds41_engram_gate(ns_gpu* g, float* X, const float* kv, const float* qk, int64_t T, int64_t C, float eps) {
    NS_TRY
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(T * 4 * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t ts = it.get_group(0), t = ts / 4, s = ts % 4;
        const int lid = it.get_local_id(0);
        float* x = X + (t * 4 + s) * C;
        const float* key = kv + t * 5 * C + s * C;
        const float* val = kv + t * 5 * C + 4 * C;
        const float* w = qk + s * C;
        float dot = 0.f, xx = 0.f, kk = 0.f;
        for (int64_t c = lid; c < C; c += WG) {
            dot += x[c] * w[c] * key[c];
            xx += x[c] * x[c];
            kk += key[c] * key[c];
        }
        dot = sycl::reduce_over_group(it.get_group(), dot, sycl::plus<float>());
        xx = sycl::reduce_over_group(it.get_group(), xx, sycl::plus<float>());
        kk = sycl::reduce_over_group(it.get_group(), kk, sycl::plus<float>());
        const float d = dot * sycl::rsqrt(xx / (float) C + eps) * sycl::rsqrt(kk / (float) C + eps) * sycl::rsqrt((float) C);
        const float r = sycl::sqrt(sycl::fmax(sycl::fabs(d), 1e-6f));
        const float gate = 1.f / (1.f + sycl::exp(-(d < 0 ? -r : r)));
        sycl::group_barrier(it.get_group());
        for (int64_t c = lid; c < C; c += WG) x[c] += gate * val[c];
    });
    return 0;
    NS_CATCH
}

int ns_ds41_idx_score(ns_gpu* g, const float* q, const float* w, const uint16_t* k, float* score, int64_t T, int64_t H, int64_t D, int64_t n,
                      int64_t ld, int64_t pos0, int64_t ratio) {
    NS_TRY
    constexpr int WG = 128, HM = 32, CH = 16;
    if (H > HM || D % CH != 0 || ratio < 1) return ns_fail("ns_ds41_idx_score: at most 32 heads, dims a multiple of 16");
    const sycl::half* kh = reinterpret_cast<const sycl::half*>(k);
    const int64_t per = (ld + WG - 1) / WG;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(HM * 128 > H * D ? (size_t) (H * D) : (size_t) (H * D)), h);
        sycl::local_accessor<float, 1> ws(sycl::range<1>(HM), h);
        h.parallel_for(sycl::nd_range<1>(T * per * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0), t = gi / per, j = (gi % per) * WG + it.get_local_id(0);
            const int lid = it.get_local_id(0);
            for (int64_t i = lid; i < H * D; i += WG) qs[i] = q[t * H * D + i];
            if (lid < H) ws[lid] = w[t * H + lid];
            sycl::group_barrier(it.get_group());
            if (j >= ld) return;
            const int64_t nv = sycl::min<int64_t>(n, (pos0 + t + 1) / ratio);
            if (j >= nv) {
                score[t * ld + j] = -INFINITY;
                return;
            }
            float acc[HM];
            for (int hh = 0; hh < HM; ++hh) acc[hh] = 0.f;
            const sycl::half* kr = kh + j * D;
            for (int64_t c0 = 0; c0 < D; c0 += CH) {
                float kv[CH];
                for (int c = 0; c < CH; ++c) kv[c] = (float) kr[c0 + c];
                for (int hh = 0; hh < HM; ++hh) {
                    if (hh >= H) break;
                    float s = 0.f;
                    for (int c = 0; c < CH; ++c) s += qs[hh * D + c0 + c] * kv[c];
                    acc[hh] += s;
                }
            }
            float s = 0.f;
            for (int hh = 0; hh < HM; ++hh)
                if (hh < H) s += sycl::fmax(acc[hh], 0.f) * ws[hh];
            score[t * ld + j] = s;
        });
    });
    return 0;
    NS_CATCH
}

int ns_ds41_block_max(ns_gpu* g, const float* score, float* bmax, int64_t T, int64_t n, int64_t ld, int64_t bs, int64_t pos0, int64_t ratio) {
    NS_TRY
    const int64_t nb = (n + bs - 1) / bs;
    g->q.parallel_for(sycl::range<2>(T, nb), [=](sycl::id<2> id) {
        const int64_t t = id[0], b = id[1];
        const int64_t lens = sycl::min<int64_t>(n, (pos0 + t + 1) / ratio);
        float m = -INFINITY;
        for (int64_t j = b * bs; j < sycl::min<int64_t>(n, (b + 1) * bs); ++j) m = sycl::fmax(m, score[t * ld + j]);
        // the block of the row's newest position: in, whatever it scores (it is only partly filled)
        if (lens > 0 && b == (lens - 1) / bs) m = INFINITY;
        bmax[t * nb + b] = m;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_cand_mask(ns_gpu* g, float* score, const int32_t* cand, uint8_t* mark, int64_t T, int64_t n, int64_t ld, int64_t K, int64_t bs) {
    NS_TRY
    const int64_t nb = (n + bs - 1) / bs;
    auto& q = g->q;
    q.memset(mark, 0, (size_t) (T * nb));
    q.parallel_for(sycl::range<2>(T, K), [=](sycl::id<2> id) {
        const int32_t b = cand[id[0] * K + id[1]];
        if (b >= 0 && b < nb) mark[id[0] * nb + b] = 1;
    });
    q.parallel_for(sycl::range<2>(T, n), [=](sycl::id<2> id) {
        const int64_t t = id[0], j = id[1];
        if (!mark[t * nb + j / bs]) score[t * ld + j] = -INFINITY;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_markov_step(ns_gpu* g, const float* logits, const float* bias, int64_t n, int32_t* tok) {
    NS_TRY
    constexpr int WG = 1024;
    g->q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const int lid = it.get_local_id(0);
        float best = -INFINITY;
        int32_t bi = 0;
        for (int64_t i = lid; i < n; i += WG) {
            const float v = logits[i] + bias[i];
            if (v > best) {
                best = v;
                bi = (int32_t) i;
            }
        }
        const float m = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
        // the lowest index of the best (the host's argmax keeps the first)
        const int32_t w = sycl::reduce_over_group(it.get_group(), best == m ? bi : INT32_MAX, sycl::minimum<int32_t>());
        if (lid == 0) *tok = w;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_gather_rows(ns_gpu* g, const float* table, const int32_t* idx, int64_t count, int64_t cols, float* out) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(count, cols), [=](sycl::id<2> id) { out[id[0] * cols + id[1]] = table[(int64_t) idx[id[0]] * cols + id[1]]; });
    return 0;
    NS_CATCH
}

}  // extern "C"
