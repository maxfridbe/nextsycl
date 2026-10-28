// skintokens.cpp: SkinTokens / TokenRig's kernels (skintokens.h): layer and RMS norms, activations, the Qwen3 decoder's
// per-head norm and rotary embedding (writing the keys and values to the beams' slot pool), attention (dense, and a decode
// step's over a slot pool shared by the beams; grouped-query), gathers - all float32 as the reference runs. SYCL only, on the
// GPU's own queue; the linear layers are the shared diffusion kernels' (oneDNN, float32).
#include "ns.h"
#include "ns_internal.hpp"
#include "skintokens.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <limits>

namespace {
inline float gelu_erf(float v) {
    return 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
}
}  // namespace

extern "C" {

int ns_model3d_skintokens_ln(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, const float* b, float* out) {
    NS_TRY
    constexpr int WG = 256, PER = 32;
    if (C > (int64_t) WG * PER) return ns_fail("ns_model3d_skintokens_ln: rows longer than 8192");
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const float* row = x + r * x_row;
        float v[PER];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            v[j] = i < C ? row[i] : 0.0f;
            s += v[j];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float s2 = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            const float d = i < C ? v[j] - mean : 0.0f;
            s2 += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
        sycl::group_barrier(it.get_group());
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) {
                float y = (v[j] - mean) * inv;
                if (w) y = y * w[i];
                if (b) y = y + b[i];
                out[r * C + i] = y;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_rms(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, float* out) {
    NS_TRY
    constexpr int WG = 256, PER = 32;
    if (C > (int64_t) WG * PER) return ns_fail("ns_model3d_skintokens_rms: rows longer than 8192");
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const float* row = x + r * x_row;
        float v[PER];
        float s2 = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            v[j] = i < C ? row[i] : 0.0f;
            s2 += v[j] * v[j];
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
        sycl::group_barrier(it.get_group());
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) out[r * C + i] = v[j] * inv * (w ? w[i] : 1.0f);
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_act(ns_gpu* g, float* x, int64_t n, int kind) {
    NS_TRY
    if (kind < 0 || kind > 1) return ns_fail("ns_model3d_skintokens_act: kind 0 (gelu) or 1 (sigmoid)");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        x[i] = kind == 0 ? gelu_erf(v) : 1.0f / (1.0f + sycl::exp(-v));
    });
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] = x[i] + y[i]; });
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_swiglu(ns_gpu* g, const float* gu, int64_t M, int64_t I, float* out) {
    NS_TRY
    if (M <= 0 || I <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) I), [=](sycl::id<2> id) {
        const float a = gu[id[0] * 2 * I + id[1]], b = gu[id[0] * 2 * I + I + id[1]];
        out[id[0] * I + id[1]] = a / (1.0f + sycl::exp(-a)) * b;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_rowdot(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float b, int act, float* out) {
    NS_TRY
    constexpr int SG = 16, NSG = 8;
    if (M <= 0 || C <= 0) return 0;
    const int64_t groups = (M + NSG - 1) / NSG;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * NSG * SG)), sycl::range<1>(NSG * SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t r = it.get_group(0) * NSG + sg.get_group_linear_id();
        if (r >= M) return;
        float s = 0.0f;
        for (int64_t i = l; i < C; i += SG) s += x[r * C + i] * w[i];
        s = sycl::reduce_over_group(sg, s, sycl::plus<float>()) + b;
        if (l == 0) out[r] = act == 1 ? 1.0f / (1.0f + sycl::exp(-s)) : s;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_gather(ns_gpu* g, const float* x, int64_t x_row, const int32_t* idx, int64_t R, int64_t C, float* out) {
    NS_TRY
    if (R <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) R, (size_t) C), [=](sycl::id<2> id) {
        out[id[0] * C + id[1]] = x[(int64_t) idx[id[0]] * x_row + id[1]];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_qkv_post(ns_gpu* g, float* qkv, int64_t M, int64_t Hq, int64_t Hkv, int64_t D, float eps, const float* qw, const float* kw,
                                   const int32_t* pos, const float* inv_freq, const int32_t* slot, float* kpool, float* vpool) {
    NS_TRY
    constexpr int MAXD = 128;
    if (D <= 0 || D > MAXD || (D & 1)) return ns_fail("ns_model3d_skintokens_qkv_post: D even, at most 128");
    if (M <= 0) return 0;
    const int64_t row = (Hq + 2 * Hkv) * D, half = D / 2, kvrow = Hkv * D;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) (Hq + Hkv)), [=](sycl::id<2> id) {
        const int64_t m = id[0], h = id[1];
        const bool is_q = h < Hq;
        float* p = qkv + m * row + h * D;
        const float* w = is_q ? qw : kw;
        float v[MAXD];
        float s2 = 0.0f;
        for (int64_t i = 0; i < D; ++i) {
            v[i] = p[i];
            s2 += v[i] * v[i];
        }
        const float inv = sycl::rsqrt(s2 / (float) D + eps);
        for (int64_t i = 0; i < D; ++i) v[i] = v[i] * inv * w[i];
        const float t = (float) pos[m];
        for (int64_t i = 0; i < half; ++i) {
            const float a = t * inv_freq[i];
            const float c = sycl::cos(a), s = sycl::sin(a);
            const float x1 = v[i], x2 = v[i + half];
            p[i] = x1 * c - x2 * s;
            p[i + half] = x2 * c + x1 * s;
        }
        if (!is_q) {
            const int64_t kh = h - Hq;
            const int64_t at = (int64_t) slot[m] * kvrow + kh * D;
            const float* pv = qkv + m * row + (Hq + Hkv + kh) * D;
            for (int64_t i = 0; i < D; ++i) {
                kpool[at + i] = p[i];
                vpool[at + i] = pv[i];
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                    const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv,
                                    int64_t D, int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row) {
    NS_TRY
    constexpr int WG = 256, MAXD = 1024;
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (Hkv <= 0 || H % Hkv) return ns_fail("ns_model3d_skintokens_attention: the heads must be a multiple of the key / value heads");
    if (D > MAXD || D <= 0 || Dv <= 0) return ns_fail("ns_model3d_skintokens_attention: 0 < D <= 1024, Dv > 0");
    if (scratch_n < Skv) return ns_fail("ns_model3d_skintokens_attention: the scratch holds less than one row of scores");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_skintokens_attention: a causal mask needs Sq = Skv");
    const int64_t total = B * H * Sq;
    const int64_t per = std::max<int64_t>(1, std::min<int64_t>(total, scratch_n / Skv));
    const int64_t group = H / Hkv;
    for (int64_t g0 = 0; g0 < total; g0 += per) {
        const int64_t n = std::min(per, total - g0);
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> qv(sycl::range<1>(MAXD), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
                const int64_t gr = it.get_group(0), gi = g0 + gr;
                const int64_t i = gi % Sq, hh = (gi / Sq) % H, b = gi / (Sq * H);
                const int64_t hk = hh / group;
                const int l = (int) it.get_local_id(0);
                const float* qr = q + (b * Sq + i) * q_row + hh * q_head;
                for (int64_t d = l; d < D; d += WG) qv[d] = qr[d];
                sycl::group_barrier(it.get_group());
                const int64_t nk = causal ? i + 1 : Skv;
                float* sc = scratch + gr * Skv;
                float mx = -std::numeric_limits<float>::infinity();
                for (int64_t j = l; j < nk; j += WG) {
                    const float* kr = k + (b * Skv + j) * k_row + hk * k_head;
                    float s = 0.0f;
                    for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
                    s *= scale;
                    sc[j] = s;
                    mx = sycl::fmax(mx, s);
                }
                mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
                float sum = 0.0f;
                for (int64_t j = l; j < nk; j += WG) {
                    const float e = sycl::exp(sc[j] - mx);
                    sc[j] = e;
                    sum += e;
                }
                sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
                sycl::group_barrier(it.get_group());
                const float inv = 1.0f / sum;
                for (int64_t d = l; d < Dv; d += WG) {
                    const float* vc = v + b * Skv * v_row + hk * v_head + d;
                    float acc = 0.0f;
                    for (int64_t j = 0; j < nk; ++j) acc += sc[j] * vc[j * v_row];
                    out[(b * Sq + i) * out_row + hh * Dv + d] = acc * inv;
                }
            });
        });
    }
    return 0;
    NS_CATCH
}

int ns_model3d_skintokens_attn_slots(ns_gpu* g, const float* q, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride,
                                     int64_t len, int64_t B, int64_t H, int64_t Hkv, int64_t D, float scale, float* out) {
    NS_TRY
    constexpr int WG = 256, MAXL = 8192, MAXD = 128;
    if (len <= 0 || B <= 0) return 0;
    if (len > MAXL || D > MAXD || D <= 0) return ns_fail("ns_model3d_skintokens_attn_slots: len <= 8192, 0 < D <= 128");
    if (Hkv <= 0 || H % Hkv) return ns_fail("ns_model3d_skintokens_attn_slots: the heads must be a multiple of the key / value heads");
    const int64_t row = Hkv * D, group = H / Hkv;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(MAXL), h);
        sycl::local_accessor<float, 1> qv(sycl::range<1>(MAXD), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * H * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0), b = gi / H, hh = gi % H, hk = hh / group;
            const int l = (int) it.get_local_id(0);
            for (int64_t d = l; d < D; d += WG) qv[d] = q[b * q_row + hh * D + d];
            sycl::group_barrier(it.get_group());
            const int32_t* st = slots + b * stride;
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t t = l; t < len; t += WG) {
                const float* kr = kpool + (int64_t) st[t] * row + hk * D;
                float s = 0.0f;
                for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
                s *= scale;
                sc[t] = s;
                mx = sycl::fmax(mx, s);
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t t = l; t < len; t += WG) {
                const float e = sycl::exp(sc[t] - mx);
                sc[t] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            sycl::group_barrier(it.get_group());
            const float inv = 1.0f / sum;
            for (int64_t d = l; d < D; d += WG) {
                float acc = 0.0f;
                for (int64_t t = 0; t < len; ++t) acc += sc[t] * vpool[(int64_t) st[t] * row + hk * D + d];
                out[b * H * D + hh * D + d] = acc * inv;
            }
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
