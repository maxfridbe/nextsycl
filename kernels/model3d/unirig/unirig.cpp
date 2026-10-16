// unirig.cpp: UniRig's kernels (unirig.h): norms, activations, attention (dense, and a decode step's over a slot pool
// of keys and values shared by the beams), gathers, the sparse convolutions' im2col and the skin head's per-head bone
// attention, all float32 as the reference runs. SYCL only, on the GPU's own queue; the linear layers are the shared
// diffusion kernels' (oneDNN, float32).
#include "ns.h"
#include "ns_internal.hpp"
#include "unirig.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <limits>

namespace {
inline float gelu_erf(float v) {
    return 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
}
}  // namespace

extern "C" {

int ns_model3d_unirig_ln(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, const float* b, float* out) {
    NS_TRY
    constexpr int WG = 256, PER = 32;
    if (C > (int64_t) WG * PER) return ns_fail("ns_model3d_unirig_ln: rows longer than 8192");
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

int ns_model3d_unirig_head_norm(ns_gpu* g, float* x, int64_t M, int64_t row, int64_t col, int64_t n_heads, int64_t D, float eps) {
    NS_TRY
    if (D <= 0 || D > 64) return ns_fail("ns_model3d_unirig_head_norm: 0 < D <= 64");
    if (M <= 0 || n_heads <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) n_heads), [=](sycl::id<2> id) {
        float* p = x + id[0] * row + col + id[1] * D;
        float v[64];
        float s = 0.0f;
        for (int64_t i = 0; i < D; ++i) {
            v[i] = p[i];
            s += v[i];
        }
        const float mean = s / (float) D;
        float s2 = 0.0f;
        for (int64_t i = 0; i < D; ++i) s2 += (v[i] - mean) * (v[i] - mean);
        const float inv = sycl::rsqrt(s2 / (float) D + eps);
        for (int64_t i = 0; i < D; ++i) p[i] = (v[i] - mean) * inv;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_round_half(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] = (float) (sycl::half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_act(ns_gpu* g, float* x, int64_t n, int kind) {
    NS_TRY
    if (kind < 0 || kind > 1) return ns_fail("ns_model3d_unirig_act: kind 0 (gelu) or 1 (relu)");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        x[i] = kind == 0 ? gelu_erf(v) : sycl::fmax(v, 0.0f);
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] = x[i] + y[i]; });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_affine(ns_gpu* g, float* x, int64_t M, int64_t C, const float* scale, const float* shift) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t i = id[0] * C + id[1];
        x[i] = x[i] * scale[id[1]] + shift[id[1]];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                                int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row) {
    NS_TRY
    constexpr int WG = 256, MAXD = 1024;
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (D > MAXD || D <= 0 || Dv <= 0) return ns_fail("ns_model3d_unirig_attention: 0 < D <= 1024, Dv > 0");
    if (scratch_n < Skv) return ns_fail("ns_model3d_unirig_attention: the scratch holds less than one row of scores");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_unirig_attention: a causal mask needs Sq = Skv");
    const int64_t total = B * H * Sq;
    const int64_t per = std::max<int64_t>(1, std::min<int64_t>(total, scratch_n / Skv));
    for (int64_t g0 = 0; g0 < total; g0 += per) {
        const int64_t n = std::min(per, total - g0);
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> qv(sycl::range<1>(MAXD), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
                const int64_t gr = it.get_group(0), gi = g0 + gr;
                const int64_t i = gi % Sq, hh = (gi / Sq) % H, b = gi / (Sq * H);
                const int l = (int) it.get_local_id(0);
                const float* qr = q + (b * Sq + i) * q_row + hh * q_head;
                for (int64_t d = l; d < D; d += WG) qv[d] = qr[d];
                sycl::group_barrier(it.get_group());
                const int64_t nk = causal ? i + 1 : Skv;
                float* sc = scratch + gr * Skv;
                float mx = -std::numeric_limits<float>::infinity();
                for (int64_t j = l; j < nk; j += WG) {
                    const float* kr = k + (b * Skv + j) * k_row + hh * k_head;
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
                    const float* vc = v + b * Skv * v_row + hh * v_head + d;
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

int ns_model3d_unirig_attn_slots(ns_gpu* g, const float* q, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride, int64_t len,
                                 int64_t B, int64_t H, int64_t D, float scale, float* out) {
    NS_TRY
    constexpr int WG = 256, MAXL = 8192, MAXD = 128;
    if (len <= 0 || B <= 0) return 0;
    if (len > MAXL || D > MAXD || D <= 0) return ns_fail("ns_model3d_unirig_attn_slots: len <= 8192, 0 < D <= 128");
    const int64_t row = H * D;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(MAXL), h);
        sycl::local_accessor<float, 1> qv(sycl::range<1>(MAXD), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * H * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0), b = gi / H, hh = gi % H;
            const int l = (int) it.get_local_id(0);
            for (int64_t d = l; d < D; d += WG) qv[d] = q[b * row + hh * D + d];
            sycl::group_barrier(it.get_group());
            const int32_t* st = slots + b * stride;
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t t = l; t < len; t += WG) {
                const float* kr = kpool + (int64_t) st[t] * row + hh * D;
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
                for (int64_t t = 0; t < len; ++t) acc += sc[t] * vpool[(int64_t) st[t] * row + hh * D + d];
                out[b * row + hh * D + d] = acc * inv;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_gather(ns_gpu* g, const float* x, int64_t x_row, const int32_t* idx, int64_t R, int64_t C, float* out) {
    NS_TRY
    if (R <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) R, (size_t) C), [=](sycl::id<2> id) {
        out[id[0] * C + id[1]] = x[(int64_t) idx[id[0]] * x_row + id[1]];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_scatter(ns_gpu* g, const float* x, const int32_t* idx, int64_t R, int64_t C, float* out) {
    NS_TRY
    if (R <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) R, (size_t) C), [=](sycl::id<2> id) {
        out[(int64_t) idx[id[0]] * C + id[1]] = x[id[0] * C + id[1]];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_im2col(ns_gpu* g, const float* x, int64_t C, const int32_t* nbr, int64_t K, int64_t r0, int64_t R, float* out) {
    NS_TRY
    if (R <= 0 || C <= 0 || K <= 0) return 0;
    g->q.parallel_for(sycl::range<3>((size_t) R, (size_t) K, (size_t) C), [=](sycl::id<3> id) {
        const int64_t r = id[0], kk = id[1], c = id[2];
        const int32_t j = nbr[(r0 + r) * K + kk];
        out[(r * K + kk) * C + c] = j < 0 ? 0.0f : x[(int64_t) j * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_bone_attn(ns_gpu* g, const float* q, const float* k, int64_t R, int64_t J, int64_t H, int64_t D, float scale, float* out) {
    NS_TRY
    constexpr int WG = 128, MAXD = 1024;
    if (R <= 0 || J <= 0 || H <= 0) return 0;
    if (D <= 0 || D > MAXD) return ns_fail("ns_model3d_unirig_bone_attn: 0 < D <= 1024");
    const int64_t row = H * D;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qv(sycl::range<1>(MAXD), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (R * H * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0), n = gi / H, hh = gi % H;
            const int l = (int) it.get_local_id(0);
            for (int64_t d = l; d < D; d += WG) qv[d] = q[n * row + hh * D + d];
            sycl::group_barrier(it.get_group());
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t j = l; j < J; j += WG) {
                const float* kr = k + j * row + hh * D;
                float s = 0.0f;
                for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
                mx = sycl::fmax(mx, s * scale);
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t j = l; j < J; j += WG) {
                const float* kr = k + j * row + hh * D;
                float s = 0.0f;
                for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
                const float e = sycl::exp(s * scale - mx);
                out[(n * J + j) * H + hh] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
            sycl::group_barrier(it.get_group());
            const float inv = 1.0f / sum;
            for (int64_t j = l; j < J; j += WG) out[(n * J + j) * H + hh] *= inv;
        });
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unirig_skin_in(ns_gpu* g, const float* a, const float* vox, int64_t RJ, int64_t H, const float* aw, const float* ab,
                              const float* ew, const float* eb, const float* vw, const float* vb, float* out) {
    NS_TRY
    if (H <= 0 || H > 64) return ns_fail("ns_model3d_unirig_skin_in: 0 < H <= 64");
    if (RJ <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) RJ), [=](sycl::id<1> id) {
        const int64_t r = id[0];
        float v[64];
        // the attention's heads, normalized
        float s = 0.0f;
        for (int64_t c = 0; c < H; ++c) {
            v[c] = a[r * H + c];
            s += v[c];
        }
        float mean = s / (float) H, s2 = 0.0f;
        for (int64_t c = 0; c < H; ++c) s2 += (v[c] - mean) * (v[c] - mean);
        float inv = sycl::rsqrt(s2 / (float) H + 1e-5f);
        for (int64_t c = 0; c < H; ++c) out[r * 2 * H + c] = (v[c] - mean) * inv * aw[c] + ab[c];
        // the voxel prior through its embedding, normalized
        const float x = vox[r];
        s = 0.0f;
        for (int64_t c = 0; c < H; ++c) {
            v[c] = x * ew[c] + eb[c];
            s += v[c];
        }
        mean = s / (float) H;
        s2 = 0.0f;
        for (int64_t c = 0; c < H; ++c) s2 += (v[c] - mean) * (v[c] - mean);
        inv = sycl::rsqrt(s2 / (float) H + 1e-5f);
        for (int64_t c = 0; c < H; ++c) out[r * 2 * H + H + c] = (v[c] - mean) * inv * vw[c] + vb[c];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
