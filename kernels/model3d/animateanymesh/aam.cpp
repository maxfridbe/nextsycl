// aam.cpp: AnimateAnyMesh's kernels (aam.h): the norms, activations, attention (dense and over a mesh's neighbours),
// farthest point sampling and the flow solver's steps, all float32 as the reference runs. SYCL only, on the GPU's own
// queue; the linear layers are the shared diffusion kernels' (oneDNN, float32).
#include "ns.h"
#include "ns_internal.hpp"
#include "aam.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <climits>
#include <limits>

namespace {
inline float gelu_erf(float v) {
    return 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
}
}  // namespace

extern "C" {

int ns_model3d_aam_ln(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, int64_t S, int64_t T,
                      const float* w0, const float* b0, const float* scale0, const float* shift0,
                      const float* w1, const float* b1, const float* scale1, const float* shift1, float* out) {
    NS_TRY
    constexpr int WG = 256, PER = 32;            // a row of up to 8,192 values held in registers
    if (C > (int64_t) WG * PER) return ns_fail("ns_model3d_aam_ln: rows longer than 8192");
    if (S <= 0 || T <= 0 || T > S) return ns_fail("ns_model3d_aam_ln: 0 < T <= S");
    if (M <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const bool s1 = (r % S) >= T;
        const float* w = s1 ? w1 : w0;
        const float* b = s1 ? b1 : b0;
        const float* sc = s1 ? scale1 : scale0;
        const float* sh = s1 ? shift1 : shift0;
        const float* row = x + r * C;
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
        sycl::group_barrier(it.get_group());     // every lane has read its row before any writes it (out may be x)
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) {
                float y = (v[j] - mean) * inv;
                if (w) y = y * w[i];
                if (b) y = y + b[i];
                if (sc) y = y * (1.0f + sc[i]);
                if (sh) y = y + sh[i];
                out[r * C + i] = y;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_gated_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, int64_t S, int64_t T, const float* gate0,
                             const float* gate1) {
    NS_TRY
    if (S <= 0 || T <= 0 || T > S) return ns_fail("ns_model3d_aam_gated_add: 0 < T <= S");
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        const float* gt = (r % S) >= T ? gate1 : gate0;
        const float add = gt ? gt[c] * y[r * C + c] : y[r * C + c];
        x[r * C + c] = x[r * C + c] + add;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_act(ns_gpu* g, float* x, int64_t n, int kind) {
    NS_TRY
    if (kind < 0 || kind > 2) return ns_fail("ns_model3d_aam_act: kind 0 (gelu), 1 (silu) or 2 (quick gelu)");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        float y;
        if (kind == 0) y = gelu_erf(v);
        else if (kind == 1) y = v / (1.0f + sycl::exp(-v));
        else y = v / (1.0f + sycl::exp(-1.702f * v));
        x[i] = y;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_geglu(ns_gpu* g, const float* x, int64_t M, int64_t C, float* out) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        out[r * C + c] = x[r * 2 * C + c] * gelu_erf(x[r * 2 * C + C + c]);
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                             const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                             int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row) {
    NS_TRY
    constexpr int WG = 256, MAXD = 1024;
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (D > MAXD || D <= 0 || Dv <= 0) return ns_fail("ns_model3d_aam_attention: 0 < D <= 1024, Dv > 0");
    if (scratch_n < Skv) return ns_fail("ns_model3d_aam_attention: the scratch holds less than one row of scores");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_aam_attention: a causal mask needs Sq = Skv");
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
                // every lane's weights (global memory) seen by every lane
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

int ns_model3d_aam_csr_attention(ns_gpu* g, const float* q, const float* k, const float* v, int64_t N, int64_t D, const int32_t* rowptr,
                                 const int32_t* cols, float scale, float* out) {
    NS_TRY
    constexpr int WG = 128, MAXN = 1024;
    if (D <= 0 || D > 1024) return ns_fail("ns_model3d_aam_csr_attention: 0 < D <= 1024");
    if (N <= 0) return 0;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(MAXN), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (N * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t i = it.get_group(0);
            const int l = (int) it.get_local_id(0);
            const int64_t a = rowptr[i];
            const int64_t n = std::min<int64_t>((int64_t) rowptr[i + 1] - a, (int64_t) MAXN);
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t e = 0; e < n; ++e) {
                const float* kr = k + (int64_t) cols[a + e] * D;
                float s = 0.0f;
                for (int64_t d = l; d < D; d += WG) s += q[i * D + d] * kr[d];
                s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) * scale;
                if (l == 0) sc[e] = s;
                mx = sycl::fmax(mx, s);
            }
            sycl::group_barrier(it.get_group());
            float sum = 0.0f;
            for (int64_t e = 0; e < n; ++e) sum += sycl::exp(sc[e] - mx);
            for (int64_t d = l; d < D; d += WG) {
                float acc = 0.0f;
                for (int64_t e = 0; e < n; ++e) acc += sycl::exp(sc[e] - mx) * v[(int64_t) cols[a + e] * D + d];
                out[i * D + d] = n > 0 ? acc / sum : 0.0f;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_fps(ns_gpu* g, const float* x, int64_t N, int64_t D, int64_t K, float* dist, int32_t* idx) {
    NS_TRY
    constexpr int WG = 1024, MAXD = 1024;
    if (K <= 0 || N <= 0) return 0;
    if (K > N || D <= 0 || D > MAXD) return ns_fail("ns_model3d_aam_fps: K <= N, 0 < D <= 1024");
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sel(sycl::range<1>(MAXD), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
#pragma clang fp contract(off) reassociate(off)
            const int l = (int) it.get_local_id(0);
            for (int64_t p = l; p < N; p += WG) dist[p] = std::numeric_limits<float>::max();
            if (l == 0) idx[0] = 0;
            int64_t cur = 0;
            for (int64_t kk = 1; kk < K; ++kk) {
                for (int64_t d = l; d < D; d += WG) sel[d] = x[cur * D + d];
                sycl::group_barrier(it.get_group());
                float best = -1.0f;
                int bi = INT_MAX;
                for (int64_t p = l; p < N; p += WG) {
                    const float* xp = x + p * D;
                    float s = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        const float diff = sel[d] - xp[d];
                        const float sq = diff * diff;
                        s = s + sq;
                    }
                    const float dd = sycl::fmin(s, dist[p]);
                    dist[p] = dd;
                    if (dd > best) {
                        best = dd;
                        bi = (int) p;
                    }
                }
                const float m = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
                const int c = sycl::reduce_over_group(it.get_group(), best == m ? bi : INT_MAX, sycl::minimum<int>());
                cur = c;
                if (l == 0) idx[kk] = c;
                sycl::group_barrier(it.get_group());
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_gather(ns_gpu* g, const float* x, const int32_t* idx, int64_t R, int64_t C, float* out) {
    NS_TRY
    if (R <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) R, (size_t) C), [=](sycl::id<2> id) {
        out[id[0] * C + id[1]] = x[(int64_t) idx[id[0]] * C + id[1]];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_flow_in(ns_gpu* g, const float* f0, const float* y, int64_t K, int64_t C, int64_t F, float* out) {
    NS_TRY
    if (K <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) (2 * K), (size_t) C), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1], kk = r % K;
        out[r * C + c] = c < F ? f0[kk * F + c] : y[kk * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_cfg(ns_gpu* g, const float* flow, int64_t n, float scale, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
#pragma clang fp contract(off)
        const float c = flow[i], u = flow[n + i];
        const float d = c - u;
        const float s = scale * d;
        out[i] = u + s;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_aam_axpy(ns_gpu* g, const float* y, const float* f, float a, int64_t n, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
#pragma clang fp contract(off)
        const float s = f[i] * a;
        out[i] = y[i] + s;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
