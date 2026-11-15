// b3d.cpp: the Budget3D-Idol engine's own kernels - what the shared diffusion kernels (nsd) do not have: the layer
// norm with a modulation row per sequence, the gated residual adds, Idol's learned rotary positions (RePo) with its
// per-head RMS norm, the refiners' attention within a merged token's few children, the token assembly gathers, the
// ScaleMixer's mixing across children, and the FLUX.2 VAE encoder's group norm, padding and softmax. Every loop is
// bounded by its arguments; each kernel is checked against the reference stage it serves (reference/budget3d/ref.py,
// `nextsycl model3d check`).
#include "ns.h"
#include "ns_internal.hpp"
#include "b3d.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int NB = 64;   // the group norm: blocks of pixels a group's statistics are split in

inline float ld(const void* p, int dt, size_t i) {
    return dt == 0 ? ((const float*) p)[i] : (float) ((const half*) p)[i];
}
inline void st(void* p, int dt, size_t i, float v) {
    if (dt == 0) ((float*) p)[i] = v;
    else ((half*) p)[i] = (half) v;
}
bool dt_ok(int dt) { return dt == 0 || dt == 1; }
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
}  // namespace

extern "C" {

int ns_model3d_budget3d_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                               const float* scale, const float* shift, int64_t stride, int64_t group, void* out, int out_dt) {
    NS_TRY
    constexpr int WG = 256, PER = 32;            // a row of up to 8,192 values held in registers
    if (C > (int64_t) WG * PER || C <= 0) return ns_fail("ns_model3d_budget3d_ln_mod: rows of 1..8192");
    if (!dt_ok(x_dt) || !dt_ok(out_dt)) return ns_fail("ns_model3d_budget3d_ln_mod: float32 or half");
    if (M <= 0) return 0;
    constexpr int64_t kRows = int64_t(1) << 22;  // launches of at most 2^22 rows
    for (int64_t r0 = 0; r0 < M; r0 += kRows) {
    const int64_t rows = std::min(kRows, M - r0);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = r0 + it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const int64_t mr = group > 0 ? (int64_t) r / group * stride : 0;
        float v[PER];
        float s = 0.f;
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * WG;
            v[i] = c < C ? ld(x, x_dt, r * C + c) : 0.f;
            s += v[i];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float q = 0.f;
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * WG;
            if (c < C) q += (v[i] - mean) * (v[i] - mean);
        }
        const float var = sycl::reduce_over_group(it.get_group(), q, sycl::plus<float>()) / (float) C;
        const float rs = sycl::rsqrt(var + eps);
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * WG;
            if (c >= C) break;
            float y = (v[i] - mean) * rs;
            if (weight) y = y * weight[c] + (bias ? bias[c] : 0.f);
            if (scale) y = y * (1.f + scale[mr + c]);
            if (shift) y += shift[mr + c];
            st(out, out_dt, r * C + c, y);
        }
    });
    }
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* h, int h_dt, const float* gate, int64_t stride,
                                 int64_t group) {
    NS_TRY
    if (!dt_ok(h_dt)) return ns_fail("ns_model3d_budget3d_gate_add: float32 or half");
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / C, c = i % C;
        const float v = ld(h, h_dt, (size_t) i);
        x[i] += gate ? v * gate[(group > 0 ? r / group * stride : 0) + c] : v;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_rope_norm(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* delta, const float* freqs,
                                  int64_t n0, int64_t n1, const float* gamma) {
    NS_TRY
    if (D != 64) return ns_fail("ns_model3d_budget3d_rope_norm: heads of 64");
    if (stride < H * D) return ns_fail("ns_model3d_budget3d_rope_norm: a stride under a row");
    if (n0 < 0 || n1 < 0 || n0 + n1 > D / 2) return ns_fail("ns_model3d_budget3d_rope_norm: the axes' pairs exceed the head");
    if (M <= 0 || H <= 0) return 0;
    half* xh = (half*) x;
    // a sub-group of 32 a head: lane l holds pair l (features 2l, 2l + 1)
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * H * 32)), sycl::range<1>(32)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t grp = (int64_t) it.get_group(0), m = grp / H, h = grp % H;
        const int l = (int) it.get_local_id(0);
        half* p = xh + m * stride + h * D + 2 * l;
        float a = (float) p[0], b = (float) p[1];
        if (delta) {
            const int axis = l < n0 ? 0 : (l < n0 + n1 ? 1 : 2);
            const float ang = delta[(m * H + h) * 3 + axis] * freqs[l] * 3.14159265358979323846f;
            const float c = sycl::cos(ang), s = sycl::sin(ang);
            const float ra = a * c - b * s, rb = a * s + b * c;
            a = ra;
            b = rb;
        }
        const float ss = sycl::reduce_over_group(it.get_sub_group(), a * a + b * b, sycl::plus<float>());
        const float k = sycl::sqrt((float) D) / sycl::fmax(sycl::sqrt(ss), 1e-12f);
        p[0] = (half) (a * k * gamma[h * D + 2 * l]);
        p[1] = (half) (b * k * gamma[h * D + 2 * l + 1]);
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_small_attn(ns_gpu* g, const float* qkv, int64_t B, int64_t S, int64_t H, int64_t D, float* out) {
    NS_TRY
    constexpr int SMAX = 16, DMAX = 64;
    if (S <= 0 || S > SMAX || D <= 0 || D > DMAX) return ns_fail("ns_model3d_budget3d_small_attn: sequences of 1..16, heads of 1..64");
    if (B <= 0 || H <= 0) return 0;
    const int64_t W = H * D, row = 3 * W;
    const float scale = 1.f / std::sqrt((float) D);
    // one work-item a (sequence, head, query row)
    g->q.parallel_for(sycl::range<1>((size_t) (B * H * S)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], qs = i % S, h = (i / S) % H, b = i / (S * H);
        const float* q = qkv + (b * S + qs) * row + h * D;
        float sc[SMAX];
        float mx = -INFINITY;
        for (int64_t k = 0; k < S; ++k) {
            const float* kr = qkv + (b * S + k) * row + W + h * D;
            float d = 0.f;
            for (int64_t j = 0; j < D; ++j) d += q[j] * kr[j];
            sc[k] = d * scale;
            mx = sycl::fmax(mx, sc[k]);
        }
        float sum = 0.f;
        for (int64_t k = 0; k < S; ++k) {
            sc[k] = sycl::exp(sc[k] - mx);
            sum += sc[k];
        }
        float acc[DMAX];
        for (int64_t j = 0; j < D; ++j) acc[j] = 0.f;
        for (int64_t k = 0; k < S; ++k) {
            const float* vr = qkv + (b * S + k) * row + 2 * W + h * D;
            const float p = sc[k] / sum;
            for (int64_t j = 0; j < D; ++j) acc[j] += p * vr[j];
        }
        float* o = out + (b * S + qs) * W + h * D;
        for (int64_t j = 0; j < D; ++j) o[j] = acc[j];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_gather_add(ns_gpu* g, const float* src, int64_t C, const int32_t* idx, int64_t n, const float* add, const int32_t* aidx,
                                   float* out) {
    NS_TRY
    if (n <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], j = i / C, c = i % C;
        float v = src[(int64_t) idx[j] * C + c];
        if (add) v += add[(aidx ? (int64_t) aidx[j] : 0) * C + c];
        out[i] = v;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_slot_add(ns_gpu* g, const float* x, int64_t R, int64_t S, int64_t C, const float* pos, float* out) {
    NS_TRY
    if (R <= 0 || S <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (R * S * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C, s = (i / C) % S;
        out[i] = x[i] + pos[s * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_mix(ns_gpu* g, const float* h, int64_t R, int64_t S, int64_t C, const float* w, const float* b, float* out) {
    NS_TRY
    if (S <= 0 || S > 16) return ns_fail("ns_model3d_budget3d_mix: 1..16 children");
    if (R <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (R * S * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C, t = (i / C) % S, r = i / (C * S);
        float acc = b[t * C + c];
        for (int64_t s = 0; s < S; ++s) acc += h[(r * S + s) * C + c] * w[(c * S + t) * S + s];
        out[i] = 0.5f * acc * (1.f + sycl::erf(acc * 0.70710678118654752f));
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_group_mean(ns_gpu* g, const float* x, int64_t R, int64_t S, int64_t C, float* out) {
    NS_TRY
    if (R <= 0 || S <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (R * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C, r = i / C;
        float s = 0.f;
        for (int64_t k = 0; k < S; ++k) s += x[(r * S + k) * C + c];
        out[i] = s / (float) S;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_refine_init(ns_gpu* g, const float* e, int64_t R, int64_t S, int64_t C, const float* slot, const float* zt,
                                    const float* carry, float* out) {
    NS_TRY
    if (R <= 0 || S <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (R * S * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C, s = (i / C) % S, r = i / (C * S);
        float v = e[r * C + c] + slot[s * C + c] + zt[i];
        if (carry) v += carry[r * C + c];
        out[i] = v;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_convert(ns_gpu* g, const void* src, int src_dt, int64_t n, void* dst, int dst_dt) {
    NS_TRY
    if (!dt_ok(src_dt) || !dt_ok(dst_dt)) return ns_fail("ns_model3d_budget3d_convert: float32 or half");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { st(dst, dst_dt, i[0], ld(src, src_dt, i[0])); });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_silu(ns_gpu* g, void* x, int dt, int64_t n) {
    NS_TRY
    if (!dt_ok(dt)) return ns_fail("ns_model3d_budget3d_silu: float32 or half");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = ld(x, dt, i[0]);
        st(x, dt, i[0], v / (1.f + sycl::exp(-v)));
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_add(ns_gpu* g, void* x, int x_dt, const void* y, int y_dt, int64_t n) {
    NS_TRY
    if (!dt_ok(x_dt) || !dt_ok(y_dt)) return ns_fail("ns_model3d_budget3d_add: float32 or half");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { st(x, x_dt, i[0], ld(x, x_dt, i[0]) + ld(y, y_dt, i[0])); });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_group_norm(ns_gpu* g, const void* x, int64_t P, int64_t C, int64_t G, float eps, const float* gamma, const float* beta,
                                   int silu, void* out, float* scratch) {
    NS_TRY
    constexpr int WG = 256;
    if (P <= 0) return 0;
    if (G <= 0 || C % G != 0) return ns_fail("ns_model3d_budget3d_group_norm: C a multiple of the groups");
    const half* xs = (const half*) x;
    half* o = (half*) out;
    const int64_t cg = C / G;
    float* part = scratch;                   // [G][NB]
    float* stat = scratch + G * NB;          // [G][2]: mean, then rstd
    const float n = (float) (P * cg);
    // pass 1: the partial sums, then the means; pass 2: the partial sums of squared deviations, then the rstd
    for (int pass = 0; pass < 2; ++pass) {
        g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>(sz(G * NB) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0) / NB, b = it.get_group(0) % NB, l = it.get_local_id(0);
            const int64_t p0 = P * b / NB, p1 = P * (b + 1) / NB;
            const float mean = pass == 0 ? 0.0f : stat[gi * 2];
            float s = 0.0f;
            for (int64_t i = l; i < (p1 - p0) * cg; i += WG) {
                const float v = (float) xs[(p0 + i / cg) * C + gi * cg + i % cg] - mean;
                s += pass == 0 ? v : v * v;
            }
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            if (l == 0) part[gi * NB + b] = s;
        });
        g->q.parallel_for(sycl::range<1>(sz(G)), [=](sycl::id<1> id) {
            const int64_t gi = id[0];
            float s = 0.0f;
            for (int b = 0; b < NB; ++b) s += part[gi * NB + b];
            if (pass == 0) stat[gi * 2] = s / n;
            else stat[gi * 2 + 1] = 1.0f / sycl::sqrt(s / n + eps);
        });
    }
    g->q.parallel_for(sycl::range<1>(sz(P * C)), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % C, gi = c / cg;
        float v = ((float) xs[i] - stat[gi * 2]) * stat[gi * 2 + 1] * gamma[c] + beta[c];
        if (silu) v = v / (1.0f + sycl::exp(-v));
        o[i] = (half) v;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_pad_br(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    NS_TRY
    if (H <= 0 || W <= 0 || C <= 0) return 0;
    const half* xs = (const half*) x;
    half* o = (half*) out;
    const int64_t W1 = W + 1;
    g->q.parallel_for(sycl::range<1>(sz((H + 1) * W1 * C)), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % C, xx = (i / C) % W1, y = i / (C * W1);
        o[i] = (y < H && xx < W) ? xs[(y * W + xx) * C + c] : (half) 0.f;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_budget3d_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out) {
    NS_TRY
    constexpr int WG = 256;
    half* o = (half*) out;
    if (M > 0 && N > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>(sz(M) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_local_id(0);
        const float* row = x + r * N;
        float m = -INFINITY;
        for (int64_t i = l; i < N; i += WG) m = sycl::fmax(m, row[i] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t i = l; i < N; i += WG) s += sycl::exp(row[i] * scale - m);
        const float inv = 1.0f / sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        for (int64_t i = l; i < N; i += WG) o[r * N + i] = (half) (sycl::exp(row[i] * scale - m) * inv);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
