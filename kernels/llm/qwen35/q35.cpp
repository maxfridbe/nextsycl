// q35.cpp: the qwen35 engine's own kernels (q35.h) - what dense Qwen3.5 / Qwen3.8 has that the shared ones do not:
// the gated full attention (head 256, a gate per head in the query projection, partial rotary positions) against a
// half cache, and DeltaNet's gates and SiLU-gated output norm. After llama.cpp's qwen35 graph (src/models/qwen35.cpp).
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;

// ---- cached causal attention (the MiniMax Music engine's, mm3.cpp): a work-group per (row, key head, split of the
// keys), NSG sub-groups taking the split's keys in turn, each lane DL features of the G query heads that share the key
// head; the sub-groups' (max, sum, weighted values) merged in local memory, the splits (decode) by a second kernel
constexpr int NSG_A = 4;

// a decode row's keys a split: enough splits that the card has ~512 work-groups (key heads x splits) whatever the
// context - a fixed 256 left 36 work-groups at 2K (8% of the bandwidth) - at least 32 keys each; prompt rows one split
int64_t split_keys(int64_t S, int64_t kv, int64_t Hkv) {
    if (S != 1) return kv;
    const int64_t want = (kv * Hkv + 511) / 512;
    return sycl::max<int64_t>(32, (want + 31) / 32 * 32);
}
int64_t splits(int64_t S, int64_t kv, int64_t Hkv) {
    const int64_t kb = split_keys(S, kv, Hkv);
    return kv > kb ? (kv + kb - 1) / kb : 1;
}

template <int DL, int G>
void attn(sycl::queue& q, const float* qp, int64_t qs, const half* kc, const half* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t T, int64_t p0,
          float* out, float* part) {
    constexpr int D = SG * DL, WG = SG * NSG_A;
    const int64_t ns = splits(S, p0 + S, Hkv), KB = split_keys(S, p0 + S, Hkv);
    const float scale = 1.0f / sycl::sqrt((float) D);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NSG_A * G * (D + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * Hkv * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, r = gi / Hkv;
            const int64_t kv_end = p0 + r + 1;
            const int64_t j0 = sp * (ns > 1 ? KB : kv_end), j1 = sycl::min(kv_end, ns > 1 ? (sp + 1) * KB : kv_end);
            float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float* qr = qp + r * qs + (hk * G + g) * D + lane * DL;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    qv[g][i] = qr[i] * scale;
                    acc[g][i] = 0.0f;
                }
                m[g] = -INFINITY;
                l[g] = 0.0f;
            }
            const half* kb = kc + hk * T * D + lane * DL;
            const half* vb = vc + hk * T * D + lane * DL;
            for (int64_t j = j0 + sgi; j < j1; j += NSG_A) {
                const sycl::vec<half, DL> kv = *reinterpret_cast<const sycl::vec<half, DL>*>(kb + j * D);
                const sycl::vec<half, DL> vv = *reinterpret_cast<const sycl::vec<half, DL>*>(vb + j * D);
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    float d = 0.0f;
#pragma unroll
                    for (int i = 0; i < DL; ++i) d += qv[g][i] * (float) kv[i];
                    const float sc = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                    const float mn = sycl::fmax(m[g], sc);
                    const float c = sycl::exp(m[g] - mn), e = sycl::exp(sc - mn);
#pragma unroll
                    for (int i = 0; i < DL; ++i) acc[g][i] = acc[g][i] * c + e * (float) vv[i];
                    l[g] = l[g] * c + e;
                    m[g] = mn;
                }
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float* o = &slm[(sgi * G + g) * (D + 2)];
#pragma unroll
                for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
                if (lane == 0) {
                    o[D] = m[g];
                    o[D + 1] = l[g];
                }
            }
            sycl::group_barrier(it.get_group());
            for (int idx = (int) it.get_local_id(0); idx < G * D; idx += WG) {
                const int g = idx / D, d = idx % D;
                float mx = -INFINITY;
                for (int k = 0; k < NSG_A; ++k) mx = sycl::fmax(mx, slm[(k * G + g) * (D + 2) + D]);
                float L = 0.0f, A = 0.0f;
                if (mx != -INFINITY) {
                    for (int k = 0; k < NSG_A; ++k) {
                        const float* o = &slm[(k * G + g) * (D + 2)];
                        const float e = sycl::exp(o[D] - mx);
                        L += o[D + 1] * e;
                        A += o[d] * e;
                    }
                }
                const int64_t hq = hk * G + g;
                if (ns == 1) {
                    out[r * Hq * D + hq * D + d] = L > 0.0f ? A / L : 0.0f;
                } else {
                    float* p = part + ((r * Hq + hq) * ns + sp) * (D + 2);
                    p[d] = A;
                    if (d == 0) {
                        p[D] = mx;
                        p[D + 1] = L;
                    }
                }
            }
        });
    });
    if (ns > 1) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * Hq) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
            const int64_t rh = it.get_group(0), d = it.get_local_id(0);
            const float* p = part + rh * ns * (D + 2);
            float mx = -INFINITY;
            for (int64_t k = 0; k < ns; ++k) mx = sycl::fmax(mx, p[k * (D + 2) + D]);
            float L = 0.0f, A = 0.0f;
            if (mx != -INFINITY) {
                for (int64_t k = 0; k < ns; ++k) {
                    const float e = sycl::exp(p[k * (D + 2) + D] - mx);
                    L += p[k * (D + 2) + D + 1] * e;
                    A += p[k * (D + 2) + d] * e;
                }
            }
            out[rh * D + d] = L > 0.0f ? A / L : 0.0f;
        });
    }
}
}  // namespace

extern "C" {

int ns_q35_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                        const float* w, float eps, int64_t n_rot, float theta, int64_t p0) {
    NS_TRY
    if (T <= 0) return 0;
    if (D % SG != 0 || n_rot > D || n_rot % 2) return ns_fail("ns_q35_qk_norm_rope: D a multiple of 16, n_rot even and at most D");
    const int64_t half_rot = n_rot / 2;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t th = it.get_group(0), t = th / H, h = th % H;
        const float* x = src + t * ss + h * hs;
        float* y = dst + t * ds + h * D;
        float s = 0.0f;
        for (int64_t i = l; i < D; i += SG) s += x[i] * x[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) D + eps);
        const float pos = (float) (p0 + t);
        for (int64_t i = l; i < D; i += SG) {
            const float v = x[i] * inv * w[i];
            if (i < n_rot) {
                // rotate-half over the first n_rot: pair (i, i + n_rot / 2)
                const int64_t k = i < half_rot ? i : i - half_rot;
                const float a = pos * sycl::pow(theta, -2.0f * (float) k / (float) n_rot);
                const float c = sycl::cos(a), sn = sycl::sin(a);
                const int64_t o = i < half_rot ? i + half_rot : i - half_rot;
                const float other = x[o] * inv * w[o];
                y[i] = i < half_rot ? v * c - other * sn : v * c + other * sn;
            } else {
                y[i] = v;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_q35_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                    void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_q35_kv_store: past the cache's capacity");
    half* ko = (half*) kc;
    half* vo = (half*) vc;
    g->q.parallel_for(sycl::range<1>((size_t) (T * Hkv * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
        const int64_t at = (h * cap + p0 + t) * D + d;
        ko[at] = (half) k[t * ks + h * D + d];
        vo[at] = (half) v[t * vs + h * D + d];
    });
    return 0;
    NS_CATCH
}

int64_t ns_q35_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0) {
    const int64_t ns = splits(T, p0 + T, Hkv);
    return ns > 1 ? T * Hq * ns * (D + 2) : 0;
}

int ns_q35_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_TRY
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_q35_attn: past the cache's capacity");
    if (ns_q35_attn_scratch(T, Hq, Hkv, D, p0) > 0 && !scratch) return ns_fail("ns_q35_attn: no scratch");
    const half* k = (const half*) kc;
    const half* v = (const half*) vc;
    if (D == 256 && Hq == 6 * Hkv) attn<16, 6>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else if (D == 256 && Hq == 4 * Hkv) attn<16, 4>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else if (D == 256 && Hq == 2 * Hkv) attn<16, 2>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else if (D == 256 && Hq == 8 * Hkv) attn<16, 8>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else return ns_fail("ns_q35_attn: head 256 with 2, 4, 6 or 8 query heads a key head");
    return 0;
    NS_CATCH
}

int ns_q35_gate_mul(ns_gpu* g, float* out, const float* q_full, int64_t gs, int64_t T, int64_t H, int64_t D) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * H * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % H, t = i / (D * H);
        const float z = q_full[t * gs + h * 2 * D + D + d];
        out[i] *= 1.0f / (1.0f + sycl::exp(-z));
    });
    return 0;
    NS_CATCH
}

int ns_q35_gdn_gates(ns_gpu* g, const float* alpha, float* beta, const float* dt, const float* a, float* eg, int64_t T, int64_t H, int64_t d) {
    NS_TRY
    if (T <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (T * H * d)), [=](sycl::id<1> id) {
        const int64_t i = id[0], h = (i / d) % H, t = i / (d * H);
        const float x = alpha[t * H + h] + dt[h];
        // ggml's softplus: x past 20 as it is
        const float sp = x > 20.0f ? x : sycl::log1p(sycl::exp(x));
        eg[i] = sycl::exp(sp * a[h]);
    });
    g->q.parallel_for(sycl::range<1>((size_t) (T * H)), [=](sycl::id<1> i) { beta[i] = 1.0f / (1.0f + sycl::exp(-beta[i])); });
    return 0;
    NS_CATCH
}

int ns_q35_expand(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t T, int64_t Hk, int64_t Hv, int64_t d) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * Hv * d)), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % d, h = (i / d) % Hv, t = i / (d * Hv);
        dst[i] = src[t * ss + (h % Hk) * d + c];
    });
    return 0;
    NS_CATCH
}

int ns_q35_gdn_out(ns_gpu* g, const float* o, const float* z, int64_t zs, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps) {
    NS_TRY
    if (T <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t r = it.get_group(0), t = r / H, h = r % H;
        const float* x = o + r * d;
        float s = 0.0f;
        for (int64_t i = l; i < d; i += SG) s += x[i] * x[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) d + eps);
        for (int64_t i = l; i < d; i += SG) {
            const float zz = z[t * zs + h * d + i];
            y[r * d + i] = x[i] * inv * w[i] * (zz / (1.0f + sycl::exp(-zz)));
        }
    });
    return 0;
    NS_CATCH
}

int ns_q35_argmax(ns_gpu* g, const float* x, int64_t rows, int64_t n, int32_t* out) {
    NS_TRY
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        float best = -INFINITY;
        int32_t bi = 0;
        for (int64_t i = it.get_local_id(0); i < n; i += WG) {
            const float v = x[r * n + i];
            if (v > best) {
                best = v;
                bi = (int32_t) i;
            }
        }
        const float m = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
        const int32_t cand = best == m ? bi : INT32_MAX;
        const int32_t idx = sycl::reduce_over_group(it.get_group(), cand, sycl::minimum<int32_t>());
        if (it.get_local_id(0) == 0) out[r] = idx;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
