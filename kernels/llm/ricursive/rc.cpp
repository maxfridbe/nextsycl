// rc.cpp: the ricursive engine's own kernels (rc.h) - Ricursive Ripple-65B's attention: per-head q / k norms with
// rotary positions on the sliding-window layers only, the keys in a ring of the window's size, a window mask, the
// elementwise sigmoid gate and the value residual. After the reference's modeling_ricursive.py. The cached attention
// is the qwen35 engine's (q35.cpp) with a window and a ring: a work-group per (row, key head, split of the keys),
// NSG_A sub-groups taking the split's keys in turn, each lane DL features of the G query heads that share the key
// head; the sub-groups' (max, sum, weighted values) merged in local memory, the splits (decode) by a second kernel.
#include "ns.h"
#include "ns_internal.hpp"
#include "rc.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;
constexpr int NSG_A = 4;

// the most keys any row of this call sees: its window, or all so far
inline int64_t keys_max(int64_t S, int64_t W, int64_t p0) {
    const int64_t all = p0 + S;
    return W > 0 && W < all ? W : all;
}
// a decode row's keys a split: enough splits that the card has ~512 work-groups (key heads x splits), at least 32 keys
// each; prompt rows one split
int64_t split_keys(int64_t S, int64_t kv, int64_t Hkv) {
    if (S > 8) return kv;
    const int64_t want = (kv * Hkv + 511) / 512;
    return sycl::max<int64_t>(32, (want + 31) / 32 * 32);
}
int64_t splits(int64_t S, int64_t kv, int64_t Hkv) {
    const int64_t kb = split_keys(S, kv, Hkv);
    return kv > kb ? (kv + kb - 1) / kb : 1;
}

template <int DL, int G>
void attn(sycl::queue& q, const float* qp, int64_t qs, const half* kc, const half* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t R, int64_t W,
          int64_t p0, float* out, float* part) {
    constexpr int D = SG * DL, WG = SG * NSG_A;
    const int64_t kvm = keys_max(S, W, p0);
    const int64_t ns = splits(S, kvm, Hkv), KB = split_keys(S, kvm, Hkv);
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
            const int64_t lo = (W > 0 && kv_end > W) ? kv_end - W : 0;
            const int64_t j0 = lo + sp * (ns > 1 ? KB : kv_end), j1 = sycl::min(kv_end, ns > 1 ? lo + (sp + 1) * KB : kv_end);
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
            const half* kb = kc + hk * R * D + lane * DL;
            const half* vb = vc + hk * R * D + lane * DL;
            for (int64_t j = j0 + sgi; j < j1; j += NSG_A) {
                const int64_t slot = j % R;
                const sycl::vec<half, DL> kv = *reinterpret_cast<const sycl::vec<half, DL>*>(kb + slot * D);
                const sycl::vec<half, DL> vv = *reinterpret_cast<const sycl::vec<half, DL>*>(vb + slot * D);
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

int ns_rc_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                       const float* w, float eps, int rope, float theta, int64_t p0) {
    NS_TRY
    if (T <= 0) return 0;
    if (D % SG != 0 || D % 2) return ns_fail("ns_rc_qk_norm_rope: D a multiple of 16");
    const int64_t half_rot = D / 2;
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
            if (rope) {
                const int64_t k = i < half_rot ? i : i - half_rot;
                const float a = pos * sycl::pow(theta, -2.0f * (float) k / (float) D);
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

int ns_rc_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t R, int64_t p0,
                   void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    if (T > R) return ns_fail("ns_rc_kv_store: more rows than the ring holds");
    half* ko = (half*) kc;
    half* vo = (half*) vc;
    g->q.parallel_for(sycl::range<1>((size_t) (T * Hkv * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
        const int64_t at = (h * R + (p0 + t) % R) * D + d;
        ko[at] = (half) k[t * ks + h * D + d];
        vo[at] = (half) v[t * vs + h * D + d];
    });
    return 0;
    NS_CATCH
}

int64_t ns_rc_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t W, int64_t p0) {
    const int64_t ns = splits(T, keys_max(T, W, p0), Hkv);
    return ns > 1 ? T * Hq * ns * (D + 2) : 0;
}

int ns_rc_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t R,
               int64_t W, int64_t p0, float* out, float* scratch) {
    NS_TRY
    if (T <= 0) return 0;
    if (W > 0 && R < W + T - 1 && p0 + T > R) return ns_fail("ns_rc_attn: the ring is smaller than the window and the rows");
    if (W == 0 && p0 + T > R) return ns_fail("ns_rc_attn: past the cache's capacity");
    if (ns_rc_attn_scratch(T, Hq, Hkv, D, W, p0) > 0 && !scratch) return ns_fail("ns_rc_attn: no scratch");
    if (D != 128) return ns_fail("ns_rc_attn: heads of 128");
    const half* k = (const half*) kc;
    const half* v = (const half*) vc;
    if (Hq == 10 * Hkv) attn<8, 10>(g->q, q, qs, k, v, T, Hq, Hkv, R, W, p0, out, scratch);
    else if (Hq == 8 * Hkv) attn<8, 8>(g->q, q, qs, k, v, T, Hq, Hkv, R, W, p0, out, scratch);
    else if (Hq == 6 * Hkv) attn<8, 6>(g->q, q, qs, k, v, T, Hq, Hkv, R, W, p0, out, scratch);
    else if (Hq == 4 * Hkv) attn<8, 4>(g->q, q, qs, k, v, T, Hq, Hkv, R, W, p0, out, scratch);
    else if (Hq == 2 * Hkv) attn<8, 2>(g->q, q, qs, k, v, T, Hq, Hkv, R, W, p0, out, scratch);
    else if (Hq == Hkv) attn<8, 1>(g->q, q, qs, k, v, T, Hq, Hkv, R, W, p0, out, scratch);
    else return ns_fail("ns_rc_attn: 1, 2, 4, 6, 8 or 10 query heads a key head");
    return 0;
    NS_CATCH
}

int ns_rc_gate_mul(ns_gpu* g, float* out, const float* gate, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        out[i] *= 1.0f / (1.0f + sycl::exp(-gate[i]));
    });
    return 0;
    NS_CATCH
}

int ns_rc_scale(ns_gpu* g, float* x, int64_t n, float s) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { x[id[0]] *= s; });
    return 0;
    NS_CATCH
}

int ns_rc_vres(ns_gpu* g, const float* x0, const float* h, float* y, int64_t n, float s) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        y[i] = x0[i] + s * (h[i] - x0[i]);
    });
    return 0;
    NS_CATCH
}

int ns_rc_norm_add(ns_gpu* g, float* y, const float* x, const float* w, int64_t rows, int64_t C, float eps) {
    NS_TRY
    if (rows <= 0) return 0;
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const float* xr = x + r * C;
        float* yr = y + r * C;
        float s = 0.0f;
        for (int64_t i = it.get_local_id(0); i < C; i += WG) s += xr[i] * xr[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C + eps);
        for (int64_t i = it.get_local_id(0); i < C; i += WG) yr[i] += xr[i] * inv * w[i];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
