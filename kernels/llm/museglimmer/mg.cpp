// mg.cpp: the museglimmer engine's own kernels (mg.h) - Muse Glimmer's attention: q / k norms with rotary positions on
// the sliding-window layers (llama.cpp's NORM rotation: adjacent pairs), ring caches, the causal attention over a
// window, the sigmoid output gate. After llama.cpp's muse-glimmer graph (src/models/muse-glimmer.cpp).
#include "ns.h"
#include "ns_internal.hpp"
#include "mg.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;

// ---- cached causal attention (the qwen35 engine's, q35.cpp, with a window and ring slots): a work-group per (row,
// group of G query heads, split of the keys), NSG sub-groups taking the split's keys in turn, each lane DL features of
// the G heads; the sub-groups' (max, sum, weighted values) merged in local memory, the splits (decode) by a second
// kernel
constexpr int NSG_A = 4;

// a decode row's keys a split: ~512 work-groups whatever the context, at least 32 keys each; prompt rows one split
int64_t split_keys(int64_t S, int64_t kv, int64_t groups) {
    if (S != 1) return kv;
    const int64_t want = (kv * groups + 511) / 512;
    return sycl::max<int64_t>(32, (want + 31) / 32 * 32);
}
// the keys the last row sees
int64_t seen(int64_t S, int64_t p0, int64_t win) {
    const int64_t end = p0 + S;
    return win > 0 ? sycl::min(end, win) : end;
}
int64_t splits(int64_t S, int64_t kv, int64_t groups) {
    const int64_t kb = split_keys(S, kv, groups);
    return kv > kb ? (kv + kb - 1) / kb : 1;
}

template <int DL, int G>
void attn(sycl::queue& q, const float* qp, const half* kc, const half* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win,
          float* out, float* part) {
    constexpr int D = SG * DL, WG = SG * NSG_A;
    const int64_t groups = Hq / G, per_kv = Hq / Hkv;
    const int64_t kvn = seen(S, p0, win);
    const int64_t ns = splits(S, kvn, groups), KB = split_keys(S, kvn, groups);
    const float scale = 1.0f / sycl::sqrt((float) D);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NSG_A * G * (D + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * groups * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hg = gi % groups, r = gi / groups;
            const int64_t hk = hg * G / per_kv;
            // positions this row sees: [first, end)
            const int64_t end = p0 + r + 1, first = win > 0 ? sycl::max<int64_t>(0, end - win) : 0;
            const int64_t n = end - first;
            const int64_t j0 = first + sp * (ns > 1 ? KB : n), j1 = sycl::min(end, ns > 1 ? first + (sp + 1) * KB : end);
            float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float* qr = qp + r * Hq * D + (hg * G + g) * D + lane * DL;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    qv[g][i] = qr[i] * scale;
                    acc[g][i] = 0.0f;
                }
                m[g] = -INFINITY;
                l[g] = 0.0f;
            }
            const half* kb = kc + hk * cap * D + lane * DL;
            const half* vb = vc + hk * cap * D + lane * DL;
            for (int64_t j = j0 + sgi; j < j1; j += NSG_A) {
                const int64_t slot = j % cap;
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
                const int64_t hq = hg * G + g;
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

// query heads a work-group: 4 (G 16 would hold 16 heads' queries and sums in registers)
constexpr int GQ = 4;
}  // namespace

extern "C" {

int ns_mg_qk_norm_rope(ns_gpu* g, const float* x, int64_t xs, float* y, int64_t ys, int64_t T, int64_t H, int64_t D, const float* w, float eps,
                       int rope, float theta, int64_t p0) {
    NS_TRY
    if (T <= 0) return 0;
    if (D % (2 * SG) != 0) return ns_fail("ns_mg_qk_norm_rope: D a multiple of 32");
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t th = it.get_group(0), t = th / H, h = th % H;
        const float* xr = x + t * xs + h * D;
        float* yr = y + t * ys + h * D;
        float s = 0.0f;
        for (int64_t i = l; i < D; i += SG) s += xr[i] * xr[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) D + eps);
        const float pos = (float) (p0 + t);
        // a lane a pair (2i, 2i + 1) at a time
        for (int64_t i = l; i < D / 2; i += SG) {
            const float a0 = xr[2 * i] * inv * w[2 * i], a1 = xr[2 * i + 1] * inv * w[2 * i + 1];
            if (rope) {
                const float a = pos * sycl::pow(theta, -2.0f * (float) i / (float) D);
                const float c = sycl::cos(a), sn = sycl::sin(a);
                yr[2 * i] = a0 * c - a1 * sn;
                yr[2 * i + 1] = a0 * sn + a1 * c;
            } else {
                yr[2 * i] = a0;
                yr[2 * i + 1] = a1;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_mg_kv_store(ns_gpu* g, const float* k, const float* v, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    if (T > cap) return ns_fail("ns_mg_kv_store: more rows than the cache's slots");
    half* ko = (half*) kc;
    half* vo = (half*) vc;
    g->q.parallel_for(sycl::range<1>((size_t) (T * Hkv * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
        const int64_t at = (h * cap + (p0 + t) % cap) * D + d;
        ko[at] = (half) k[t * Hkv * D + h * D + d];
        vo[at] = (half) v[t * Hkv * D + h * D + d];
    });
    return 0;
    NS_CATCH
}

int64_t ns_mg_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0, int64_t win) {
    const int64_t ns = splits(T, seen(T, p0, win), Hq / GQ);
    return ns > 1 ? T * Hq * ns * (D + 2) : 0;
}

int ns_mg_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
               int64_t win, float* out, float* scratch) {
    NS_TRY
    if (T <= 0) return 0;
    if (D != 128 || Hq % Hkv != 0 || (Hq / Hkv) % GQ != 0) return ns_fail("ns_mg_attn: head 128 with a multiple of 4 query heads a key head");
    // the ring holds the oldest position the first row sees and the newest the last writes
    const int64_t need = win > 0 ? sycl::min(p0 + T, win + T - 1) : p0 + T;
    if (need > cap) return ns_fail("ns_mg_attn: the cache's ring is shorter than the window and the pass");
    if (ns_mg_attn_scratch(T, Hq, Hkv, D, p0, win) > 0 && !scratch) return ns_fail("ns_mg_attn: no scratch");
    attn<8, GQ>(g->q, q, (const half*) kc, (const half*) vc, T, Hq, Hkv, cap, p0, win, out, scratch);
    return 0;
    NS_CATCH
}

int ns_mg_gate_mul(ns_gpu* g, float* x, const float* gate, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        x[i] *= 1.0f / (1.0f + sycl::exp(-gate[i]));
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
