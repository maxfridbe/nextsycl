// vv.cpp: the VibeVoice engine's prompt attention (vv.h): the MiniMax Music engine's cached attention (mm3.cpp)
// copied for Qwen2.5-1.5B's six query heads a key head - the shared one keeps the shapes it was tuned for (CONTRIBUTING
// rule 6). The decode step's attention is vvlm.cpp's.
#include "ns.h"
#include "ns_internal.hpp"
#include "vv.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;

// ---- cached causal attention (mm3.cpp's): a work-group per (row, key head, split of the keys), NSG sub-groups taking
// the split's keys in turn, each lane DL features of the G query heads that share the key head; the sub-groups' running
// (max, sum, weighted values) merged in local memory, the splits (decode only) by a second kernel
constexpr int64_t KB = 256;   // keys a split (decode)
constexpr int NSG_A = 4;

int64_t splits(int64_t S, int64_t kv) {
    return S == 1 && kv > KB ? (kv + KB - 1) / KB : 1;
}

template <int DL, int G>
void attn(sycl::queue& q, const float* qp, int64_t qs, const half* kc, const half* vc, int64_t B, int64_t S, int64_t Hq, int64_t Hkv,
          int64_t T, int64_t p0, float* out, float* part) {
    constexpr int D = SG * DL, WG = SG * NSG_A;
    const int64_t ns = splits(S, p0 + S);
    const float scale = 1.0f / sycl::sqrt((float) D);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NSG_A * G * (D + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * S * Hkv * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, r = gi / Hkv, b = r / S, s = r % S;
            const int64_t kv_end = p0 + s + 1;
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
            const half* kb = kc + (b * Hkv + hk) * T * D + lane * DL;
            const half* vb = vc + (b * Hkv + hk) * T * D + lane * DL;
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
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * S * Hq) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
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

int64_t ns_audio_vv_attn_scratch(int64_t B, int64_t S, int64_t Hq, int64_t D, int64_t p0) {
    const int64_t ns = splits(S, p0 + S);
    return ns > 1 ? B * S * Hq * ns * (D + 2) : 0;
}

int ns_audio_vv_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t B, int64_t S, int64_t Hq,
                     int64_t Hkv, int64_t D, int64_t T, int64_t p0, float* out, float* part) {
    NS_TRY
    if (B <= 0 || S <= 0) return 0;
    if (p0 + S > T) return ns_fail("ns_audio_vv_attn: past the cache's capacity");
    if (ns_audio_vv_attn_scratch(B, S, Hq, D, p0) > 0 && !part) return ns_fail("ns_audio_vv_attn: no scratch");
    if (D == 128 && Hq == 6 * Hkv) attn<8, 6>(g->q, q, qs, (const half*) kc, (const half*) vc, B, S, Hq, Hkv, T, p0, out, part);
    else return ns_fail("ns_audio_vv_attn: D 128 with 6 query heads a key head");
    return 0;
    NS_CATCH
}

}  // extern "C"
