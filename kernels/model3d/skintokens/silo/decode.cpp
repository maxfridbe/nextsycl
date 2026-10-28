// decode.cpp: SkinTokens' decode step's attention over the beams' slot pool in the silo (silo.h). The engine's own kernel
// (skintokens.cpp) gives a work-group to every (beam, query head) and a thread to every key: neighbouring threads read rows
// 4 KB apart, the grouped heads read the same keys twice, and the weighted values run on half the threads - 1.6 s of a
// 2.4 s skeleton on the beams' 600 positions. Here a work-group takes a (beam, key / value head) with all the query heads that
// share it; a sub-group of 16 reads a key row whole (a lane 8 floats of its 128: one contiguous 512 bytes), the dot products
// reduce in the sub-group; the weighted values run a thread a dimension over rows that are contiguous across the threads.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
constexpr int SG = 16, NSG = 8, WG = SG * NSG, D = 128, MAXL = 4096;
static_assert(WG == D, "the weighted values run a thread a dimension");

template <int G>
void run(sycl::queue& q, const float* qv, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride, int64_t len,
         int64_t B, int64_t Hkv, float scale, float* out) {
    const int64_t H = Hkv * G, row = Hkv * D;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(G * MAXL), h);
        sycl::local_accessor<int32_t, 1> sl(sycl::range<1>(MAXL), h);
        sycl::local_accessor<float, 1> inv(sycl::range<1>(G), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * Hkv * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            const int s = (int) sg.get_group_linear_id();
            const int l = (int) it.get_local_id(0);
            const int64_t gi = it.get_group(0), b = gi / Hkv, hk = gi % Hkv;
            const int32_t* st = slots + b * stride;
            for (int64_t t = l; t < len; t += WG) sl[t] = st[t];
            sycl::group_barrier(it.get_group());
            // the scores: a sub-group a key, its lanes 8 values each
            float qr[G][8];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float* qp = qv + b * q_row + (hk * G + g) * D + lane * 8;
                const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(qp);
                const sycl::float4 c = *reinterpret_cast<const sycl::float4*>(qp + 4);
                qr[g][0] = a.x(); qr[g][1] = a.y(); qr[g][2] = a.z(); qr[g][3] = a.w();
                qr[g][4] = c.x(); qr[g][5] = c.y(); qr[g][6] = c.z(); qr[g][7] = c.w();
            }
            // four keys at a time: their loads are independent, so the memory latency overlaps
            constexpr int U = 4;
            for (int64_t t0 = s; t0 < len; t0 += (int64_t) NSG * U) {
                float kk[U][8];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int64_t t = t0 + (int64_t) u * NSG;
                    const float* kr = kpool + (int64_t) sl[t < len ? t : 0] * row + hk * D + lane * 8;
                    const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(kr);
                    const sycl::float4 c = *reinterpret_cast<const sycl::float4*>(kr + 4);
                    kk[u][0] = a.x(); kk[u][1] = a.y(); kk[u][2] = a.z(); kk[u][3] = a.w();
                    kk[u][4] = c.x(); kk[u][5] = c.y(); kk[u][6] = c.z(); kk[u][7] = c.w();
                }
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int64_t t = t0 + (int64_t) u * NSG;
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        float d = 0.0f;
#pragma unroll
                        for (int i = 0; i < 8; ++i) d = sycl::fma(qr[g][i], kk[u][i], d);
                        d = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                        if (lane == 0 && t < len) sc[g * MAXL + t] = d * scale;
                    }
                }
            }
            sycl::group_barrier(it.get_group());
            // the softmax of each head's scores
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float mx = -std::numeric_limits<float>::infinity();
                for (int64_t t = l; t < len; t += WG) mx = sycl::fmax(mx, sc[g * MAXL + t]);
                mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
                float sum = 0.0f;
                for (int64_t t = l; t < len; t += WG) {
                    const float e = sycl::exp(sc[g * MAXL + t] - mx);
                    sc[g * MAXL + t] = e;
                    sum += e;
                }
                sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                if (l == 0) inv[g] = 1.0f / sum;
            }
            sycl::group_barrier(it.get_group());
            // the weighted values: a thread a dimension (WG = 128 = D), the rows contiguous across the threads
            float acc[G];
#pragma unroll
            for (int g = 0; g < G; ++g) acc[g] = 0.0f;
            const float* vb = vpool + hk * D + l;
            int64_t t = 0;
            for (; t + 8 <= len; t += 8) {
                float v[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) v[u] = vb[(int64_t) sl[t + u] * row];
#pragma unroll
                for (int u = 0; u < 8; ++u)
#pragma unroll
                    for (int g = 0; g < G; ++g) acc[g] = sycl::fma(sc[g * MAXL + t + u], v[u], acc[g]);
            }
            for (; t < len; ++t) {
                const float v = vb[(int64_t) sl[t] * row];
#pragma unroll
                for (int g = 0; g < G; ++g) acc[g] = sycl::fma(sc[g * MAXL + t], v, acc[g]);
            }
#pragma unroll
            for (int g = 0; g < G; ++g) out[(b * H + hk * G + g) * D + l] = acc[g] * inv[g];
        });
    });
}
}  // namespace

extern "C" {

int ns_model3d_skintokens_silo_attn_slots_supported(int64_t D_, int64_t len, int64_t H, int64_t Hkv) {
    return D_ == D && len > 0 && len <= MAXL && Hkv > 0 && (H == Hkv || H == 2 * Hkv) ? 1 : 0;
}

int ns_model3d_skintokens_silo_attn_slots(ns_gpu* g, const float* q, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride,
                                          int64_t len, int64_t B, int64_t H, int64_t Hkv, int64_t D_, float scale, float* out) {
    NS_TRY
    if (B <= 0 || len <= 0) return 0;
    if (!ns_model3d_skintokens_silo_attn_slots_supported(D_, len, H, Hkv)) return ns_fail("ns_model3d_skintokens_silo_attn_slots: shapes it does not take");
    if (((uintptr_t) q & 15) || q_row % 4 || ((uintptr_t) kpool & 15) || ((uintptr_t) vpool & 15)) return ns_fail("ns_model3d_skintokens_silo_attn_slots: float4-aligned rows needed");
    if (H == Hkv) run<1>(g->q, q, q_row, kpool, vpool, slots, stride, len, B, Hkv, scale, out);
    else run<2>(g->q, q, q_row, kpool, vpool, slots, stride, len, B, Hkv, scale, out);
    return 0;
    NS_CATCH
}

}  // extern "C"
