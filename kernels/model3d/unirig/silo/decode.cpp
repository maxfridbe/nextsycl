// decode.cpp: the skeleton decoder's step attention over the beams' slot histories (silo.h), for its 64-wide heads.
// The engine's own kernel (unirig.cpp) gives a work-group to a (beam, head) and lets a quarter of it walk all ~1,100
// keys one after another for the weighted values: two thirds of a decode step. Here every sub-group of 16 reads whole
// 64-float rows at once (a float4 a lane): the scores a row each, then the weighted values of its share of the rows,
// the sub-groups' sums added at the end.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <limits>

extern "C" {

int ns_model3d_unirig_silo_attn_slots_supported(int64_t D, int64_t len) {
    return D == 64 && len > 0 && len <= 8192 ? 1 : 0;
}

int ns_model3d_unirig_silo_attn_slots(ns_gpu* g, const float* q, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride,
                                      int64_t len, int64_t B, int64_t H, int64_t D, float scale, float* out) {
    NS_TRY
    if (len <= 0 || B <= 0 || H <= 0) return 0;
    if (!ns_model3d_unirig_silo_attn_slots_supported(D, len)) return ns_fail("ns_model3d_unirig_silo_attn_slots: D = 64, 0 < len <= 8192");
    constexpr int SG = 16, NSG = 16, WG = SG * NSG, DH = 64;
    const int64_t row = H * DH;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>((size_t) len), h);
        sycl::local_accessor<sycl::float4, 1> part(sycl::range<1>(NSG * SG), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * H * WG)), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t gi = it.get_group(0), b = gi / H, hh = gi % H;
            const auto sg = it.get_sub_group();
            const int s = (int) sg.get_group_linear_id(), l = (int) sg.get_local_linear_id();
            const int32_t* st = slots + b * stride;
            const sycl::float4 qv = *reinterpret_cast<const sycl::float4*>(q + b * row + hh * DH + 4 * l);
            const int64_t col = hh * DH + 4 * l;
            // the scores: a row a sub-group
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t t = s; t < len; t += NSG) {
                const sycl::float4 kv = *reinterpret_cast<const sycl::float4*>(kpool + (int64_t) st[t] * row + col);
                float p = qv.x() * kv.x();
                p = sycl::fma(qv.y(), kv.y(), p);
                p = sycl::fma(qv.z(), kv.z(), p);
                p = sycl::fma(qv.w(), kv.w(), p);
                const float sv = sycl::reduce_over_group(sg, p, sycl::plus<float>()) * scale;
                if (l == 0) sc[t] = sv;
                mx = sycl::fmax(mx, sv);
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            sycl::group_barrier(it.get_group());     // every score written
            const int lid = (int) it.get_local_id(0);
            float sum = 0.0f;
            for (int64_t t = lid; t < len; t += WG) {
                const float e = sycl::exp(sc[t] - mx);
                sc[t] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            sycl::group_barrier(it.get_group());     // every weight written
            // the weighted values: each sub-group its rows, a lane its 4 columns
            sycl::float4 acc(0.0f);
            for (int64_t t = s; t < len; t += NSG) {
                const sycl::float4 vv = *reinterpret_cast<const sycl::float4*>(vpool + (int64_t) st[t] * row + col);
                acc += sc[t] * vv;
            }
            part[s * SG + l] = acc;
            sycl::group_barrier(it.get_group());
            if (s == 0) {
                sycl::float4 tot = part[l];
                for (int k = 1; k < NSG; ++k) tot += part[k * SG + l];
                *reinterpret_cast<sycl::float4*>(out + b * row + col) = tot * (1.0f / sum);
            }
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
