// head.cpp: the skin head's per-head bone attention (silo.h). The engine's own kernel (unirig.cpp) gives a work-group
// of 128 to a (sample, head) and lets 28 of its lanes walk a bone's 768 features each, twice: 8 ms of a 2,048-sample
// chunk. Here a sub-group of 16 takes a (sample, head), reads the query and each bone's key a float4 a lane, and adds
// the lanes' parts; the scores go through the output for the softmax.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <limits>

extern "C" {

int ns_model3d_unirig_silo_bone_attn_supported(int64_t J, int64_t D) {
    return J > 0 && J <= 4096 && D == 768 ? 1 : 0;
}

int ns_model3d_unirig_silo_bone_attn(ns_gpu* g, const float* q, const float* k, int64_t R, int64_t J, int64_t H, int64_t D, float scale, float* out) {
    NS_TRY
    if (R <= 0 || J <= 0 || H <= 0) return 0;
    if (!ns_model3d_unirig_silo_bone_attn_supported(J, D)) return ns_fail("ns_model3d_unirig_silo_bone_attn: 0 < J <= 4096, D = 768");
    constexpr int SG = 16, NSG = 8, NV = 768 / 64;   // a lane's float4s of a row
    const int64_t row = H * D, pairs = R * H;
    const int64_t groups = (pairs + NSG - 1) / NSG;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * NSG * SG)), sycl::range<1>(NSG * SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t pi = it.get_group(0) * NSG + sg.get_group_linear_id();
        if (pi >= pairs) return;
        const int64_t n = pi / H, hh = pi % H;
        sycl::float4 qv[NV];
        const float* qr = q + n * row + hh * D;
#pragma unroll
        for (int i = 0; i < NV; ++i) qv[i] = *reinterpret_cast<const sycl::float4*>(qr + 64 * i + 4 * l);
        float mx = -std::numeric_limits<float>::infinity();
        float* o = out + n * J * H + hh;
        for (int64_t j = 0; j < J; ++j) {
            const float* kr = k + j * row + hh * D;
            float p = 0.0f;
#pragma unroll
            for (int i = 0; i < NV; ++i) {
                const sycl::float4 kv = *reinterpret_cast<const sycl::float4*>(kr + 64 * i + 4 * l);
                p = sycl::fma(qv[i].x(), kv.x(), p);
                p = sycl::fma(qv[i].y(), kv.y(), p);
                p = sycl::fma(qv[i].z(), kv.z(), p);
                p = sycl::fma(qv[i].w(), kv.w(), p);
            }
            const float s = sycl::reduce_over_group(sg, p, sycl::plus<float>()) * scale;
            mx = sycl::fmax(mx, s);
            if (l == 0) o[j * H] = s;
        }
        sycl::group_barrier(sg);
        float sum = 0.0f;
        for (int64_t j = l; j < J; j += SG) {
            const float e = sycl::exp(o[j * H] - mx);
            o[j * H] = e;
            sum += e;
        }
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        const float inv = 1.0f / sum;
        for (int64_t j = l; j < J; j += SG) o[j * H] *= inv;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
