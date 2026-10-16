// norm.cpp: layer norm, and layer norm then GELU (silo.h). The engine's own kernel (unirig.cpp) gives a work-group of
// 256 to a row: the skin head's 57,344 rows of 512 (and of 16) a chunk leave most of it idle, and its GELU is a pass of
// its own. Here a sub-group of 16 takes a row (its values in registers, a lane every 16th) and the GELU is applied on
// the way out.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

namespace {
inline float gelu_erf(float v) {
    return 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
}

template <int PER>
void ln_rows(sycl::queue& q, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, const float* b, int act, float* out) {
    constexpr int SG = 16, NSG = 8;
    const int64_t groups = (M + NSG - 1) / NSG;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * NSG * SG)), sycl::range<1>(NSG * SG)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t r = it.get_group(0) * NSG + sg.get_group_linear_id();
        if (r >= M) return;
        const float* row = x + r * x_row;
        float v[PER];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * SG + l;
            v[j] = i < C ? row[i] : 0.0f;
            s += v[j];
        }
        const float mean = sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) C;
        float s2 = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * SG + l;
            const float d = i < C ? v[j] - mean : 0.0f;
            s2 += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s2, sycl::plus<float>()) / (float) C + eps);
        // every lane has read its part of the row before any writes (out may be x)
        sycl::group_barrier(sg);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * SG + l;
            if (i < C) {
                float y = (v[j] - mean) * inv;
                if (w) y = y * w[i];
                if (b) y = y + b[i];
                if (act == 0) y = gelu_erf(y);
                out[r * C + i] = y;
            }
        }
    });
}
}  // namespace

extern "C" {

int ns_model3d_unirig_silo_ln_supported(int64_t C) {
    return C > 0 && C <= 1024 ? 1 : 0;
}

int ns_model3d_unirig_silo_ln(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, const float* b, int act,
                              float* out) {
    NS_TRY
    if (!ns_model3d_unirig_silo_ln_supported(C)) return ns_fail("ns_model3d_unirig_silo_ln: 0 < C <= 1024");
    if (act != -1 && act != 0) return ns_fail("ns_model3d_unirig_silo_ln: act -1 (none) or 0 (GELU)");
    if (M <= 0) return 0;
    if (C <= 16) ln_rows<1>(g->q, x, x_row, M, C, eps, w, b, act, out);
    else if (C <= 64) ln_rows<4>(g->q, x, x_row, M, C, eps, w, b, act, out);
    else if (C <= 128) ln_rows<8>(g->q, x, x_row, M, C, eps, w, b, act, out);
    else if (C <= 256) ln_rows<16>(g->q, x, x_row, M, C, eps, w, b, act, out);
    else if (C <= 512) ln_rows<32>(g->q, x, x_row, M, C, eps, w, b, act, out);
    else ln_rows<64>(g->q, x, x_row, M, C, eps, w, b, act, out);
    return 0;
    NS_CATCH
}

}  // extern "C"
