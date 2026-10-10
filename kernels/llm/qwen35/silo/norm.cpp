// norm.cpp: the qwen35 silo's RMS norm for decode rows (silo.h: ns_q35_rms_q8) - the norm and the Q8_1 blocks the
// decode products read, in one launch. The shared rms_norm took 10.7 us a 5,120 row on the B70 (a 256-wide reduction
// over the work-group) and each product then quantized the same row again (1.7-1.9 us, 3-4 times a norm). The
// quantization is Strata's (quantize_q8_1: the same scale, rounding and finite clamp - its header), so the blocks are
// the bits the products got before.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include "strata/kernels/q8_1_finite.hpp"

namespace {
constexpr int WG = 256, SG = 32, NSG = WG / SG;

struct Q81 {
    sycl::half2 ds;
    int8_t qs[32];
};
}  // namespace

extern "C" {

int ns_q35_rms_q8(ns_gpu* g, const float* x, const float* w, float* y, void* q8_1, int64_t rows, int64_t C, float eps) {
    NS_TRY
    if (C % WG != 0) return ns_fail("ns_q35_rms_q8: rows of a multiple of 256");
    Q81* q = static_cast<Q81*>(q8_1);
    const int64_t slices = C / WG;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(NSG), h);
        // a work-group a (row, slice of 256 values): each sums the whole row's squares (20 KB from the cache) and
        // quantizes its slice's 8 blocks - 20 work-groups a 5,120 row, not one walking it (22 us)
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (rows * slices) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int tid = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int64_t r = it.get_group(0) / slices, c = (it.get_group(0) % slices) * WG + tid;
            const float* xr = x + r * C;
            float s = 0.f;
            for (int64_t i = tid; i < C; i += WG) s += xr[i] * xr[i];
            s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
            if (lane == 0) part[sgi] = s;
            sycl::group_barrier(it.get_group());
            float t = 0.f;
#pragma unroll
            for (int i = 0; i < NSG; ++i) t += part[i];
            const float inv = 1.f / sycl::sqrt(t / (float) C + eps);
            const float v = xr[c] * inv * (w ? w[c] : 1.f);
            if (y) y[r * C + c] = v;
            float amax = sycl::fabs(v), sum = v;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
                sum += sycl::permute_group_by_xor(sg, sum, o);
            }
            const float d = strata::kernels::q8_1_finite(amax / 127.0f);
            Q81* b = q + (r * C + c) / 32;
            b->qs[lane] = strata::kernels::q8_1_quant(v, d, amax);
            if (lane == 0) b->ds = strata::kernels::q8_1_ds(d, sum);
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
