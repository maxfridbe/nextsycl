// variants/b65/norm.cpp: the qwen35 silo's decode-row RMS norm (silo.h: ns_q35_rms_q8) for the B65 variant - a copy of
// ../../norm.cpp, the bits it makes the same. It is the variant path's demonstration (one op, matched by the card's
// compute units, checked against NS_SILO_VARIANT=none); a real variant changes the copy for its card.
//
// A variant's kernels live in a namespace of its own (ns_q35_v_b65): an op keeps the default silo's C name, and a SYCL
// kernel inside a function of the same name in two libraries would be two kernels of one name in the process.
#include "ns.h"
#include "ns_internal.hpp"
#include "../../silo.h"

#include <sycl/sycl.hpp>
#include "ns_q8_1.hpp"

namespace ns_q35_v_b65 {
namespace {
constexpr int WG = 256, SG = 32, NSG = WG / SG;

struct Q81 {
    sycl::half2 ds;
    int8_t qs[32];
};
}  // namespace

static int rms_q8(ns_gpu* g, const float* x, const float* w, float* y, void* q8_1, void* xs, int64_t rows, int64_t C, float eps) {
    NS_TRY
    if (C % WG != 0) return ns_fail("ns_q35_rms_q8: rows of a multiple of 256");
    Q81* q = static_cast<Q81*>(q8_1);
    int8_t* xq = static_cast<int8_t*>(xs);   // the split layout (nullable): quants [rows][C], then scales [rows][C / 32]
    sycl::half* xd = xq ? reinterpret_cast<sycl::half*>(xq + rows * C) : nullptr;
    int16_t* xsm = xq ? reinterpret_cast<int16_t*>(xd + rows * C / 32) : nullptr;   // the 16-value groups' sums
    const int64_t slices = C / WG;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(NSG), h);
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
            const float d = ns::q8_1_finite(amax / 127.0f);
            Q81* b = q + (r * C + c) / 32;
            const int8_t qv = ns::q8_1_quant(v, d, amax);
            const sycl::half2 ds = ns::q8_1_ds(d, sum);
            b->qs[lane] = qv;
            if (lane == 0) b->ds = ds;
            int t16 = qv;   // the 16-value group's sum (lanes 0-15, 16-31), every lane in the shuffles
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) t16 += sycl::permute_group_by_xor(sg, t16, o);
            if (xq) {
                xq[r * C + c] = qv;
                if (lane == 0) xd[(r * C + c) / 32] = ds[0];
                if (lane % 16 == 0) xsm[(r * C + c) / 16] = (int16_t) t16;
            }
        });
    });
    return 0;
    NS_CATCH
}

}  // namespace ns_q35_v_b65

extern "C" int ns_q35_rms_q8(ns_gpu* g, const float* x, const float* w, float* y, void* q8_1, void* xs, int64_t rows, int64_t C, float eps) {
    return ns_q35_v_b65::rms_q8(g, x, w, y, q8_1, xs, rows, C, eps);
}
