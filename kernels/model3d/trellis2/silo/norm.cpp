// norm.cpp (the trellis2 silo): the modulated layer norm a sub-group a row. The engine's own (t2.cpp) gives a row a
// work-group of 256 and two work-group reductions; at TRELLIS.2's 1536-wide rows and the decoders' millions of rows of
// 64..1024 that is most of a launch in barriers (10% of a 1024 cascade's device time). Here a row is a sub-group of 32
// lanes holding up to 64 values each in registers, reduced in the sub-group; the arithmetic is t2.cpp's (the mean, then
// the mean square of the deviations, float32).
#include "ns.h"
#include "ns_internal.hpp"
#include "t2s.h"

#include <sycl/sycl.hpp>
#include <algorithm>

namespace {
using half = sycl::half;

inline float ld(const void* p, int dt, size_t i) {
    return dt == 0 ? ((const float*) p)[i] : (float) ((const half*) p)[i];
}
inline void st(void* p, int dt, size_t i, float v) {
    if (dt == 0) ((float*) p)[i] = v;
    else ((half*) p)[i] = (half) v;
}

constexpr int SG = 32, ROWS = 8;   // a sub-group a row, 8 rows a work-group

template <int PER>
void ln_rows(sycl::queue& q, const void* x, int x_dt, int64_t r0, int64_t rows, int64_t C, float eps, const float* weight, const float* bias,
             const float* scale, const float* shift, void* out, int out_dt) {
    const size_t groups = (size_t) ((rows + ROWS - 1) / ROWS);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * ROWS * SG), sycl::range<1>(ROWS * SG)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t rr = (int64_t) it.get_group(0) * ROWS + sg.get_group_linear_id();
        if (rr >= rows) return;
        const size_t r = (size_t) (r0 + rr);
        const int l = (int) sg.get_local_linear_id();
        float v[PER];
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * SG;
            v[i] = c < C ? ld(x, x_dt, r * C + c) : 0.f;
            s += v[i];
        }
        const float mean = sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) C;
        float qq = 0.f;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * SG;
            if (c < C) qq += (v[i] - mean) * (v[i] - mean);
        }
        const float var = sycl::reduce_over_group(sg, qq, sycl::plus<float>()) / (float) C;
        const float rs = sycl::rsqrt(var + eps);
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * SG;
            if (c < C) {
                float y = (v[i] - mean) * rs;
                if (weight) y = y * weight[c] + (bias ? bias[c] : 0.f);
                if (scale) y = y * (1.f + scale[c]);
                if (shift) y += shift[c];
                st(out, out_dt, r * C + c, y);
            }
        }
    });
}
}  // namespace

extern "C" {

int ns_t2s_ln_mod_supported(int64_t C, int x_dt, int out_dt) {
    return C >= 1 && C <= 64 * SG && (x_dt == 0 || x_dt == 1) && (out_dt == 0 || out_dt == 1);
}

int ns_t2s_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias, const float* scale,
                  const float* shift, void* out, int out_dt) {
    NS_TRY
    if (!ns_t2s_ln_mod_supported(C, x_dt, out_dt)) return ns_fail("ns_t2s_ln_mod: rows of 1..2048, float32 or half");
    if (M <= 0) return 0;
    // launches of at most 2^25 rows (their work-items stay within an int)
    constexpr int64_t kRows = int64_t(1) << 25;
    for (int64_t r0 = 0; r0 < M; r0 += kRows) {
        const int64_t rows = std::min(kRows, M - r0);
        const int64_t per = (C + SG - 1) / SG;
#define T2S_LN(P) ln_rows<P>(g->q, x, x_dt, r0, rows, C, eps, weight, bias, scale, shift, out, out_dt)
        if (per <= 1) T2S_LN(1);
        else if (per <= 2) T2S_LN(2);
        else if (per <= 4) T2S_LN(4);
        else if (per <= 8) T2S_LN(8);
        else if (per <= 16) T2S_LN(16);
        else if (per <= 32) T2S_LN(32);
        else if (per <= 48) T2S_LN(48);
        else T2S_LN(64);
#undef T2S_LN
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
