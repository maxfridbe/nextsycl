// gemv.cpp: SkinTokens' decode step's linear layers in the silo (silo.h): out [M, N] = x [M, K] . W^T for M at most 16 rows
// (the beams) with W held as bfloat16. TokenRig's weights are stored in bfloat16, so the bf16 copy is the checkpoint's own
// values and the product (float32 activations, float32 accumulation) differs from the float32 oneDNN product only in the order
// of the sums. At M = 10 the step is bound by reading the weights: half the bytes, one pass, no oneDNN dispatch.
//
// A work-group of NSG sub-groups of 16 takes R output rows; each sub-group a quarter of K, a lane 8 values (16 bytes of a
// row) a time for all M activation rows; the lanes' sums reduce in the sub-group, the sub-groups' through local memory.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>

namespace {
constexpr int SG = 16, NSG = 4, VEC = 8;

inline float bf(uint32_t half_bits) {
    return sycl::bit_cast<float>(half_bits << 16);
}

// MT: the activation rows padded up (M of them real), R: output rows a work-group
template <int MT, int R>
void run(sycl::queue& q, const float* x, const uint16_t* w, float* out, int64_t M, int64_t N, int64_t K) {
    const int64_t groups = (N + R - 1) / R;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(NSG * R * MT), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * NSG * SG)), sycl::range<1>(NSG * SG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            const int s = (int) sg.get_group_linear_id();
            const int64_t n0 = (int64_t) it.get_group(0) * R;
            const int64_t nv = K / VEC;
            const int64_t per = (nv + NSG - 1) / NSG;
            const int64_t v0 = s * per, v1 = std::min<int64_t>(nv, v0 + per);
            float acc[R][MT];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int m = 0; m < MT; ++m) acc[r][m] = 0.0f;
            for (int64_t v = v0 + lane; v < v1; v += SG) {
                float wv[R][VEC];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int64_t row = std::min<int64_t>(n0 + r, N - 1);
                    const sycl::uint4 u = *reinterpret_cast<const sycl::uint4*>(w + row * K + v * VEC);
                    const uint32_t p[4] = {u.x(), u.y(), u.z(), u.w()};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        wv[r][2 * i] = bf(p[i] & 0xffffu);
                        wv[r][2 * i + 1] = bf(p[i] >> 16);
                    }
                }
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    float xv[VEC];
                    if (m < M) {
                        const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(x + m * K + v * VEC);
                        const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(x + m * K + v * VEC + 4);
                        xv[0] = a.x(); xv[1] = a.y(); xv[2] = a.z(); xv[3] = a.w();
                        xv[4] = b.x(); xv[5] = b.y(); xv[6] = b.z(); xv[7] = b.w();
                    } else {
#pragma unroll
                        for (int i = 0; i < VEC; ++i) xv[i] = 0.0f;
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int i = 0; i < VEC; ++i) acc[r][m] = sycl::fma(wv[r][i], xv[i], acc[r][m]);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const float t = sycl::reduce_over_group(sg, acc[r][m], sycl::plus<float>());
                    if (lane == 0) red[(s * R + r) * MT + m] = t;
                }
            sycl::group_barrier(it.get_group());
            if (s == 0) {
                for (int i = lane; i < R * MT; i += SG) {
                    const int r = i / MT, m = i % MT;
                    float t = 0.0f;
#pragma unroll
                    for (int j = 0; j < NSG; ++j) t += red[j * R * MT + i];
                    if (m < M && n0 + r < N) out[m * N + n0 + r] = t;
                }
            }
        });
    });
}

template <int R>
void dispatch(sycl::queue& q, const float* x, const uint16_t* w, float* out, int64_t M, int64_t N, int64_t K) {
    if (M <= 1) run<1, R>(q, x, w, out, M, N, K);
    else if (M <= 2) run<2, R>(q, x, w, out, M, N, K);
    else if (M <= 4) run<4, R>(q, x, w, out, M, N, K);
    else if (M <= 8) run<8, R>(q, x, w, out, M, N, K);
    else if (M <= 12) run<12, R>(q, x, w, out, M, N, K);
    else run<16, R>(q, x, w, out, M, N, K);
}
}  // namespace

extern "C" {

int ns_model3d_skintokens_silo_gemv_supported(int64_t M, int64_t N, int64_t K) {
    return M >= 1 && M <= 16 && N > 0 && K > 0 && K % VEC == 0 ? 1 : 0;
}

int ns_model3d_skintokens_silo_gemv(ns_gpu* g, const float* x, const uint16_t* w, int64_t M, int64_t N, int64_t K, float* out) {
    NS_TRY
    if (!ns_model3d_skintokens_silo_gemv_supported(M, N, K)) return ns_fail("ns_model3d_skintokens_silo_gemv: shapes it does not take");
    if (((uintptr_t) x & 15) || ((uintptr_t) w & 15)) return ns_fail("ns_model3d_skintokens_silo_gemv: x and w must be 16-byte aligned");
    if (N >= 4096) dispatch<2>(g->q, x, w, out, M, N, K);
    else dispatch<1>(g->q, x, w, out, M, N, K);
    return 0;
    NS_CATCH
}

}  // extern "C"
