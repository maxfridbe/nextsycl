// q8.cpp: YuE2's silo: the repacked Q8_0 matrices (silo.h). The packing and the expansion to half start as
// museglimmer's (kernels/llm/museglimmer/silo/mmvq.cpp), the product as the engine's ns_audio_yue2_gemv_q8: the same sums
// in the same order, only the weights' loads differ, so a result is the engine's kernel's.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <cstdint>

namespace {
using half = sycl::half;
constexpr int SG = 16, NSG = 8, R = 2, MM = 8;
inline size_t align256(size_t n) { return (n + 255) / 256 * 256; }

void gemv(sycl::queue& q, const float* x, int M, int64_t K, const int8_t* wq, const half* ws, int64_t N, float* out, int64_t ldo, bool acc) {
    const int64_t groups = (N + NSG * R - 1) / (NSG * R);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * NSG * SG), sycl::range<1>(NSG * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
        const int64_t n0 = (it.get_group(0) * NSG + sgi) * R;
        const int64_t nb = K / 32;
        float a[MM][R];
#pragma unroll
        for (int m = 0; m < MM; ++m)
#pragma unroll
            for (int r = 0; r < R; ++r) a[m][r] = 0.0f;
        for (int64_t b = lane; b < nb; b += SG) {
            float wf[R][32];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int64_t n = n0 + r < N ? n0 + r : N - 1;
                const int8_t* p = wq + n * K + b * 32;
                const float d = (float) ws[n * nb + b];
                const sycl::vec<int8_t, 16> v0 = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(p);
                const sycl::vec<int8_t, 16> v1 = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(p + 16);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    wf[r][i] = d * (float) v0[i];
                    wf[r][16 + i] = d * (float) v1[i];
                }
            }
#pragma unroll
            for (int m = 0; m < MM; ++m) {
                if (m < M) {
                    const sycl::float4* xp = reinterpret_cast<const sycl::float4*>(x + m * K + b * 32);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const sycl::float4 xv = xp[i];
#pragma unroll
                        for (int r = 0; r < R; ++r)
                            a[m][r] += wf[r][4 * i] * xv.x() + wf[r][4 * i + 1] * xv.y() + wf[r][4 * i + 2] * xv.z() + wf[r][4 * i + 3] * xv.w();
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            if (m < M) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float s = sycl::reduce_over_group(sg, a[m][r], sycl::plus<float>());
                    const int64_t n = n0 + r;
                    if (lane == 0 && n < N) {
                        float* o = out + m * ldo + n;
                        *o = acc ? *o + s : s;
                    }
                }
            }
        }
    });
}
}  // namespace

extern "C" {

int ns_audio_yue2_silo_supported(ns_gpu* g, int64_t K) {
    if (K <= 0 || K % 32 != 0) return 0;
    const auto s = g->dev.get_info<sycl::info::device::sub_group_sizes>();
    for (auto v : s)
        if (v == SG) return 1;
    return 0;
}

size_t ns_audio_yue2_silo_scales_at(int64_t N, int64_t K) { return align256((size_t) (N * K)); }
size_t ns_audio_yue2_silo_bytes(int64_t N, int64_t K) { return ns_audio_yue2_silo_scales_at(N, K) + (size_t) (N * (K / 32)) * 2; }

int ns_audio_yue2_silo_pack(ns_gpu* g, const void* q8_0, int64_t N, int64_t K, void* dst) {
    NS_TRY
    if (K % 32 != 0) return ns_fail("ns_audio_yue2_silo_pack: K a multiple of 32");
    const uint8_t* src = (const uint8_t*) q8_0;
    int8_t* oq = (int8_t*) dst;
    half* os = (half*) ((uint8_t*) dst + ns_audio_yue2_silo_scales_at(N, K));
    g->q.parallel_for(sycl::range<1>((size_t) (N * (K / 32))), [=](sycl::id<1> id) {
        const int64_t b = id[0];
        const uint8_t* blk = src + b * 34;
        os[b] = *reinterpret_cast<const half*>(blk);
        for (int i = 0; i < 32; ++i) oq[b * 32 + i] = (int8_t) blk[2 + i];
    });
    return 0;
    NS_CATCH
}

int ns_audio_yue2_silo_gemv(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* wq, const void* ws, int64_t N, float* out, int64_t ldo, int acc) {
    NS_TRY
    if (M <= 0 || N <= 0) return 0;
    if (M > MM || K % 32 != 0) return ns_fail("ns_audio_yue2_silo_gemv: at most 8 rows, K a multiple of 32");
    gemv(g->q, x, (int) M, K, (const int8_t*) wq, (const half*) ws, N, out, ldo ? ldo : N, acc != 0);
    return 0;
    NS_CATCH
}

int ns_audio_yue2_silo_half(ns_gpu* g, const void* w, const void* wsc, int64_t N, int64_t K, void* out) {
    NS_TRY
    if (N * K <= 0) return 0;
    const int8_t* wq = (const int8_t*) w;
    const half* ws = (const half*) wsc;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) (N * K / 16)), [=](sycl::id<1> id) {
        const int64_t i = id[0] * 16;
        const float d = (float) ws[i / 32];
        const sycl::vec<int8_t, 16> v = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(wq + i);
        sycl::vec<half, 16> h;
#pragma unroll
        for (int k = 0; k < 16; ++k) h[k] = (half) ((float) v[k] * d);
        *reinterpret_cast<sycl::vec<half, 16>*>(o + i) = h;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
