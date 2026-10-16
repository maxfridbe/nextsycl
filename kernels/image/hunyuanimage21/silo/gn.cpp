// gn.cpp: the VAE's GroupNorm (+ SiLU) for its shapes (silo.h). The engine's own kernel (hi21.cpp) gives a
// work-group a (group, part of the pixels) and reads one group's few channels a pixel at a time - at 2048 x 2048 the
// last levels' 128 channels in 32 groups leave 60 of its 64 lanes idle, ~27 ms a call. Here every pass reads the
// picture as one flat stream of 8-half vectors: a lane keeps the same 8 channels throughout (2048 values a sweep of the
// work-group, a multiple of the row), so its sums stay per channel until the work-group folds them into groups. The
// same two passes as PyTorch (the mean, then the mean square about it), the same float32 arithmetic per value.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

namespace {
using half = sycl::half;
constexpr int WG = 256, VEC = 8, SWEEP = WG * VEC;   // values a work-group reads in one sweep
constexpr int64_t PARTS = 512;                        // work-groups a pass (each a run of whole rows)
using h8 = sycl::vec<half, VEC>;
}  // namespace

extern "C" {

int64_t ns_image_hi21_silo_gn_scratch(int64_t G) {
    return PARTS * G + 2 * G;
}

int ns_image_hi21_silo_group_norm_supported(int64_t C, int64_t G) {
    return C >= VEC && SWEEP % C == 0 && G > 0 && C % G == 0 ? 1 : 0;
}

int ns_image_hi21_silo_group_norm(ns_gpu* g, const void* x, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias, float eps,
                                  int silu, float* scratch, void* out) {
    NS_TRY
    if (N <= 0) return 0;
    if (!ns_image_hi21_silo_group_norm_supported(C, G)) return ns_fail("ns_image_hi21_silo_group_norm: C a power of two from 8 to 2048, a multiple of G");
    const half* xi = (const half*) x;
    half* o = (half*) out;
    const int64_t cg = C / G;
    const int64_t rows_sweep = SWEEP / C;                 // rows a sweep covers
    // rows a part: a whole number of sweeps, so every part starts a sweep at channel 0
    const int64_t per = ((N + PARTS - 1) / PARTS + rows_sweep - 1) / rows_sweep * rows_sweep;
    float* part = scratch;
    float* stat = scratch + PARTS * G;                    // per group: mean, 1 / std
    const float cnt = (float) N * (float) cg;
    for (int pass = 0; pass < 2; ++pass) {
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> red(sycl::range<1>(SWEEP), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (PARTS * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
                const int64_t p = it.get_group(0);
                const int l = (int) it.get_local_id(0);
                const int64_t c0 = (int64_t) l * VEC % C;       // this lane's 8 channels
                float mean[VEC], a[VEC];
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    mean[j] = pass == 0 ? 0.0f : stat[(c0 + j) / cg * 2];
                    a[j] = 0.0f;
                }
                const int64_t r0 = p * per, r1 = sycl::min(N, r0 + per);
                for (int64_t i = r0 * C + (int64_t) l * VEC; i < r1 * C; i += SWEEP) {
                    const h8 v = *(const h8*) (xi + i);
#pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        const float d = (float) v[j] - mean[j];
                        a[j] += pass == 0 ? d : d * d;
                    }
                }
#pragma unroll
                for (int j = 0; j < VEC; ++j) red[l * VEC + j] = a[j];
                sycl::group_barrier(it.get_group());
                // a lane a group: its channels' sums over every lane holding them (lane c / 8 + m C / 8)
                if (l < G) {
                    float s = 0.0f;
                    for (int64_t c = l * cg; c < (l + 1) * cg; ++c)
                        for (int64_t m = 0; m < rows_sweep; ++m) s += red[(c / VEC + m * (C / VEC)) * VEC + c % VEC];
                    part[p * G + l] = s;
                }
            });
        });
        g->q.parallel_for(sycl::range<1>((size_t) G), [=](sycl::id<1> gr) {
            float s = 0.0f;
            for (int64_t p = 0; p < PARTS; ++p) s += part[p * G + gr];
            if (pass == 0) stat[gr * 2] = s / cnt;
            else stat[gr * 2 + 1] = sycl::rsqrt(s / cnt + eps);
        });
    }
    g->q.parallel_for(sycl::range<1>((size_t) (N * C / VEC)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0] * VEC, c0 = i % C;
        const h8 v = *(const h8*) (xi + i);
        h8 r;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const int64_t ch = c0 + j, gr = ch / cg;
            float y = ((float) v[j] - stat[gr * 2]) * stat[gr * 2 + 1] * weight[ch] + bias[ch];
            if (silu) y = y / (1.0f + sycl::exp(-y));
            r[j] = (half) y;
        }
        *(h8*) (o + i) = r;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
