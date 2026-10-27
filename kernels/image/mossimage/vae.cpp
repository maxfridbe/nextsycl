// vae.cpp: the MOSS-Image engine's VAE decoder kernels (mo.h): the group norm, the attention's softmax, the nearest
// upsample and the picture's bytes. Channels-last half pictures.
#include "ns.h"
#include "ns_internal.hpp"
#include "mo.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int WG = 256, NB = 64;   // work-group size; blocks of pixels a group's statistics are split in
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
inline sycl::nd_range<1> rows(int64_t m) { return sycl::nd_range<1>(sycl::range<1>(sz(m) * WG), sycl::range<1>(WG)); }
}  // namespace

extern "C" {

int64_t ns_image_mo_group_norm_scratch(int64_t groups) { return groups * (NB + 2); }

int ns_image_mo_group_norm(ns_gpu* g, const void* x, int64_t px, int64_t C, int64_t groups, float eps, const float* gamma, const float* beta, int silu, void* out,
                           float* scratch) {
    NS_TRY
    if (px <= 0) return 0;
    if (C % groups != 0) return ns_fail("ns_image_mo_group_norm: C a multiple of the groups");
    const half* xs = (const half*) x;
    half* o = (half*) out;
    const int64_t cg = C / groups;
    float* part = scratch;                      // [groups][NB]
    float* stat = scratch + groups * NB;        // [groups][2]: mean, then rstd
    const float n = (float) (px * cg);
    // pass 1: the partial sums, then the means; pass 2: the partial sums of squared deviations, then the rstd
    for (int pass = 0; pass < 2; ++pass) {
        g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>(sz(groups * NB) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0) / NB, b = it.get_group(0) % NB, l = it.get_local_id(0);
            const int64_t p0 = px * b / NB, p1 = px * (b + 1) / NB;
            const float mean = pass == 0 ? 0.0f : stat[gi * 2];
            float s = 0.0f;
            for (int64_t i = l; i < (p1 - p0) * cg; i += WG) {
                const float v = (float) xs[(p0 + i / cg) * C + gi * cg + i % cg] - mean;
                s += pass == 0 ? v : v * v;
            }
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            if (l == 0) part[gi * NB + b] = s;
        });
        g->q.parallel_for(sycl::range<1>(sz(groups)), [=](sycl::id<1> id) {
            const int64_t gi = id[0];
            float s = 0.0f;
            for (int b = 0; b < NB; ++b) s += part[gi * NB + b];
            if (pass == 0) stat[gi * 2] = s / n;
            else stat[gi * 2 + 1] = 1.0f / sycl::sqrt(s / n + eps);
        });
    }
    const int64_t total = px * C;
    g->q.parallel_for(sycl::range<1>(sz(total)), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % C, gi = c / cg;
        float v = ((float) xs[i] - stat[gi * 2]) * stat[gi * 2 + 1] * gamma[c] + beta[c];
        if (silu) v = v / (1.0f + sycl::exp(-v));
        o[i] = (half) v;
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out) {
    NS_TRY
    half* o = (half*) out;
    if (M > 0) g->q.parallel_for(rows(M), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_local_id(0);
        const float* row = x + r * N;
        float m = -INFINITY;
        for (int64_t i = l; i < N; i += WG) m = sycl::fmax(m, row[i] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t i = l; i < N; i += WG) s += sycl::exp(row[i] * scale - m);
        const float inv = 1.0f / sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        for (int64_t i = l; i < N; i += WG) o[r * N + i] = (half) (sycl::exp(row[i] * scale - m) * inv);
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out) {
    NS_TRY
    const half* s = (const half*) x;
    half* o = (half*) out;
    const int64_t n = 4 * H * W * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t c = i % C, xx = (i / C) % (2 * W), y = i / (C * 2 * W);
        o[i] = s[((y / 2) * W + xx / 2) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_to_rgb8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgb) {
    NS_TRY
    const half* s = (const half*) x;
    const int64_t n = H * W * 3;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) {
        const float v = sycl::fmin(sycl::fmax((float) s[i], -1.0f), 1.0f) * 0.5f + 0.5f;
        rgb[i] = (uint8_t) sycl::fmin(sycl::fmax(sycl::round(v * 255.0f), 0.0f), 255.0f);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
