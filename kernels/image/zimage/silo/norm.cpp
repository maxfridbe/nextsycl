// norm.cpp: the Z-Image silo's fused passes (silo.h). The engine's own path runs each residual step as three passes over
// the activations (x float32 [4224, 3840] is 65 MB): the product's norm to half, the gated add, the next modulated norm
// to half. Here a work-group takes a row, keeps it in registers, reads the product and x once, writes x and the half
// row once.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

namespace {
using half = sycl::half;
constexpr int WG = 256;
constexpr int K = 4;  // a lane's float4 chunks of a row: up to 4 x 4 x 256 = 4096 values
using f4 = sycl::vec<float, 4>;
using h8 = sycl::vec<half, 8>;
}  // namespace

extern "C" {

int ns_image_zimage_silo_supported(ns_gpu* g, int64_t C, int64_t D) {
    if (C % 4 != 0 || C > 4 * K * WG || D != 128) return 0;
    for (size_t s : g->dev.get_info<sycl::info::device::sub_group_sizes>())
        if (s == 16) return 1;
    return 0;
}

int ns_image_zimage_silo_norm_gate_norm(ns_gpu* g, const float* o, float* x, int64_t M, int64_t C, const float* wa, float eps_a, const float* ga,
                                        const float* wb, float eps_b, const float* sb, void* out) {
    NS_TRY
    if (M <= 0) return 0;
    if (C % 4 != 0 || C > 4 * K * WG) return ns_fail("ns_image_zimage_silo_norm_gate_norm: C a multiple of 4, at most 4096");
    half* hout = (half*) out;
    const int64_t chunks = C / 4;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        float* xr = x + r * C;
        f4 v[K];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int64_t c = (int64_t) k * WG + l;
            v[k] = c < chunks ? *(const f4*) (xr + 4 * c) : f4(0.0f);
        }
        if (o) {
            const float* orow = o + r * C;
            f4 p[K];
            float s = 0.0f;
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const int64_t c = (int64_t) k * WG + l;
                p[k] = c < chunks ? *(const f4*) (orow + 4 * c) : f4(0.0f);
                s += p[k][0] * p[k][0] + p[k][1] * p[k][1] + p[k][2] * p[k][2] + p[k][3] * p[k][3];
            }
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            const float inv = 1.0f / sycl::sqrt(s / (float) C + eps_a);
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const int64_t c = (int64_t) k * WG + l;
                if (c < chunks) {
                    const f4 w = *(const f4*) (wa + 4 * c);
                    // the engine's path rounds the normed product to half before the add: kept, so the two agree
                    f4 h = p[k] * inv * w;
                    for (int e = 0; e < 4; ++e) h[e] = (float) (half) h[e];
                    v[k] = v[k] + (ga ? h * *(const f4*) (ga + 4 * c) : h);
                    *(f4*) (xr + 4 * c) = v[k];
                }
            }
        }
        if (hout) {
            float s = 0.0f;
#pragma unroll
            for (int k = 0; k < K; ++k) s += v[k][0] * v[k][0] + v[k][1] * v[k][1] + v[k][2] * v[k][2] + v[k][3] * v[k][3];
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            const float inv = 1.0f / sycl::sqrt(s / (float) C + eps_b);
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const int64_t c = (int64_t) k * WG + l;
                if (c < chunks) {
                    f4 y = v[k] * inv * *(const f4*) (wb + 4 * c);
                    if (sb) y = y * (1.0f + *(const f4*) (sb + 4 * c));
                    sycl::vec<half, 4> hv;
                    for (int e = 0; e < 4; ++e) hv[e] = (half) y[e];
                    *(sycl::vec<half, 4>*) (hout + r * C + 4 * c) = hv;
                }
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_zimage_silo_qk_rope(ns_gpu* g, void* qkv, int64_t M, int64_t H, int64_t D, int64_t stride, int64_t kcol, const float* wq, const float* wk,
                                 float eps, const float* cs) {
    NS_TRY
    if (M <= 0) return 0;
    if (D != 128 || stride % 8 != 0 || kcol % 8 != 0) return ns_fail("ns_image_zimage_silo_qk_rope: head 128, row stride and k column multiples of 8");
    half* p = (half*) qkv;
    // a sub-group of 16 a head: lane l owns pairs 4 l .. 4 l + 3 (features 4 l.. and 64 + 4 l..), 8-byte loads
    constexpr int SGS = 16, WGS = 4 * SGS;
    const int64_t heads = 2 * H;                            // q's then k's
    const int64_t total = M * heads;
    const int64_t groups = (total + 3) / 4;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (groups * WGS)), sycl::range<1>(WGS)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int64_t u = it.get_group(0) * 4 + sg.get_group_id()[0];
        const int l = (int) sg.get_local_id()[0];
        if (u >= total) return;
        const int64_t t = u / heads, h = u % heads;
        const bool isk = h >= H;
        const int64_t col = (isk ? kcol : 0) + (isk ? h - H : h) * D;
        half* hp = p + t * stride + col;
        const float* w = isk ? wk : wq;
        const sycl::vec<half, 4> va = *(const sycl::vec<half, 4>*) (hp + 4 * l), vb = *(const sycl::vec<half, 4>*) (hp + 64 + 4 * l);
        float a[4], b[4], s = 0.0f;
        for (int e = 0; e < 4; ++e) {
            a[e] = (float) va[e];
            b[e] = (float) vb[e];
            s += a[e] * a[e] + b[e] * b[e];
        }
        s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
        const float inv = sycl::rsqrt(s / (float) D + eps);
        const f4 wa = *(const f4*) (w + 4 * l), wb = *(const f4*) (w + 64 + 4 * l);
        const f4 c0 = *(const f4*) (cs + t * D + 8 * l), c1 = *(const f4*) (cs + t * D + 8 * l + 4);
        const float co[4] = {c0[0], c0[2], c1[0], c1[2]}, si[4] = {c0[1], c0[3], c1[1], c1[3]};
        sycl::vec<half, 4> oa, ob;
        for (int e = 0; e < 4; ++e) {
            const float xx = a[e] * inv * wa[e], y = b[e] * inv * wb[e];
            oa[e] = (half) (xx * co[e] - y * si[e]);
            ob[e] = (half) (y * co[e] + xx * si[e]);
        }
        *(sycl::vec<half, 4>*) (hp + 4 * l) = oa;
        *(sycl::vec<half, 4>*) (hp + 64 + 4 * l) = ob;
    });
    return 0;
    NS_CATCH
}

int ns_image_zimage_silo_swiglu(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    if (C % 8 != 0) return ns_fail("ns_image_zimage_silo_swiglu: C a multiple of 8");
    const half* xs = (const half*) x;
    half* o = (half*) out;
    const int64_t c8 = C / 8;
    g->q.parallel_for(sycl::range<1>((size_t) (M * c8)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / c8, c = (i % c8) * 8;
        const h8 av = *(const h8*) (xs + r * 2 * C + c), bv = *(const h8*) (xs + r * 2 * C + C + c);
        h8 ov;
        for (int e = 0; e < 8; ++e) {
            const float a = (float) av[e], b = (float) bv[e];
            ov[e] = (half) (a / (1.0f + sycl::exp(-a)) * b * s);
        }
        *(h8*) (o + r * C + c) = ov;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
