// norm.cpp: the MOSS-Image silo's fused passes (silo.h). The engine's own path runs the gated residual add, the row
// statistics and the modulated norm as three passes over the activations (x float32 [4128, 3072] is 50 MB); here a
// work-group takes a row, keeps it in registers, and reads x and the product once, writes x and the half rows once.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

namespace {
using half = sycl::half;
constexpr int WG = 256;
using f4 = sycl::vec<float, 4>;
using h8 = sycl::vec<half, 8>;
using u16x4 = sycl::vec<uint16_t, 4>;
}  // namespace

extern "C" {

int ns_image_moss_silo_supported(ns_gpu* g, int64_t C, int64_t D) {
    if (C % 1024 != 0 || C > 4096 || D != 128) return 0;
    for (size_t s : g->dev.get_info<sycl::info::device::sub_group_sizes>())
        if (s == 16) return 1;
    return 0;
}

int ns_image_moss_silo_add_norm(ns_gpu* g, float* x, const float* resid, const float* gate, int64_t M, int64_t C, const float* weight, float eps,
                                const float* scale, const float* shift, void* out) {
    NS_TRY
    if (M <= 0) return 0;
    // a work-group a row, a lane 4 consecutive floats a sweep of 1024
    if (C % (4 * WG) != 0 || C / (4 * WG) > 4) return ns_fail("ns_image_moss_silo_add_norm: C a multiple of 1024, at most 4096");
    half* o = (half*) out;
    const int64_t sweeps = C / (4 * WG);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        f4 v[4];
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k < sweeps) {
                const int64_t i = (int64_t) k * 4 * WG + (int64_t) l * 4;
                f4 a = *(const f4*) (x + r * C + i);
                if (resid) {
                    const f4 b = *(const f4*) (resid + r * C + i), gt = *(const f4*) (gate + i);
                    a = a + b * gt;
                    *(f4*) (x + r * C + i) = a;
                }
                v[k] = a;
                s += a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
            }
        }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = sycl::rsqrt(s / (float) C + eps);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k < sweeps) {
                const int64_t i = (int64_t) k * 4 * WG + (int64_t) l * 4;
                const f4 w = *(const f4*) (weight + i);
                f4 y = v[k] * inv * w;
                if (scale) {
                    const f4 sc = *(const f4*) (scale + i), sh = *(const f4*) (shift + i);
                    y = y * (1.0f + sc) + sh;
                }
                sycl::vec<half, 4> hv;
                for (int e = 0; e < 4; ++e) hv[e] = (half) y[e];
                *(sycl::vec<half, 4>*) (o + r * C + i) = hv;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_moss_silo_qk_rope(ns_gpu* g, void* qkv, int64_t M, int64_t H, int64_t D, int64_t stride, int64_t kcol, const float* wq, const float* wk,
                               float eps, const float* cs) {
    NS_TRY
    if (M <= 0) return 0;
    if (D != 128 || stride % 8 != 0 || kcol % 8 != 0) return ns_fail("ns_image_moss_silo_qk_rope: head 128, row stride and k column multiples of 8");
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
            const float x = a[e] * inv * wa[e], y = b[e] * inv * wb[e];
            oa[e] = (half) (x * co[e] - y * si[e]);
            ob[e] = (half) (y * co[e] + x * si[e]);
        }
        *(sycl::vec<half, 4>*) (hp + 4 * l) = oa;
        *(sycl::vec<half, 4>*) (hp + 64 + 4 * l) = ob;
    });
    return 0;
    NS_CATCH
}

int ns_image_moss_silo_situ(ns_gpu* g, const void* hid, const void* gate, int64_t ldh, int64_t M, int64_t C, void* out, int64_t ldo) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    if (C % 8 != 0 || ldh % 8 != 0 || ldo % 8 != 0) return ns_fail("ns_image_moss_silo_situ: C, ldh and ldo multiples of 8");
    const half* hp = (const half*) hid;
    const half* gp = (const half*) gate;
    half* o = (half*) out;
    const int64_t c8 = C / 8;
    g->q.parallel_for(sycl::range<1>((size_t) (M * c8)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / c8, c = (i % c8) * 8;
        const h8 hv = *(const h8*) (hp + r * ldh + c), gv = *(const h8*) (gp + r * ldh + c);
        h8 ov;
        for (int e = 0; e < 8; ++e) {
            const float h = (float) hv[e], gt = (float) gv[e];
            const float gain = 4.0f * sycl::tanh(gt / 4.0f) * (1.0f / (1.0f + sycl::exp(-gt)));
            ov[e] = (half) (gain * (25.0f * sycl::tanh(h / 25.0f)));
        }
        *(h8*) (o + r * ldo + c) = ov;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
