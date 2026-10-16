// h3s.cpp: Hunyuan3D's silo (silo.h) - its norms, GEGLU and residual adds, started as copies of the engine's own
// kernels (kernels/model3d/hunyuan3d/h3d.cpp) and changed for its shapes:
//
// - group norm: the engine's gives one work-group to a frame and group, which walks its channels in strided 2-byte
//   loads with 64-bit index math - 32 work-groups for a VAE picture at 512 px. Here each work-group sums a slab of rows
//   of every group (whole rows read, two channels a lane), the slabs' sums are added in order, and the mean and then
//   the variance about it are taken as before (two passes); the normalization runs two channels a lane.
// - layer norm: a sub-group a row where the engine's gives a row 256 lanes (at the UNet's 320 channels most of them
//   wait at its two work-group reductions).
// - GEGLU and the float32 += half adds: four values a lane, 32-bit indices (the engine's: one value, 64-bit `/` and
//   `%`, which the GPU emulates).
//
// The float32 sums are taken in another order than the engine's kernels take them; nothing else differs.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>

namespace {
using half = sycl::half;
constexpr int SG = 16;

constexpr int GN_WG = 256;
constexpr int64_t GN_MAX_PAIRS = 2048;   // channel pairs a row (the local sums of one work-group)

// row slabs a frame: enough work-groups to fill the card (~512), at least 32 rows each
int64_t gn_parts(int64_t F, int64_t N) {
    return std::max<int64_t>(1, std::min<int64_t>((512 + F - 1) / F, N / 32));
}

// The sums over each (frame, group, slab) of x (pass 0) or of (x - the group's mean)^2 (pass 1) into part
// [F * G][P]: a work-group a (frame, slab); lane t sums channel pair t % pairs of rows t / pairs, t / pairs + RI, ...
// (or pairs t, t + 256, ... of every row when a row has more than 256 pairs), then lane g < G adds its group's
// lane sums in a fixed order
void gn_sums(sycl::queue& q, const half* a, int64_t F, int64_t N, int64_t C, int64_t G, int64_t P, const float* stats, int pass, float* part) {
    const int pairs = (int) (C / 2), cg = (int) (C / G);
    const bool wide = pairs > GN_WG;
    const int ri = wide ? 1 : GN_WG / pairs;
    const int lanes = wide ? GN_WG : ri * pairs;
    const int kk = wide ? (pairs + GN_WG - 1) / GN_WG : 1;
    const int nloc = std::max(GN_WG, pairs);
    const uint32_t uN = (uint32_t) N, uC = (uint32_t) C;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> loc(sycl::range<1>((size_t) nloc), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (F * P) * GN_WG), sycl::range<1>(GN_WG)), [=](sycl::nd_item<1> it) {
            const uint32_t wg = (uint32_t) it.get_group(0);
            const uint32_t f = wg / (uint32_t) P, p = wg % (uint32_t) P;
            const uint32_t r0 = (uint32_t) ((uint64_t) uN * p / (uint64_t) P), r1 = (uint32_t) ((uint64_t) uN * (p + 1) / (uint64_t) P);
            const int t = (int) it.get_local_id(0);
            const half* fr = a + (size_t) f * uN * uC;
            for (int k = 0; k < kk; ++k) {
                const int cp = wide ? t + k * GN_WG : t % pairs;
                if (cp >= pairs || t >= lanes) continue;
                const int rr = wide ? 0 : t / pairs;
                const float mu = pass ? stats[(f * (uint32_t) G + (uint32_t) (2 * cp / cg)) * 2] : 0.0f;
                float s = 0.0f;
                for (uint32_t r = r0 + (uint32_t) rr; r < r1; r += (uint32_t) ri) {
                    const half* px = fr + r * uC + 2 * (uint32_t) cp;
                    const float x0 = (float) px[0], x1 = (float) px[1];
                    if (pass) {
                        s += (x0 - mu) * (x0 - mu) + (x1 - mu) * (x1 - mu);
                    } else {
                        s += x0 + x1;
                    }
                }
                loc[wide ? cp : t] = s;
            }
            sycl::group_barrier(it.get_group());
            if (t < G) {
                float s = 0.0f;
                for (int rr = 0; rr < ri; ++rr)
                    for (int cp = t * cg / 2; cp < (t + 1) * cg / 2; ++cp) s += loc[rr * pairs + cp];
                part[((size_t) f * G + t) * P + p] = s;
            }
        });
    });
}
}  // namespace

extern "C" {

int ns_model3d_hy_silo_group_norm_supported(int64_t F, int64_t N, int64_t C, int64_t G) {
    if (F <= 0 || N <= 0 || G <= 0 || G > GN_WG || C % G != 0 || (C / G) % 2 != 0) return 0;
    if (C / 2 > GN_MAX_PAIRS || F * N * C >= ((int64_t) 1 << 31)) return 0;
    return 1;
}

int64_t ns_model3d_hy_silo_group_norm_scratch(int64_t F, int64_t N, int64_t C, int64_t G) {
    (void) C;
    return F * G * gn_parts(F, N) + F * G * 2;
}

int ns_model3d_hy_silo_group_norm(ns_gpu* g, const void* x, int64_t F, int64_t N, int64_t C, int64_t G, const float* w, const float* b, float eps,
                                  int silu, float* scratch, void* out) {
    NS_TRY
    if (!ns_model3d_hy_silo_group_norm_supported(F, N, C, G)) return ns_fail("ns_model3d_hy_silo_group_norm: shapes it does not take");
    const int64_t P = gn_parts(F, N);
    const half* a = (const half*) x;
    half* o = (half*) out;
    float* part = scratch;
    float* stats = scratch + F * G * P;
    const int64_t cg = C / G;
    const float count = (float) (N * cg);
    for (int pass = 0; pass < 2; ++pass) {
        gn_sums(g->q, a, F, N, C, G, P, stats, pass, part);
        g->q.parallel_for(sycl::range<1>((size_t) (F * G)), [=](sycl::id<1> id) {
            const size_t fg = id[0];
            float s = 0.0f;
            for (int64_t p = 0; p < P; ++p) s += part[fg * P + p];
            if (pass == 0) stats[fg * 2] = s / count;
            else stats[fg * 2 + 1] = sycl::rsqrt(s / count + eps);
        });
    }
    const uint32_t uN = (uint32_t) N, uC = (uint32_t) C, ucg = (uint32_t) cg, uG = (uint32_t) G;
    g->q.parallel_for(sycl::range<2>((size_t) (F * N), (size_t) (C / 2)), [=](sycl::id<2> id) {
        const uint32_t row = (uint32_t) id[0], c = 2 * (uint32_t) id[1];
        const uint32_t fg = (row / uN) * uG + c / ucg;
        const float mean = stats[fg * 2], inv = stats[fg * 2 + 1];
        const half* px = a + row * uC + c;
        half* po = o + row * uC + c;
        for (int j = 0; j < 2; ++j) {
            float y = ((float) px[j] - mean) * inv * w[c + j] + b[c + j];
            if (silu) y = y / (1.0f + sycl::exp(-y));
            po[j] = (half) y;
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hy_silo_ln_supported(int64_t C) {
    return C > 0 && C <= SG * 64;
}

}  // extern "C"

namespace {
// a sub-group a row, `RPW` rows a work-group; lane l holds values l, l + 16, ... (PER at most)
template <int PER>
void ln_rows(sycl::queue& q, const void* x, int x_half, int64_t M, int64_t C, float eps, const float* w, const float* b, const float* scale,
             const float* shift, void* out, int out_half) {
    constexpr int RPW = 8;
    const size_t groups = (size_t) ((M + RPW - 1) / RPW);
    const int c = (int) C;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * RPW * SG), sycl::range<1>(RPW * SG)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t r = (int64_t) it.get_group(0) * RPW + (int64_t) sg.get_group_linear_id();
        if (r >= M) return;
        const int l = (int) sg.get_local_linear_id();
        float v[PER];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = j * SG + l;
            float a = 0.0f;
            if (i < c) a = x_half ? (float) ((const half*) x)[r * c + i] : ((const float*) x)[r * c + i];
            v[j] = a;
            s += a;
        }
        const float mean = sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) c;
        float s2 = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = j * SG + l;
            const float d = i < c ? v[j] - mean : 0.0f;
            s2 += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s2, sycl::plus<float>()) / (float) c + eps);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = j * SG + l;
            if (i < c) {
                float y = (v[j] - mean) * inv;
                if (w) y *= w[i];
                if (b) y += b[i];
                if (scale) y *= 1.0f + scale[i];
                if (shift) y += shift[i];
                if (out_half) ((half*) out)[r * c + i] = (half) y;
                else ((float*) out)[r * c + i] = y;
            }
        }
    });
}
}  // namespace

extern "C" {

int ns_model3d_hy_silo_ln(ns_gpu* g, const void* x, int x_half, int64_t M, int64_t C, float eps, const float* w, const float* b, const float* scale,
                          const float* shift, void* out, int out_half) {
    NS_TRY
    if (!ns_model3d_hy_silo_ln_supported(C)) return ns_fail("ns_model3d_hy_silo_ln: rows of 1 to 1,024 values");
    if (M <= 0) return 0;
    const int64_t per = (C + SG - 1) / SG;
    if (per <= 8) ln_rows<8>(g->q, x, x_half, M, C, eps, w, b, scale, shift, out, out_half);
    else if (per <= 16) ln_rows<16>(g->q, x, x_half, M, C, eps, w, b, scale, shift, out, out_half);
    else if (per <= 32) ln_rows<32>(g->q, x, x_half, M, C, eps, w, b, scale, shift, out, out_half);
    else ln_rows<64>(g->q, x, x_half, M, C, eps, w, b, scale, shift, out, out_half);
    return 0;
    NS_CATCH
}

int ns_model3d_hy_silo_geglu_supported(int64_t M, int64_t F) {
    return M > 0 && F > 0 && F % 4 == 0 && M * 2 * F < ((int64_t) 1 << 31);
}

int ns_model3d_hy_silo_geglu(ns_gpu* g, const void* x, int64_t M, int64_t F, void* out) {
    NS_TRY
    if (!ns_model3d_hy_silo_geglu_supported(M, F)) return ns_fail("ns_model3d_hy_silo_geglu: shapes it does not take");
    const half* a = (const half*) x;
    half* o = (half*) out;
    const uint32_t uF = (uint32_t) F;
    using h4 = sycl::vec<half, 4>;
    const bool aligned = (uintptr_t) x % 8 == 0 && (uintptr_t) out % 8 == 0;
    if (aligned) {
        g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) (F / 4)), [=](sycl::id<2> id) {
            const uint32_t r = (uint32_t) id[0], c = 4 * (uint32_t) id[1];
            const h4 hv = *reinterpret_cast<const h4*>(a + r * 2 * uF + c);
            const h4 gv = *reinterpret_cast<const h4*>(a + r * 2 * uF + uF + c);
            h4 y;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gt = (float) gv[j];
                y[j] = (half) ((float) hv[j] * 0.5f * gt * (1.0f + sycl::erf(gt * 0.70710678118654752f)));
            }
            *reinterpret_cast<h4*>(o + r * uF + c) = y;
        });
    } else {
        g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) F), [=](sycl::id<2> id) {
            const uint32_t r = (uint32_t) id[0], c = (uint32_t) id[1];
            const float h = (float) a[r * 2 * uF + c], gt = (float) a[r * 2 * uF + uF + c];
            o[r * uF + c] = (half) (h * 0.5f * gt * (1.0f + sycl::erf(gt * 0.70710678118654752f)));
        });
    }
    return 0;
    NS_CATCH
}

int ns_model3d_hy_silo_add_supported(int64_t M, int64_t C) {
    return M > 0 && C > 0 && C % 4 == 0 && M * C < ((int64_t) 1 << 31);
}

int ns_model3d_hy_silo_add(ns_gpu* g, float* x, const void* y, int y_half, int64_t M, int64_t C, const float* gate) {
    NS_TRY
    if (!ns_model3d_hy_silo_add_supported(M, C)) return ns_fail("ns_model3d_hy_silo_add: shapes it does not take");
    const uint32_t uC = (uint32_t) C;
    const bool aligned = (uintptr_t) x % 16 == 0 && (uintptr_t) y % (y_half ? 8 : 16) == 0;
    if (!aligned) {
        g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
            const uint32_t i = (uint32_t) id[0] * uC + (uint32_t) id[1];
            const float v = y_half ? (float) ((const half*) y)[i] : ((const float*) y)[i];
            x[i] += gate ? v * gate[id[1]] : v;
        });
        return 0;
    }
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) (C / 4)), [=](sycl::id<2> id) {
        const uint32_t c = 4 * (uint32_t) id[1];
        const uint32_t i = (uint32_t) id[0] * uC + c;
        sycl::float4 v;
        if (y_half) v = reinterpret_cast<const sycl::vec<half, 4>*>((const half*) y + i)->convert<float>();
        else v = *reinterpret_cast<const sycl::float4*>((const float*) y + i);
        if (gate) v *= sycl::float4(gate[c], gate[c + 1], gate[c + 2], gate[c + 3]);
        sycl::float4* px = reinterpret_cast<sycl::float4*>(x + i);
        *px += v;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
