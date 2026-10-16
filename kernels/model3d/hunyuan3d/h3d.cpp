// h3d.cpp: the Hunyuan3D engine's own small kernels - what the shared diffusion kernels (nsd) do not have: the layer
// norm with its affine and modulation in one pass, the per-head q / k norms in strided rows, the residual adds, the
// sampler's steps, the shape decoder's query embedding and the MoE's routing. On the GPU's queue.
#include "ns.h"
#include "ns_internal.hpp"
#include "h3d.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>

namespace {
using half = sycl::half;
}

extern "C" {

int ns_model3d_hunyuan3d_ln(ns_gpu* g, const void* x, int x_half, int64_t M, int64_t C, float eps, const float* w, const float* b,
                            const float* scale, const float* shift, void* out, int out_half) {
    NS_TRY
    constexpr int WG = 256, PER = 32;            // a row of up to 8,192 values held in registers
    if (C > (int64_t) WG * PER || C <= 0) return ns_fail("ns_model3d_hunyuan3d_ln: rows of 1 to 8192 values");
    if (M <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        float v[PER];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            float a = 0.0f;
            if (i < C) a = x_half ? (float) ((const half*) x)[r * C + i] : ((const float*) x)[r * C + i];
            v[j] = a;
            s += a;
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float s2 = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            const float d = i < C ? v[j] - mean : 0.0f;
            s2 += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) {
                float y = (v[j] - mean) * inv;
                if (w) y *= w[i];
                if (b) y += b[i];
                if (scale) y *= 1.0f + scale[i];
                if (shift) y += shift[i];
                if (out_half) ((half*) out)[r * C + i] = (half) y;
                else ((float*) out)[r * C + i] = y;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_head_norm(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* w, const float* b,
                                   float eps, int rms) {
    NS_TRY
    constexpr int WG = 32, PER = 8;
    if (D % WG != 0 || D > WG * PER) return ns_fail("ns_model3d_hunyuan3d_head_norm: D a multiple of 32, at most 256");
    if (stride < H * D) return ns_fail("ns_model3d_hunyuan3d_head_norm: rows narrower than their heads");
    if (M <= 0 || H <= 0) return 0;
    half* p = (half*) x;
    const int per = (int) (D / WG);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * H) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t gi = (int64_t) it.get_group(0);
        const int64_t m = gi / H, h = gi % H;
        const int l = (int) it.get_local_id(0);
        half* row = p + m * stride + h * D;
        float v[PER];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            v[j] = j < per ? (float) row[j * WG + l] : 0.0f;
            s += rms ? v[j] * v[j] : v[j];
        }
        const float tot = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        float mean = 0.0f, inv;
        if (rms) {
            inv = sycl::rsqrt(tot / (float) D + eps);
        } else {
            mean = tot / (float) D;
            float s2 = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float d = j < per ? v[j] - mean : 0.0f;
                s2 += d * d;
            }
            inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) D + eps);
        }
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            if (j < per) {
                const int i = j * WG + l;
                float y = (v[j] - mean) * inv * w[i];
                if (b) y += b[i];
                row[i] = (half) y;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_silu(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float a = x[i];
        o[i] = (half) (a / (1.0f + sycl::exp(-a)));
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    const half* a = (const half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = (float) a[i]; });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_add(ns_gpu* g, float* x, const void* y, int y_half, int64_t M, int64_t C, const float* gate) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const float v = y_half ? (float) ((const half*) y)[i] : ((const float*) y)[i];
        x[i] += gate ? v * gate[i % C] : v;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += a * y[i]; });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_cfg(ns_gpu* g, float* out, const float* c, const float* u, int64_t n, float s) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = u[i] + s * (c[i] - u[i]); });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_fourier(ns_gpu* g, const float* pts, const int32_t* idx, int64_t start, int64_t N, int64_t R1, float lo,
                                 float step, int nf, int pi, void* out) {
    NS_TRY
    if (nf < 0 || nf > 16) return ns_fail("ns_model3d_hunyuan3d_fourier: 0 to 16 frequencies");
    if (!pts && R1 <= 0) return ns_fail("ns_model3d_hunyuan3d_fourier: no points and no lattice");
    if (N <= 0) return 0;
    half* o = (half*) out;
    const int64_t width = 3 * (2 * (int64_t) nf + 1);
    const float fpi = pi ? 3.14159265358979323846f : 1.0f;
    g->q.parallel_for(sycl::range<1>((size_t) N), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        float p[3];
        if (pts) {
            p[0] = pts[i * 3];
            p[1] = pts[i * 3 + 1];
            p[2] = pts[i * 3 + 2];
        } else {
            const int64_t q = idx ? (int64_t) idx[i] : start + i;
            const int64_t ix = q / (R1 * R1), iy = (q / R1) % R1, iz = q % R1;
            p[0] = lo + (float) ix * step;
            p[1] = lo + (float) iy * step;
            p[2] = lo + (float) iz * step;
        }
        half* row = o + i * width;
        for (int d = 0; d < 3; ++d) row[d] = (half) p[d];
        for (int d = 0; d < 3; ++d) {
            float f = fpi;
            for (int j = 0; j < nf; ++j) {
                const float a = p[d] * f;
                row[3 + d * nf + j] = (half) sycl::sin(a);
                row[3 + 3 * nf + d * nf + j] = (half) sycl::cos(a);
                f *= 2.0f;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_route(ns_gpu* g, const float* logits, int64_t M, int64_t E, int64_t k, int32_t* idx, float* w) {
    NS_TRY
    if (E <= 0 || E > 64 || k <= 0 || k > 8 || k > E) return ns_fail("ns_model3d_hunyuan3d_route: 1 to 64 experts, 1 to 8 chosen");
    if (M <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) M), [=](sycl::id<1> id) {
        const int64_t r = (int64_t) id[0];
        const float* l = logits + r * E;
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t e = 0; e < E; ++e) mx = sycl::fmax(mx, l[e]);
        float sum = 0.0f;
        for (int64_t e = 0; e < E; ++e) sum += sycl::exp(l[e] - mx);
        uint64_t taken = 0;
        for (int64_t j = 0; j < k; ++j) {
            int64_t best = -1;
            float bv = -std::numeric_limits<float>::infinity();
            for (int64_t e = 0; e < E; ++e) {
                if ((taken >> e) & 1) continue;
                if (best < 0 || l[e] > bv) {
                    best = e;
                    bv = l[e];
                }
            }
            taken |= (uint64_t) 1 << best;
            idx[r * k + j] = (int32_t) best;
            w[r * k + j] = sycl::exp(bv - mx) / sum;
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_gather(ns_gpu* g, const void* x, const int32_t* idx, int64_t n, int64_t C, void* out) {
    NS_TRY
    if (n <= 0 || C <= 0) return 0;
    const half* a = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) C), [=](sycl::id<2> id) {
        const int64_t i = (int64_t) id[0], c = (int64_t) id[1];
        o[i * C + c] = a[(int64_t) idx[i] * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hunyuan3d_scatter_add(ns_gpu* g, const void* y, const int32_t* idx, const float* w, int64_t n, int64_t C, float* out) {
    NS_TRY
    if (n <= 0 || C <= 0) return 0;
    const half* a = (const half*) y;
    g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) C), [=](sycl::id<2> id) {
        const int64_t i = (int64_t) id[0], c = (int64_t) id[1];
        out[(int64_t) idx[i] * C + c] += w[i] * (float) a[i * C + c];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
