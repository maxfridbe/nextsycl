// te.cpp: the MOSS-Image engine's text encoder kernels (mo.h) - Qwen3.5-9B's gated DeltaNet and gated full attention
// pieces, float32. The kernels of kernels/llm/qwen35/q35.cpp and glm5next/glm.cpp (llama.cpp's qwen35 graph, checked
// there), as this kind's own: an image engine does not call the language library.
#include "ns.h"
#include "ns_internal.hpp"
#include "mo.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
constexpr int SG = 16;

// NC value columns of one head a sub-group: they share the step's key, decay and query loads, and their reductions
// overlap (glm5next's kda_scan)
template <int NC>
void kda_scan(sycl::queue& q, const float* qv, const float* kv, const float* vv, const float* eg, const float* beta, float* S, float* o, int64_t T,
              int64_t H) {
    constexpr int KPL = 128 / SG, WGS = 8;   // keys a lane, sub-groups a work-group
    const int64_t groups = H * 128 / NC;
    q.parallel_for(sycl::nd_range<1>((groups + WGS - 1) / WGS * WGS * SG, WGS * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t c0 = it.get_global_id(0) / SG * NC;   // the first column: head hd, values j0..j0 + NC
        if (c0 >= H * 128) return;
        const int64_t hd = c0 / 128;
        const int j0 = (int) (c0 % 128), lane = (int) sg.get_local_id()[0], i0 = lane * KPL;
        float* Sh = S + hd * 128 * 128;
        float col[NC][KPL];
#pragma unroll
        for (int n = 0; n < NC; ++n)
#pragma unroll
            for (int i = 0; i < KPL; ++i) col[n][i] = Sh[(i0 + i) * 128 + j0 + n];
        const float scale = 1.f / sycl::sqrt(128.f);
        for (int64_t t = 0; t < T; ++t) {
            const int64_t base = (t * H + hd) * 128;
            const sycl::vec<float, KPL> kq = *reinterpret_cast<const sycl::vec<float, KPL>*>(kv + base + i0);
            const sycl::vec<float, KPL> gq = *reinterpret_cast<const sycl::vec<float, KPL>*>(eg + base + i0);
            const sycl::vec<float, KPL> qq = *reinterpret_cast<const sycl::vec<float, KPL>*>(qv + base + i0);
            const float bt = beta[t * H + hd];
            float dot[NC], out[NC];
#pragma unroll
            for (int n = 0; n < NC; ++n) {
                dot[n] = 0.f;
#pragma unroll
                for (int i = 0; i < KPL; ++i) { col[n][i] *= gq[i]; dot[n] += col[n][i] * kq[i]; }
            }
#pragma unroll
            for (int n = 0; n < NC; ++n) dot[n] = sycl::reduce_over_group(sg, dot[n], sycl::plus<float>());
#pragma unroll
            for (int n = 0; n < NC; ++n) {
                const float delta = (vv[base + j0 + n] - dot[n]) * bt;
                out[n] = 0.f;
#pragma unroll
                for (int i = 0; i < KPL; ++i) { col[n][i] += kq[i] * delta; out[n] += col[n][i] * qq[i]; }
            }
#pragma unroll
            for (int n = 0; n < NC; ++n) out[n] = sycl::reduce_over_group(sg, out[n], sycl::plus<float>());
            if (lane < NC) {
                float mine = out[0];
#pragma unroll
                for (int n = 1; n < NC; ++n) mine = lane == n ? out[n] : mine;
                o[base + j0 + lane] = mine * scale;
            }
        }
#pragma unroll
        for (int n = 0; n < NC; ++n)
#pragma unroll
            for (int i = 0; i < KPL; ++i) Sh[(i0 + i) * 128 + j0 + n] = col[n][i];
    });
}
}  // namespace

extern "C" {

int ns_image_mo_gdn_gates(ns_gpu* g, const float* alpha, float* beta, const float* dt, const float* a, float* eg, int64_t T, int64_t H, int64_t d) {
    NS_TRY
    if (T <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (T * H * d)), [=](sycl::id<1> id) {
        const int64_t i = id[0], h = (i / d) % H, t = i / (d * H);
        const float x = alpha[t * H + h] + dt[h];
        // ggml's softplus: x past 20 as it is
        const float sp = x > 20.0f ? x : sycl::log1p(sycl::exp(x));
        eg[i] = sycl::exp(sp * a[h]);
    });
    g->q.parallel_for(sycl::range<1>((size_t) (T * H)), [=](sycl::id<1> i) { beta[i] = 1.0f / (1.0f + sycl::exp(-beta[i])); });
    return 0;
    NS_CATCH
}

int ns_image_mo_conv_silu(ns_gpu* g, const float* x, float* state, const float* w, float* out, int64_t T, int64_t D, int k) {
    NS_TRY
    auto& q = g->q;
    q.parallel_for(sycl::range<2>(T, D), [=](sycl::id<2> id) {
        const int64_t t = id[0], c = id[1];
        auto in = [&](int64_t src) { return src >= 0 ? x[src * D + c] : state[(k - 1 + src) * D + c]; };
        float s = 0.f;
        for (int j = 0; j < k; ++j) s += in(t - (k - 1) + j) * w[c * k + j];   // inputs t-k+1 .. t
        out[t * D + c] = s / (1.f + sycl::exp(-s));
    });
    // then (the queue is in order) the new state: the last k-1 inputs, per channel
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
        const int64_t c = id[0];
        auto in = [&](int64_t src) { return src >= 0 ? x[src * D + c] : state[(k - 1 + src) * D + c]; };
        float last[8];
        for (int j = 0; j < k - 1; ++j) last[j] = in(T - (k - 1) + j);
        for (int j = 0; j < k - 1; ++j) state[j * D + c] = last[j];
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_expand(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t T, int64_t Hk, int64_t Hv, int64_t d) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * Hv * d)), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % d, h = (i / d) % Hv, t = i / (d * Hv);
        dst[i] = src[t * ss + (h % Hk) * d + c];
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_l2_norm(ns_gpu* g, float* x, int64_t rows, int64_t n, float eps) {
    NS_TRY
    constexpr int WG = 128;
    g->q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        float* xr = x + it.get_group(0) * n;
        float s = 0.f;
        for (int64_t c = it.get_local_id(0); c < n; c += WG) s += xr[c] * xr[c];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = 1.f / sycl::sqrt(s + eps);
        for (int64_t c = it.get_local_id(0); c < n; c += WG) xr[c] *= inv;
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_kda_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* eg, const float* beta, float* S, float* o, int64_t T,
                         int64_t H) {
    NS_TRY
    if (T <= 0) return 0;
    kda_scan<4>(g->q, q, k, v, eg, beta, S, o, T, H);
    return 0;
    NS_CATCH
}

int ns_image_mo_gdn_out(ns_gpu* g, const float* o, const float* z, int64_t zs, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps) {
    NS_TRY
    if (T <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t r = it.get_group(0), t = r / H, h = r % H;
        const float* x = o + r * d;
        float s = 0.0f;
        for (int64_t i = l; i < d; i += SG) s += x[i] * x[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) d + eps);
        for (int64_t i = l; i < d; i += SG) {
            const float zz = z[t * zs + h * d + i];
            y[r * d + i] = x[i] * inv * w[i] * (zz / (1.0f + sycl::exp(-zz)));
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t T, int64_t H, int64_t D, const float* w,
                             float eps, int64_t n_rot, float theta) {
    NS_TRY
    if (T <= 0) return 0;
    if (D % SG != 0 || n_rot > D || n_rot % 2) return ns_fail("ns_image_mo_qk_norm_rope: D a multiple of 16, n_rot even and at most D");
    const int64_t half_rot = n_rot / 2;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t th = it.get_group(0), t = th / H, h = th % H;
        const float* x = src + t * ss + h * hs;
        float* y = dst + (t * H + h) * D;
        float s = 0.0f;
        for (int64_t i = l; i < D; i += SG) s += x[i] * x[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) D + eps);
        const float pos = (float) t;
        for (int64_t i = l; i < D; i += SG) {
            const float v = x[i] * inv * w[i];
            if (i < n_rot) {
                // rotate-half over the first n_rot: pair (i, i + n_rot / 2)
                const int64_t k = i < half_rot ? i : i - half_rot;
                const float a = pos * sycl::pow(theta, -2.0f * (float) k / (float) n_rot);
                const float c = sycl::cos(a), sn = sycl::sin(a);
                const int64_t o = i < half_rot ? i + half_rot : i - half_rot;
                const float other = x[o] * inv * w[o];
                y[i] = i < half_rot ? v * c - other * sn : v * c + other * sn;
            } else {
                y[i] = v;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_gate_mul(ns_gpu* g, float* out, const float* q_full, int64_t gs, int64_t T, int64_t H, int64_t D) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * H * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % H, t = i / (D * H);
        const float z = q_full[t * gs + h * 2 * D + D + d];
        out[i] *= 1.0f / (1.0f + sycl::exp(-z));
    });
    return 0;
    NS_CATCH
}

int ns_image_mo_silu_mul(ns_gpu* g, const float* gate, const float* up, void* out, int64_t n) {
    NS_TRY
    sycl::half* o = (sycl::half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float a = gate[i];
        o[i] = (sycl::half) (a / (1.0f + sycl::exp(-a)) * up[i]);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
