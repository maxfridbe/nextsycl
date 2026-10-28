// unimate.cpp: UniMate's kernels (unimate.h): RMS norms with the adaLN modulation, gated residuals, SwiGLU / gated GELU,
// the rotary embeddings, attention (dense, a bias, a key mask) and the row shuffles of the token grid, all float32 as the
// reference runs. SYCL only, on the GPU's own queue; the linear layers are the shared diffusion kernels' (oneDNN).
#include "ns.h"
#include "ns_internal.hpp"
#include "unimate.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <limits>

namespace {
inline float silu(float v) {
    return v / (1.0f + sycl::exp(-v));
}
inline float gelu_erf(float v) {
    return 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
}
inline float gelu_tanh(float v) {  // T5's gelu_new
    return 0.5f * v * (1.0f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
}  // namespace

extern "C" {

int ns_model3d_unimate_rms(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t H, int64_t C, float eps, const float* w,
                           const float* scale, const float* shift, float* out, int64_t out_row) {
    NS_TRY
    constexpr int WG = 128, PER = 8;             // a row of up to 1024 values held in registers
    if (C > (int64_t) WG * PER) return ns_fail("ns_model3d_unimate_rms: rows longer than 1024");
    if (!w) return ns_fail("ns_model3d_unimate_rms: no weight");
    if (M <= 0 || H <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * H) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int64_t m = r / H, h = r % H;
        const int l = (int) it.get_local_id(0);
        const float* row = x + m * x_row + h * C;
        float v[PER];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            v[j] = i < C ? row[i] : 0.0f;
            s += v[j] * v[j];
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C + eps);
        sycl::group_barrier(it.get_group());     // every lane has read its row before any writes it (out may be x)
        float* o = out + m * out_row + h * C;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) {
                float y = w[i] * (v[j] * inv);
                if (scale) y = y * (1.0f + scale[i]);
                if (shift) y = y + shift[i];
                o[i] = y;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unimate_gated_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, const float* gate) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        x[r * C + c] += (gate ? gate[c] : 1.0f) * y[r * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unimate_act(ns_gpu* g, float* x, int64_t n, int kind) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        x[id[0]] = kind == 0 ? silu(x[id[0]]) : gelu_erf(x[id[0]]);
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unimate_glu(ns_gpu* g, const float* x, int64_t M, int64_t C, int kind, float* out) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        const float a = x[r * 2 * C + c];
        out[r * C + c] = (kind == 0 ? silu(a) : gelu_tanh(a)) * x[r * 2 * C + C + c];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unimate_rope(ns_gpu* g, float* x, int64_t row, int64_t seg, int64_t nseg, int64_t M, int64_t H, int64_t D, int64_t div,
                            int64_t mod, const float* cos, const float* sin) {
    NS_TRY
    if (M <= 0 || D % 2 != 0) return 0;
    const int64_t half = D / 2;
    g->q.parallel_for(sycl::range<3>((size_t) (nseg * M), (size_t) H, (size_t) half), [=](sycl::id<3> id) {
        const int64_t sg = id[0] / M, m = id[0] % M, h = id[1], i = id[2];
        float* p = x + sg * seg + m * row + h * D;
        const int64_t t = ((m / div) % mod) * D;
        const float a = p[i], b = p[i + half];
        p[i] = a * cos[t + i] - b * sin[t + i];
        p[i + half] = b * cos[t + i + half] + a * sin[t + i + half];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unimate_attention(ns_gpu* g, const float* q, int64_t q_row, const float* k, int64_t k_row, const float* v, int64_t v_row,
                                 int64_t G, int64_t Sq, int64_t Skv, int64_t H, int64_t D, int64_t q_gs, int64_t q_ss, int64_t kv_gs,
                                 int64_t kv_ss, int64_t nvalid, float scale, const float* bias, float* out, int64_t out_row) {
    NS_TRY
    constexpr int MAXD = 128, MAXS = 128;
    if (D > MAXD || Skv > MAXS) return ns_fail("ns_model3d_unimate_attention: head size or key count past 128");
    if (G <= 0 || Sq <= 0 || Skv <= 0) return 0;
    const int64_t nv = std::min(nvalid, Skv);
    g->q.parallel_for(sycl::range<3>((size_t) G, (size_t) H, (size_t) Sq), [=](sycl::id<3> id) {
        const int64_t gi = id[0], h = id[1], i = id[2];
        const float* qp = q + (gi * q_gs + i * q_ss) * q_row + h * D;
        float s[MAXS];
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < nv; ++j) {
            const float* kp = k + (gi * kv_gs + j * kv_ss) * k_row + h * D;
            float d = 0.0f;
            for (int64_t c = 0; c < D; ++c) d += qp[c] * kp[c];
            d *= scale;
            if (bias) d += bias[(h * Sq + i) * Skv + j];
            s[j] = d;
            mx = sycl::fmax(mx, d);
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < nv; ++j) {
            s[j] = sycl::exp(s[j] - mx);
            sum += s[j];
        }
        float acc[MAXD];
        for (int64_t c = 0; c < D; ++c) acc[c] = 0.0f;
        for (int64_t j = 0; j < nv; ++j) {
            const float p = s[j] / sum;
            const float* vp = v + (gi * kv_gs + j * kv_ss) * v_row + h * D;
            for (int64_t c = 0; c < D; ++c) acc[c] += p * vp[c];
        }
        float* op = out + (gi * q_gs + i * q_ss) * out_row + h * D;
        for (int64_t c = 0; c < D; ++c) op[c] = acc[c];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unimate_copy_rows(ns_gpu* g, const float* src, int64_t src_off, int64_t src_stride, float* dst, int64_t dst_off,
                                 int64_t dst_stride, int64_t n, int64_t C) {
    NS_TRY
    if (n <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) C), [=](sycl::id<2> id) {
        dst[(dst_off + (int64_t) id[0] * dst_stride) * C + id[1]] = src[(src_off + (int64_t) id[0] * src_stride) * C + id[1]];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unimate_cat2(ns_gpu* g, const float* a, int64_t C1, const float* b, int64_t C2, int64_t M, float* out) {
    NS_TRY
    if (M <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) (C1 + C2)), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        out[r * (C1 + C2) + c] = c < C1 ? a[r * C1 + c] : b[r * C2 + (c - C1)];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_unimate_tokens(ns_gpu* g, const float* tpos, const float* joints, const float* root, const float* depth, const float* name,
                              int64_t F, int64_t J, int64_t D, float* out) {
    NS_TRY
    if (F <= 0 || J <= 0 || D <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) ((F + 1) * J), (size_t) D), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        const int64_t f = r / J, j = r % J;
        float t;
        if (f == 0) t = tpos[j * D + c];
        else if (j == 0) t = root[(f - 1) * D + c];
        else t = joints[((f - 1) * J + j) * D + c];
        out[r * D + c] = (t + depth[j * D + c]) + name[j * D + c];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
