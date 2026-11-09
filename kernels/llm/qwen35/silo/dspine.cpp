// dspine.cpp: the qwen35 silo's kernels for the DSpine draft (silo.h: ns_q35_dsp_*; llm/qwen35/src/dspine.rs says what
// the draft is). The draft's products and attention are the engine's own; what is new is the code bridge between its
// layers (a handful of small row-wise steps) and the serial refinement of the last layer's rows, which scores each
// row's 16 candidates against the previous row's choice. All small: rows are at most 8, a vector 512 to 5,120 wide.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <cstdint>

namespace {
using half = sycl::half;
constexpr int WG = 256;
constexpr int PWG = 128;   // the refinement's work-group: a (row, sender) pair's 512-wide products
constexpr float F32_TINY = 1.1754943508222875e-38f;

inline float sigmoidf(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
}  // namespace

extern "C" {

int ns_q35_dsp_bridge_norm(ns_gpu* g, const float* x, float* y, float* rms, int64_t rows, int64_t C, float eps) {
    NS_TRY
    if (rows <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const float* xr = x + r * C;
        float s = 0.f;
        for (int64_t i = tid; i < C; i += WG) s += xr[i] * xr[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float rm = sycl::sqrt(sycl::fmax(s / (float) C, F32_TINY));
        if (tid == 0) rms[r] = rm;
        const float inv = 1.0f / (rm + eps);
        for (int64_t i = tid; i < C; i += WG) y[r * C + i] = xr[i] * inv;
    });
    return 0;
    NS_CATCH
}

int ns_q35_dsp_codes(ns_gpu* g, const float* proj, const float* anchor_code, float* codes, int64_t rows, int64_t C, float sqrt_c, float eps) {
    NS_TRY
    if (rows <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        if (r == 0) {
            for (int64_t i = tid; i < C; i += WG) codes[i] = anchor_code[i];
            return;
        }
        const float* pr = proj + r * C;
        float s = 0.f;
        for (int64_t i = tid; i < C; i += WG) s += pr[i] * pr[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float sc = sqrt_c / (sycl::sqrt(s) + eps);
        for (int64_t i = tid; i < C; i += WG) codes[r * C + i] = pr[i] * sc;
    });
    return 0;
    NS_CATCH
}

int ns_q35_dsp_message(ns_gpu* g, const float* packed, const float* gate_bias, float* msg, int64_t rows, int64_t R) {
    NS_TRY
    if (rows <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (rows * R)), [=](sycl::id<1> id) {
        const int64_t i = id[0] / R, r = id[0] % R;
        if (i == 0) {
            msg[id[0]] = 0.f;
            return;
        }
        const float* cur = packed + i * 3 * R;
        const float* prev = packed + (i - 1) * 3 * R;
        msg[id[0]] = sigmoidf(cur[r] + prev[R + r] + gate_bias[r]) * prev[2 * R + r];
    });
    return 0;
    NS_CATCH
}

int ns_q35_dsp_write(ns_gpu* g, const float* raw, const float* delta, const float* rms, float* out, int64_t rows, int64_t H) {
    NS_TRY
    if (rows <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (rows * H)), [=](sycl::id<1> id) {
        const int64_t i = id[0] / H;
        out[id[0]] = i == 0 ? raw[id[0]] : raw[id[0]] + rms[i] * delta[id[0]];
    });
    return 0;
    NS_CATCH
}

int ns_q35_dsp_base(ns_gpu* g, const float* raw, const float* norm_w, const void* rows_f16, float* base, int64_t n, int64_t K, int64_t H) {
    NS_TRY
    if (n <= 0) return 0;
    const half* rows = static_cast<const half*>(rows_f16);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * K) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t nk = it.get_group(0), r = nk / K;
        const int tid = (int) it.get_local_id(0);
        float s = 0.f;
        for (int64_t h = tid; h < H; h += WG) s += raw[r * H + h] * norm_w[h] * (float) rows[nk * H + h];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (tid == 0) base[nk] = s;
    });
    return 0;
    NS_CATCH
}

int ns_q35_dsp_pick(ns_gpu* g, const float* base, const float* rms, const float* q, int64_t qs, const int32_t* cand, int32_t anchor, const void* tsv_f16,
                    const void* tg_f16, const float* gate_bias, int32_t* idx, int64_t n, int64_t K, int64_t R) {
    NS_TRY
    if (n <= 0) return 0;
    if (R > 512 || K > 64) return ns_fail("ns_q35_dsp_pick: rank at most 512, at most 64 candidates");
    const half* tsv = static_cast<const half*>(tsv_f16);
    const half* tg = static_cast<const half*>(tg_f16);
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> m(sycl::range<1>(512), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * K) * PWG), sycl::range<1>(PWG)), [=](sycl::nd_item<1> it) {
            const int64_t nk = it.get_group(0), r = nk / K, k = nk % K;
            const int tid = (int) it.get_local_id(0);
            // the sender: the anchor for the first row, else the previous row's k-th candidate
            const int64_t sid = r == 0 ? (int64_t) anchor : (int64_t) cand[(r - 1) * K + k];
            const half* ts = tsv + sid * 2 * R;
            for (int64_t i = tid; i < R; i += PWG) m[i] = sigmoidf(q[r * qs + i] + (float) ts[i] + gate_bias[i]) * (float) ts[R + i];
            sycl::group_barrier(it.get_group());
            float best = 0.f;
            int bi = 0;
            for (int k2 = 0; k2 < K; ++k2) {
                const half* t2 = tg + (int64_t) cand[r * K + k2] * R;
                float s = 0.f;
                for (int64_t i = tid; i < R; i += PWG) s += m[i] * (float) t2[i];
                s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
                const float score = base[r * K + k2] + rms[r] * s;
                if (k2 == 0 || score > best) {
                    best = score;
                    bi = k2;
                }
            }
            if (tid == 0) idx[nk] = bi;
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
