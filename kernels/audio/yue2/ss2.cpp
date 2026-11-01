// ss2.cpp: SheetSage2's encoder kernels (audio/yue2/src/sheet/mert.rs) - the three operations of MERT-v2 the shared
// diffusion kernels (nsd) do not have: the depthwise convolutions along time (the ConvNeXt layers' k = 7, the conformer
// convolution module's k = 31), ConvNeXt-V2's global response norm (a reduction over the whole sequence per channel,
// then over the channels), and the convolution module's GLU. Everything else of the encoder is nsd's. After nextcuda's
// ss2.cu.
//
// Rows are channels-last [T, C] float32, as the products write them. The norm's sums in float64 (30000 rows of up to 512
// channels at the first stage; the reference reduces in float32 in its own order).
#include "ns.h"
#include "ns_internal.hpp"
#include "yue.h"

#include <sycl/sycl.hpp>
#include <cstdint>

namespace {
using half = sycl::half;
}

extern "C" {

int ns_audio_yue2_dwconv(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out) {
    NS_TRY
    if (T * C <= 0) return 0;
    if (K % 2 == 0 || K > 63) return ns_fail("ns_audio_yue2_dwconv: an odd kernel up to 63");
    g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / C, c = i % C, h = K / 2;
        float a = bias ? bias[c] : 0.0f;
        for (int64_t j = 0; j < K; ++j) {
            const int64_t s = t + j - h;
            if (s >= 0 && s < T) a += w[c * K + j] * x[s * C + c];
        }
        out[i] = a;
    });
    return 0;
    NS_CATCH
}

int ns_audio_yue2_grn(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* gamma, const float* beta, void* out, float* part) {
    NS_TRY
    if (T * C <= 0) return 0;
    constexpr int WG = 64;
    // each channel's L2 norm along T: a work-group a channel, the rows in WG lanes
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) C * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t c = it.get_group(0);
        double s = 0.0;
        for (int64_t t = it.get_local_id(0); t < T; t += WG) {
            const double v = x[t * C + c];
            s += v * v;
        }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<double>());
        if (it.get_local_id(0) == 0) part[c] = (float) sycl::sqrt(s);
    });
    // the channels' mean of the norms, into part[C]
    g->q.single_task([=]() {
        double a = 0.0;
        for (int64_t c = 0; c < C; ++c) a += part[c];
        part[C] = (float) (a / (double) C);
    });
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C;
        const float n = part[c] / (part[C] + 1e-6f), v = x[i];
        o[i] = (half) (gamma[c] * (v * n) + beta[c] + v);
    });
    return 0;
    NS_CATCH
}

int ns_audio_yue2_to_half(ns_gpu* g, const float* x, int64_t n, void* out) {
    NS_TRY
    if (n <= 0) return 0;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { o[id] = (half) x[id]; });
    return 0;
    NS_CATCH
}

int ns_audio_yue2_glu(ns_gpu* g, const float* x, int64_t T, int64_t C, float* out) {
    NS_TRY
    if (T * C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], t = i / C, c = i % C;
        const float a = x[t * 2 * C + c], b = x[t * 2 * C + C + c];
        out[i] = a * (1.0f / (1.0f + sycl::exp(-b)));
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
