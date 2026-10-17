// vxt.cpp: the Voxtral TTS engine's own kernels (vxt.h). The codec's windows are 3 to 17 keys, so its attention is a
// work-group a (query, head) with a lane a feature; the rest are element-wise or a row a work-group.
#include "ns.h"
#include "ns_internal.hpp"
#include "vxt.h"

#include <sycl/sycl.hpp>
#include <cmath>

extern "C" {

int ns_audio_vxt_alibi_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t L, int64_t H, int64_t D, int64_t qs, int64_t kvs,
                            int64_t W, float* out) {
    NS_TRY
    if (D % 16 != 0 || D > 256 || W < 0 || W > 4096) return ns_fail("ns_audio_vxt_alibi_attn: D a multiple of 16 up to 256, a window of 0 to 4096");
    if (L <= 0 || H <= 0) return 0;
    const float scale = 1.0f / sycl::sqrt((float) D);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (L * H * D)), sycl::range<1>((size_t) D)), [=](sycl::nd_item<1> it) {
        const int64_t row = it.get_group(0), i = row / H, h = row % H, d = it.get_local_id(0);
        // upstream's get_alibi_slopes for a power of two: r^h, r = 2^(-8 / H)
        const float slope = sycl::exp2(-8.0f * (float) h / (float) H);
        const float qd = q[i * qs + h * D + d];
        const int64_t j0 = i - W > 0 ? i - W : 0;
        float mx = -INFINITY, sum = 0.0f, acc = 0.0f;
        for (int64_t j = j0; j <= i; ++j) {
            const float s = sycl::reduce_over_group(it.get_group(), qd * k[j * kvs + h * D + d], sycl::plus<float>()) * scale + slope * (float) (j - i);
            const float m = sycl::fmax(mx, s);
            const float c = sycl::exp(mx - m), p = sycl::exp(s - m);
            sum = sum * c + p;
            acc = acc * c + p * v[j * kvs + h * D + d];
            mx = m;
        }
        out[i * H * D + h * D + d] = acc / sum;
    });
    return 0;
    NS_CATCH
}

int ns_audio_vxt_rms_rows(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t C, const float* w, float eps) {
    NS_TRY
    if (rows <= 0 || C <= 0) return 0;
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        float* r = x + (int64_t) it.get_group(0) * xs;
        const int64_t l = it.get_local_id(0);
        float ss = 0.0f;
        for (int64_t c = l; c < C; c += WG) ss += r[c] * r[c];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) C + eps);
        for (int64_t c = l; c < C; c += WG) r[c] = r[c] * inv * w[c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_vxt_pad_left(ns_gpu* g, const float* x, int64_t C, int64_t L, int64_t P, int mode, float* out) {
    NS_TRY
    if (P < 0 || L <= 0 || (mode == 1 && P >= L) || mode < 0 || mode > 1) return ns_fail("ns_audio_vxt_pad_left: mode 0 or 1, a reflection shorter than the signal");
    if (C <= 0) return 0;
    const int64_t Lo = L + P;
    g->q.parallel_for(sycl::range<1>((size_t) (C * Lo)), [=](sycl::id<1> id) {
        const int64_t c = id[0] / Lo, t = id[0] % Lo;
        const int64_t s = t >= P ? t - P : (mode == 0 ? 0 : P - t);
        out[id[0]] = x[c * L + s];
    });
    return 0;
    NS_CATCH
}

int ns_audio_vxt_fm_rows(ns_gpu* g, const float* inp, const float* tproj, const float* hp, int64_t D, float* out) {
    NS_TRY
    if (D <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (6 * D)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / D, j = id[0] % D;
        const int64_t part = r % 3;
        out[id[0]] = part == 0 ? inp[j] : part == 1 ? tproj[j] : (r == 2 ? hp[j] : 0.0f);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
