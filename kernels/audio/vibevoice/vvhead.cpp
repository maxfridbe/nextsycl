// vvhead.cpp: the VibeVoice engine's diffusion head pieces (vv.h) - its conditions for every step of a frame at once
// (so the adaLN modulations of all ten steps are one product, the weights read once), and DPM-Solver++'s step on the
// device (sched.rs's arithmetic; no read-back between the steps).
#include "ns.h"
#include "ns_internal.hpp"
#include "vv.h"

#include <sycl/sycl.hpp>

extern "C" {

int ns_audio_vv_silu_rows(ns_gpu* g, const float* c0, const float* temb, int64_t S, int64_t B, int64_t H, float* out) {
    NS_TRY
    if (S <= 0 || B <= 0 || H <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (S * B * H)), [=](sycl::id<1> id) {
        const int64_t i = id[0] % H, b = (id[0] / H) % B, s = id[0] / (H * B);
        const float v = c0[b * H + i] + temb[s * H + i];
        out[id[0]] = v / (1.0f + sycl::exp(-v));
    });
    return 0;
    NS_CATCH
}

int ns_audio_vv_dpm(ns_gpu* g, const float* v, int64_t L, float cfg, float* x, float* prev, float a_s, float s_s, float ratio, float ae, float inv_r0,
                    int second, float* x_in, int64_t ldin, float* lat, float* un, float scale, float bias) {
    NS_TRY
    if (L <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) L), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const float p = v[i], n = v[L + i];
        const float gd = n + cfg * (p - n);
        const float xv = x[i];
        // the data prediction, then the step (first order, or the midpoint second order from the previous one)
        const float d0 = a_s * xv - s_s * gd;
        float y;
        if (second) {
            const float d1 = inv_r0 * (d0 - prev[i]);
            y = ratio * xv - ae * d0 - 0.5f * ae * d1;
        } else {
            y = ratio * xv - ae * d0;
        }
        prev[i] = d0;
        x[i] = y;
        if (x_in) {
            x_in[i] = y;
            x_in[ldin + i] = y;
        }
        if (lat) lat[i] = y;
        if (un) un[i] = y / scale - bias;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
