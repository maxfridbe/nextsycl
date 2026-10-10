/* q3t.h: the Qwen3-TTS engine's own kernels (the language models' decode shares the MiniMax Music engine's mm3
 * kernels): the 12 Hz codec decoder's SnakeBeta, depthwise causal convolution, sliding-window attention, rotary
 * positions in float32 and layer-scaled residuals; small element-wise helpers. Float32 on the device; 0 or -1 (the
 * reason: ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SnakeBeta per channel of x [C, L]: x + sin^2(x e^a) / (e^b + 1e-9) */
int ns_audio_q3t_snake_beta(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* log_alpha, const float* log_beta, float* out);
/* A depthwise causal convolution of x [C, L] (K - 1 zeros ahead): out[c, t] = bias[c] + sum_k w[c, k] x[c, t - K + 1 + k] */
int ns_audio_q3t_dwconv(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* bias, int64_t K, float* out);
/* Causal attention over a window: query i sees keys (i - W, i]; q, k, v rows of H heads x D, qs / kvs floats apart;
 * out [L, H x D] */
int ns_audio_q3t_window_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t L, int64_t H, int64_t D, int64_t qs, int64_t kvs,
                             int64_t W, float* out);
/* Rotary positions (rotate-half) on rows of H heads x D, xs floats apart, positions p0 + row */
int ns_audio_q3t_rope(ns_gpu* g, float* x, int64_t xs, int64_t L, int64_t H, int64_t D, float theta, int64_t p0);
/* x [R, C] += scale[c] * y [R, C] */
int ns_audio_q3t_scale_add(ns_gpu* g, float* x, const float* y, const float* scale, int64_t R, int64_t C);
/* x = x * sigmoid(x) over n */
int ns_audio_q3t_silu(ns_gpu* g, float* x, int64_t n);
/* x = clamp(x, lo, hi) over n */
int ns_audio_q3t_clamp(ns_gpu* g, float* x, int64_t n, float lo, float hi);

#ifdef __cplusplus
}
#endif
