/* vcp.h: the VoxCPM2 engine's own kernels (its language models' decode shares the MiniMax Music engine's mm3 kernels):
 * rotary positions from a frequency table (MiniCPM4's long RoPE), attention within short groups of rows (the local
 * encoder's and the local DiT's sequences: every row sees every row of its group), the AudioVAE's Snake and dilated
 * causal depthwise convolution, a per-channel scale and bias, the scalar quantizer. Float32 on the device; 0 or -1
 * (the reason: ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Rotate-half rotary positions on rows of H heads x D (xs floats apart): row r at position p0 + r % S, pair (i, i + D/2)
 * turned by position * inv_freq[i] */
int ns_audio_vcp_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0);
/* Attention within B groups of S rows (none masked): q rows of Hq heads x D qs floats apart, k / v rows of Hkv heads
 * kvs apart (query head h reads key head h / (Hq / Hkv)); out [B * S, Hq x D]; D <= 128 */
int ns_audio_vcp_local_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t B, int64_t S, int64_t Hq, int64_t Hkv, int64_t D,
                            int64_t qs, int64_t kvs, float* out);
/* Snake on x [C, L]: x + sin^2(alpha x) / (alpha + 1e-9) */
int ns_audio_vcp_snake(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, float* out);
/* A causal depthwise convolution of x [C, L], dilation dil ((K - 1) * dil zeros ahead): w [C, K], bias [C] or NULL */
int ns_audio_vcp_dwconv(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* bias, int64_t K, int64_t dil, float* out);
/* x [C, L] = x * scale[c] + bias[c] */
int ns_audio_vcp_affine(ns_gpu* g, float* x, int64_t C, int64_t L, const float* scale, const float* bias);
/* The scalar quantizer's middle: x = round(tanh(x) * s) / s over n */
int ns_audio_vcp_fsq(ns_gpu* g, float* x, int64_t n, float s);

#ifdef __cplusplus
}
#endif
