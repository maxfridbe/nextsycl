/* vv.h: the VibeVoice engine's own kernels (audio/vibevoice) - its language model's cached attention for Qwen2.5-1.5B's
 * heads (12 query heads over 2 key / value heads of 128), and the σ-VAE tokenizers' streaming pieces: the depthwise
 * causal convolution over time rows with its carried context, and that context's update. The rest is shared: the
 * MiniMax Music engine's small-batch products, key / value store and convolutions (mm3.h), Breeze's table RoPE
 * (bz2.h), the codec's layer-scaled residual (q3t.h), the diffusion kernels' norms, GELU, SwiGLU and linear layers
 * (nsd.h). Float32 on the device; every call is queued on the GPU's queue and returns 0, or -1 (the reason:
 * ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Causal attention against a half cache, as ns_audio_mm3_attn: q float32 rows r = b * S + s of Hq heads x D at q + r * qs;
 * the cache k / v half [B, Hkv, T, D] holds positions [0, p0 + S); query s (position p0 + s) attends to [0, p0 + s];
 * query head h reads key head h / (Hq / Hkv). out float32 [B * S, Hq * D]. D = 128 with 6 query heads a key head.
 * part: float32 scratch of ns_audio_vv_attn_scratch(...) floats (NULL when that is 0: a one-row decode over more than
 * 256 keys splits them). */
int64_t ns_audio_vv_attn_scratch(int64_t B, int64_t S, int64_t Hq, int64_t D, int64_t p0);
int ns_audio_vv_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t B, int64_t S, int64_t Hq,
                     int64_t Hkv, int64_t D, int64_t T, int64_t p0, float* out, float* part);
/* A depthwise causal convolution over time rows, the context carried: x [L, C] (rows of C channels) after the P = K - 1
 * rows of ctx [P, C] (the stream's last rows so far; zeros at its start):
 *   out[t, c] = bias[c] + sum_k w[c, k] src(t + k),   src(i) = ctx[i] for i < P, else x[i - P]
 * w float32 [C, K], bias float32 [C] or NULL; out [L, C] (not x). K at most 32. */
int ns_audio_vv_dwconv(ns_gpu* g, const float* x, const float* ctx, int64_t L, int64_t C, const float* w, const float* bias, int64_t K,
                       float* out);
/* The context after x: ctx [P, C] becomes the last P rows of [ctx; x] (x [L, C]) */
int ns_audio_vv_ctx_push(ns_gpu* g, float* ctx, int64_t P, const float* x, int64_t L, int64_t C);

#ifdef __cplusplus
}
#endif
