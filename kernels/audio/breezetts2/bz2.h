/* bz2.h: the Breeze TTS 2 engine's own kernels (audio/breezetts2) - its T5Gemma2 text encoder's bidirectional
 * attention over a symmetric window and GELU-gated MLP, and a per-head RMS norm + RoPE from a frequency table (the
 * text encoder's two RoPEs, the depth decoder's llama3-scaled one). The backbone and depth decoder's cached decode is
 * the MiniMax Music engine's (mm3.h), the norms and linear layers the shared diffusion kernels (nsd.h). Float32 on the
 * device; every call is queued on the GPU's queue and returns 0, or -1 (the reason: ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-head RMS norm (weight float32 [D], NULL: none) then RoPE (rotate-half: pairs (i, D/2 + i)), in place: x rows of
 * H heads x D floats at x + r * xs, row r = b * S + s at position p0 + s; angle position * inv_freq[i] (float32
 * [D / 2]). D at most 256. */
int ns_audio_bz2_head_norm_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* weight, float eps,
                                const float* inv_freq, int64_t p0);
/* Bidirectional attention over a symmetric window: query i sees keys j with i - left < j < i + right (the whole
 * sequence when both are at least L); q rows of Hq heads x D at q + i * qs, k / v rows of Hkv heads x D at
 * k / v + j * kvs (query head h reads key head h / (Hq / Hkv)); scores * scale; out [L, Hq x D]. D at most 256. */
int ns_audio_bz2_attn_window(ns_gpu* g, const float* q, const float* k, const float* v, int64_t L, int64_t Hq, int64_t Hkv, int64_t D,
                             int64_t qs, int64_t kvs, int64_t left, int64_t right, float scale, float* out);
/* The gated MLP's middle: out [R, F] = gelu_tanh(gu[r, f]) * gu[r, F + f] (gu [R, 2F]: the gate, then the up) */
int ns_audio_bz2_gelu_mul(ns_gpu* g, const float* gu, int64_t R, int64_t F, float* out);

#ifdef __cplusplus
}
#endif
