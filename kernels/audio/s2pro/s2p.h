/* s2p.h: the Fish Audio S2 Pro engine's own kernels (audio/s2pro) - the decode's product over Q8_0 weights (int8 codes
 * and a half scale a block of 32, the GGUF's blocks split into two arrays at load), their expansion to float32 for the
 * prompt's wide products, and fish-speech's rotary positions (interleaved pairs, from a table of the angles' cosines and
 * sines: upstream keeps them in bfloat16) with an optional per-head RMS norm. The cached attention, embeddings and half
 * conversions are the MiniMax Music engine's (mm3.h), the codec's windowed attention and depthwise convolution the
 * Qwen3-TTS codec's (q3t.h), the norms, linear layers and Snake the shared diffusion kernels (nsd.h). Float32 on the
 * device; every call is queued on the GPU's queue and returns 0, or -1 (the reason: ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A few rows through a Q8_0 matrix: out[m, n] (+)= sum_k x[m, k] * q[n, k] * s[n, k / 32]      m < M <= 8
 * x float32 [M, K]; q int8 [N, K]; s half [N, K / 32]; out float32 rows `ldo` apart; acc 1: added to what out holds.
 * K a multiple of 256. */
int ns_audio_s2p_gemv_q8(ns_gpu* g, const float* x, int64_t M, int64_t K, const int8_t* q, const void* s, int64_t N, float* out, int64_t ldo,
                         int acc);
/* The same matrix in float32 for a prompt's wide products: out[n, k] = q[n, k] * s[n, k / 32] (out float32 [N, K]; K a
 * multiple of 32). Float32, not half: the activations of some prompts pass half's range (a down projection's input of
 * tens of thousands), and the product is oneDNN's float32 GEMM. */
int ns_audio_s2p_dequant_q8(ns_gpu* g, const int8_t* q, const void* s, int64_t N, int64_t K, float* out);
/* half -> float32 (n values): a half matrix for the same wide products */
int ns_audio_s2p_half_float(ns_gpu* g, const void* x, int64_t n, float* out);
/* Per-head RMS norm (weight float32 [D], NULL: none) then RoPE on interleaved pairs (2i, 2i + 1), in place: x rows of
 * H heads x D floats at x + r * xs, row r = b * S + s at position p0 + s; cs float32 [positions][D / 2][2] (cos, sin).
 * D even, at most 256. */
int ns_audio_s2p_norm_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* weight, float eps,
                           const float* cs, int64_t p0);

#ifdef __cplusplus
}
#endif
