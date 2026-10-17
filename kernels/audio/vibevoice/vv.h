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

/* vvlm.cpp - the products and the language model's decode step.
 *
 * out[m, n] (epi) x[m] . W[n] + bias[n] for M <= 20 rows of x (row m at x + m * ldx, K floats; ldx a multiple of 4),
 * W [N, K] in wt: 0 half, 1 bfloat16, 2 Q8_0 (w int8 codes [N, K], scales half [N, K / 32]; K a multiple of 32; else
 * of 8). bias float32 [N] or NULL. What epi does with v = the sum + bias, out rows ldo apart:
 *   STORE  out[m, n] = v              ACC  out[m, n] += v           GELU  out[m, n] = gelu(v) (exact, erf)
 *   SWIGLU rows 2i and 2i + 1 are gate i and up i: out[m, i] = silu(gate) * up
 *   RES    out[m, n] += gamma[n] * v (a layer-scaled residual) */
enum { NS_VV_STORE = 0, NS_VV_ACC = 1, NS_VV_SWIGLU = 2, NS_VV_GELU = 3, NS_VV_RES = 4 };
int ns_audio_vv_gemv(ns_gpu* g, const float* x, int64_t M, int64_t ldx, int64_t K, const void* w, int wt, const void* scales, int64_t N,
                     const float* bias, int epi, const float* gamma, float* out, int64_t ldo);
/* RMS norm of the rows of x [M, C] into out [M, C]: x / sqrt(mean(x^2) + eps) * w (w NULL: 1), then with scale /
 * shift (rows lds apart; NULL: none) v * (1 + scale[m]) + shift[m] */
int ns_audio_vv_rms_norm(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, const float* shift,
                         int64_t lds, float* out);
/* A decode step's RoPE and key / value store for R <= 2 rows, row r its own sequence at position pos[r] (device int32):
 * qkv rows ws apart hold [Hq heads | Hkv heads | Hkv heads] x D (q, k, v); q and k rotated (rotate-half, angle
 * pos * inv_freq[j]), q in place; k and v stored in half into row r's cache (k_r / v_r [Hkv, T_r, D]) at pos[r] */
int ns_audio_vv_rope_kv(ns_gpu* g, float* qkv, int64_t ws, int64_t R, const int32_t* pos, int64_t Hq, int64_t Hkv, int64_t D,
                        const float* inv_freq, void* k0, void* v0, int64_t T0, void* k1, void* v1, int64_t T1);
/* A decode step's attention for R <= 2 rows (row r: its sequence's cache k_r / v_r [Hkv, T_r, 128] half, positions
 * [0, pos[r]]): q rows qs apart, out [R, Hq * 128]. The keys in ns pieces (1-256), part float32 scratch of
 * R * Hq * ns * 130. D 128, 6 query heads a key head. */
int ns_audio_vv_attn_dec(ns_gpu* g, const float* q, int64_t qs, int64_t R, const int32_t* pos, const void* k0, const void* v0, int64_t T0,
                         const void* k1, const void* v1, int64_t T1, int64_t Hq, int64_t Hkv, int64_t D, int64_t ns, float* out, float* part);

/* Q8_0 codes [N, K] int8 and scales [N, K / 32] half -> out half [N, K] (a prompt's wide products' weights) */
int ns_audio_vv_dequant_q8(ns_gpu* g, const void* q, const void* d, int64_t N, int64_t K, void* out);
/* x [M, 2C] of interleaved gate / up pairs -> out [M, C] = silu(gate) * up */
int ns_audio_vv_swiglu_pairs(ns_gpu* g, const float* x, int64_t M, int64_t C, float* out);

#ifdef __cplusplus
}
#endif
