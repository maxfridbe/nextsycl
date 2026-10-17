/* vv.h: the VibeVoice engine's own kernels (audio/vibevoice), its silo (CONTRIBUTING rule 6: copies of shared kernels
 * changed for this model, and its own): the products (half, bfloat16, Q8_0 and float32 weights; a few rows, or many on
 * the matrix engines), the language model's decode step (RoPE + key / value store, split attention) and a prompt's
 * cached attention, the diffusion head's conditions and DPM-Solver++ step, the σ-VAE tokenizers' blocks and carried
 * rows. The rest is shared: the MiniMax Music engine's key / value store and conversions (mm3.h), Breeze's table RoPE
 * (bz2.h), the codec's SiLU (q3t.h), the diffusion kernels' norms and linear layers (nsd.h) - a prompt's pass and
 * the load. Float32 on the device; every call is queued on the GPU's queue and returns 0, or -1 (the reason:
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
/* vvlm.cpp - the products and the language model's decode step.
 *
 * out[m, n] (epi) x[m] . W[n] + bias[n] for M <= 20 rows of x (row m at x + m * ldx, K floats; ldx a multiple of 4),
 * W [N, K] in wt: 0 half, 1 bfloat16, 2 Q8_0 (w int8 codes [N, K], scales half [N, K / 32]; K a multiple of 32; else
 * of 8). bias float32 [N] or NULL. What epi does with v = the sum + bias, out rows ldo apart:
 *   STORE  out[m, n] = v              ACC  out[m, n] += v           GELU  out[m, n] = gelu(v) (exact, erf)
 *   SWIGLU rows 2i and 2i + 1 are gate i and up i: out[m, i] = silu(gate) * up
 *   RES    out[m, n] += gamma[m * ldg + n] * v (a layer-scaled residual; ldg 0: one scale for every row) */
enum { NS_VV_STORE = 0, NS_VV_ACC = 1, NS_VV_SWIGLU = 2, NS_VV_GELU = 3, NS_VV_RES = 4 };
int ns_audio_vv_gemv(ns_gpu* g, const float* x, int64_t M, int64_t ldx, int64_t K, const void* w, int wt, const void* scales, int64_t N,
                     const float* bias, int epi, const float* gamma, int64_t ldg, float* out, int64_t ldo);
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

/* vvcodec.cpp - the σ-VAE tokenizers.
 * The tiled product: out[m, n] (epi) x[m] . W[n] + bias for any M (rows of x ldx apart: windows may overlap), W [N, K]
 * half (0), bfloat16 (1) or float32 (3); epi as ns_audio_vv_gemv's but SwiGLU. */
int ns_audio_vv_gemm(ns_gpu* g, const float* x, const void* xh, const void* xl, int64_t M, int64_t ldx, int64_t K, const void* w, int wt, int64_t N,
                     const float* bias, int epi, const float* gamma, int64_t ldg, float* out, int64_t ldo, float* part, int64_t part_len);
/* The float32 scratch (part, part_len floats) the matrix engines' form of a product splits K with; 0: none */
int64_t ns_audio_vv_gemm_scratch(int64_t M, int64_t N, int64_t K);
/* x [n] float32 -> hi, lo bfloat16 [n]: hi = x rounded, lo = (x - hi) rounded (the matrix engines' form of x: with
 * xh / xl given, ns_audio_vv_gemm runs bfloat16 weights on them, K a multiple of 16, ldx of 8, rows past M readable) */
int ns_audio_vv_split(ns_gpu* g, const float* x, int64_t n, void* hi, void* lo);
/* A block's first norm after its depthwise convolution's carried rows: cur [P + L, C] = prev's rows [prev_off,
 * prev_off + P) (the previous call's buffer), then norm(x[t]) * w for the L rows of x */
int ns_audio_vv_norm_hist(ns_gpu* g, const float* prev, int64_t prev_off, const float* x, int64_t L, int64_t C, int64_t P, const float* w, float eps,
                          float* cur);
/* The rest of a block's mixing, a row t at a time: x[t] += gamma * (dw_b + sum_k dw_w[c, k] cur[t + k]) (the depthwise
 * causal convolution of K taps over cur [K - 1 + L, C]), then h[t] = norm(x[t]) * w2 (the MLP's input) */
int ns_audio_vv_block_mix(ns_gpu* g, const float* cur, float* x, int64_t L, int64_t C, int64_t K, const float* dw_w, const float* dw_b, const float* gamma,
                          const float* w2, float eps, float* h);
/* buf [P + L, C]: rows [L, L + P) moved to [0, P) (a convolution's carried input rows, after it ran) */
int ns_audio_vv_shift(ns_gpu* g, float* buf, int64_t P, int64_t L, int64_t C);

/* vvhead.cpp - the diffusion head.
 * out [S * B, H]: row s * B + b = silu(c0[b] + temb[s]) (c0 [B, H]: the branches' conditions; temb [S, H]: the steps'
 * timestep embeddings) - every step's conditions at once */
int ns_audio_vv_silu_rows(ns_gpu* g, const float* c0, const float* temb, int64_t S, int64_t B, int64_t H, float* out);
/* One DPM-Solver++ step (sched.rs's) on a latent x [L]: v [2, L] the two branches' v-predictions, mixed by cfg
 * (n + cfg (p - n)); d0 = a_s x - s_s v; x = ratio x - ae d0 (second: - 0.5 ae inv_r0 (d0 - prev)); prev = d0. Then x
 * copied to the head's next input rows (x_in, both rows ldin apart), lat, and un = x / scale - bias (each NULL: not) */
int ns_audio_vv_dpm(ns_gpu* g, const float* v, int64_t L, float cfg, float* x, float* prev, float a_s, float s_s, float ratio, float ae, float inv_r0,
                    int second, float* x_in, int64_t ldin, float* lat, float* un, float scale, float bias);

#ifdef __cplusplus
}
#endif
