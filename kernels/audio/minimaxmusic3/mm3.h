/* mm3.h: the MiniMax Music 3 engine's own kernels (audio/minimaxmusic3), beside the shared diffusion kernels (nsd.h).
 * Bound by its crate by name (audio/minimaxmusic3/src/ffi.rs). 16-bit buffers are IEEE half; every call is queued on
 * the GPU's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_AUDIO_MM3_H
#define NS_AUDIO_MM3_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* A few rows through a matrix, the decode's product (bound by reading the matrix once):
 *   out[m, n] (+)= (sum_k x[m, k] w[n, k]) * scale[n] + bias[n]        m < M <= 8
 * x float32 [M, K]; w [N, K]: half (wt 0) or int8 (wt 1); scale float32 [N] or NULL (1); bias float32 [N] or NULL;
 * out float32 [M, N], rows `ldo` apart; acc 1: added to what out holds. K a multiple of 256. */
int ns_audio_mm3_gemv(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* w, int wt, const float* scale, int64_t N,
                      const float* bias, float* out, int64_t ldo, int acc);

/* Causal attention against a cache: q float32, row r = b * S + s holds Hq heads x D at q + r * qs; the cache k / v
 * half [B, Hkv, T, D] (T the capacity) already holds positions [0, p0 + S); query s (position p0 + s) attends to
 * positions [0, p0 + s]. Query head h reads key / value head h / (Hq / Hkv). out float32 [B * S, Hq * D].
 * D = 128 with 4 query heads a key head, or D = 256 with one. part: float32 scratch of
 * ns_audio_mm3_attn_scratch(...) floats (NULL when that is 0). */
int64_t ns_audio_mm3_attn_scratch(int64_t B, int64_t S, int64_t Hq, int64_t D, int64_t p0);
int ns_audio_mm3_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t B, int64_t S, int64_t Hq,
                      int64_t Hkv, int64_t D, int64_t T, int64_t p0, float* out, float* part);
/* The cache's new rows: k / v float32 (row r = b * S + s at k + r * ks, Hkv heads x D) into kc / vc half
 * [B, Hkv, T, D] at positions p0 + s */
int ns_audio_mm3_kv_store(ns_gpu* g, const float* k, const float* v, int64_t ks, int64_t B, int64_t S, int64_t Hkv, int64_t D,
                          int64_t T, int64_t p0, void* kc, void* vc);
/* Qwen3's per-head RMS norm (weight float32 [128]) then RoPE on pairs (i, 64 + i), in place: x float32 rows of H heads
 * x 128 at x + r * xs, row r = b * S + s at position p0 + s, angle position * inv_freq[i] (float32 [64]) */
int ns_audio_mm3_qk_norm_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, const float* weight, float eps,
                              const float* inv_freq, int64_t p0);
/* The flow transformer's partial RoPE, in place: x half rows of H heads x D at x + r * xs, row r at position r % S;
 * the first `rot` features of each head rotated on pairs (i, rot / 2 + i) by position * inv_freq[i] (float32
 * [rot / 2]), computed in float32 */
int ns_audio_mm3_rope_partial(ns_gpu* g, void* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, int64_t rot,
                              const float* inv_freq);
/* out[c * ldo + j] = scale * sum_i table[idx[i], j] (j < D) for `copies` rows c: table half [R, D]; idx (n <= 8) */
int ns_audio_mm3_embed(ns_gpu* g, const void* table, int64_t D, const int32_t* idx, int n, float scale, float* out, int64_t ldo,
                       int copies);
/* out[j] = scale * sum_l w[l] h[l * D + j] (n <= 8 rows of h float32 [n, D]) */
int ns_audio_mm3_mix(ns_gpu* g, const float* h, int n, int64_t D, const float* w, float scale, float* out);
/* conversions and the int8 weights: float32 -> half; bfloat16 bits -> half; per row of w [N, K] float32 scale[r] =
 * max|w[r]| / 127 (at least 1e-30), q = round(w / scale) (ties to even, clamped to [-127, 127]) */
int ns_audio_mm3_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
int ns_audio_mm3_bf16_to_half(ns_gpu* g, const uint16_t* x, void* out, int64_t n);
int ns_audio_mm3_bf16_to_float(ns_gpu* g, const uint16_t* x, float* out, int64_t n);
int ns_audio_mm3_quant_rows(ns_gpu* g, const float* w, int64_t N, int64_t K, int8_t* q, float* scale);
/* x float32 += y float32 (n values) */
int ns_audio_mm3_add(ns_gpu* g, float* x, const float* y, int64_t n);
/* The flow transformer's input rows for both guidance passes: out float32 [2, L, 2 C + Cc] = [lat, 0, cond] (pass 0)
 * and [lat, 0, 0] (pass 1); lat [L, C], cond [L, Cc] float32 */
int ns_audio_mm3_dit_in(ns_gpu* g, const float* lat, const float* cond, int64_t L, int64_t C, int64_t Cc, float* out);
/* The guided Euler step: lat += dt * (vu + cfg * (vc - vu)) over n values */
int ns_audio_mm3_cfg_step(ns_gpu* g, float* lat, const float* vc, const float* vu, int64_t n, float cfg, float dt);
/* x[i] = a * p[i] + b * q[i] (n values): the overlap's blend toward the previous window */
int ns_audio_mm3_blend(ns_gpu* g, float* x, const float* p, const float* q, int64_t n, float a, float b);
/* out[j, c] = x[c, src(j)] for j < Lo: x float32 [C, Li] channels first -> rows [Lo, C], src(j) = nearest
 * (PyTorch's mode="nearest": floor(j * (Li / Lo)) in float32, at most Li - 1) */
int ns_audio_mm3_nearest_rows(ns_gpu* g, const float* x, int64_t C, int64_t Li, int64_t Lo, float* out);
/* out [C, R] = x [R, C] transposed (float32) */
int ns_audio_mm3_transpose(ns_gpu* g, const float* x, int64_t R, int64_t C, float* out);
/* x = tanh(x), in place (n values) */
int ns_audio_mm3_tanh(ns_gpu* g, float* x, int64_t n);

#ifdef __cplusplus
}
#endif
#endif
