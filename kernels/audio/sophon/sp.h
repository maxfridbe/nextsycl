/* sp.h: the Sophon-Realtime engine's own kernels (audio/sophon), beside the shared diffusion kernels (nsd.h) and the
 * MiniMax Music engine's mm3 kernels it calls. The brain (Qwen3-32B) keeps its matrices in two layouts made on the card
 * at load from the checkpoint's FP8 (or bfloat16) rows:
 *
 *   q8: int8 [N, K] and a half scale every 32 weights [N, K / 32]:              w = q * s
 *   q4: nibbles [N, K / 2] (each 32-weight group's 16 bytes: byte j = weight j in the low nibble, weight 16 + j in
 *       the high one) and a half (scale, offset) pair every 32 weights [N, K / 32, 2]:   w = q * s + m   (q in 0..15)
 *
 * Bound by its crate by name (audio/sophon/src/ffi.rs). Every call is queued on the GPU's queue (the caller syncs) and
 * returns 0, or -1 with the reason in ns_last_error(). 16-bit buffers are IEEE half. */
#ifndef NS_AUDIO_SOPHON_H
#define NS_AUDIO_SOPHON_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* A matrix's rows into a layout (wt 0 q8, 1 q4): src [N, K] FP8 E4M3 (st 0; times scale[row], float32 [N]) or
 * bfloat16 (st 1; scale NULL). K a multiple of 32. q4 picks each group's scale and offset by least squares over
 * candidate ranges (the smallest squared error). */
int ns_audio_sophon_quant(ns_gpu* g, int wt, const void* src, int st, const float* scale, int64_t N, int64_t K, void* q, void* s);

/* out[m, n] (+)= sum_k x[m, k] w[n, k] for M <= 4 rows of x float32 [M, K] (rows K apart); w in layout wt (q, s);
 * out float32, rows ldo apart; acc 1: added to what out holds. K a multiple of 32. */
int ns_audio_sophon_gemv(ns_gpu* g, int wt, const void* q, const void* s, const float* x, int64_t M, int64_t K, int64_t N, float* out,
                         int64_t ldo, int acc);

/* Rows [r0, r0 + rows) of a matrix in layout wt made half: out [rows, K] */
int ns_audio_sophon_dequant(ns_gpu* g, int wt, const void* q, const void* s, int64_t K, int64_t r0, int64_t rows, void* out);

/* Attention of S query rows against a cache: q float32 rows of Hq heads x D (rows qs apart), row s at position p0 + s
 * attending to positions [0, p0 + s]; the cache kc / vc half [T, Hkv * D] (row = position) already holds them. Query
 * head h reads key / value head h / (Hq / Hkv). out float32 [S, Hq * D]. D = 128, Hq / Hkv <= 16. part: float32
 * scratch of ns_audio_sophon_attn_scratch(S, Hq, D, p0) floats (NULL when 0). */
int64_t ns_audio_sophon_attn_scratch(int64_t S, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0);
int ns_audio_sophon_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t D,
                         int64_t p0, float* out, float* part);
/* The cache's new rows: k / v float32 rows (row s at k + s * ks, Hkv * D values) into kc / vc half [T, Hkv * D] at
 * positions p0 + s */
int ns_audio_sophon_kv_store(ns_gpu* g, const float* k, const float* v, int64_t ks, int64_t S, int64_t HD, int64_t p0, void* kc, void* vc);

/* The audio encoder's 3 x 3 convolutions with stride 2 and padding 1 as a product: x float32 channels-last
 * [B, H, W, C] -> patches out [B * Ho * Wo, C * 9] (column c * 9 + kh * 3 + kw, the weight's [Co, Ci, 3, 3] order),
 * Ho = (H + 1) / 2, Wo = (W + 1) / 2 */
int ns_audio_sophon_im2col_s2(ns_gpu* g, const float* x, int64_t B, int64_t H, int64_t W, int64_t C, float* out);
/* out [A, C, B, D] = x [A, B, C, D] (float32) */
int ns_audio_sophon_swap12(ns_gpu* g, const float* x, int64_t A, int64_t B, int64_t C, int64_t D, float* out);
/* x [B, R, C] += y [R, C] (every b), float32 */
int ns_audio_sophon_add_rows(ns_gpu* g, float* x, const float* y, int64_t B, int64_t R, int64_t C);
/* out[i] = x[idx[i]] rows of C float32 (n rows; idx int32 on the device) */
int ns_audio_sophon_gather(ns_gpu* g, const float* x, const int32_t* idx, int64_t n, int64_t C, float* out);
/* The audio encoder's attention within windows: q, k, v float32 rows of H heads x D (rows `stride` apart), row i
 * attending to the rows of its window [W * (i / W), min(W * (i / W + 1), n)), unmasked; out float32 [n, H * D].
 * D <= 64, W <= 128. */
int ns_audio_sophon_attn_win(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t H, int64_t D, int64_t stride, int64_t W,
                             float* out);

#ifdef __cplusplus
}
#endif
#endif
