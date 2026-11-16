/* haru.h: the Haru-4B engine's own kernels (audio/haru): its language model's (Qwen3.5: Gated DeltaNet layers and
 * gated full attention) - the token rows, the DeltaNet's causal convolution and recurrence with their carried state,
 * the gated norm, attention's q / k norm with partial RoPE and its half cache, the attention split over pieces of
 * the keys and their merge with the output gate. The products, norms and the codec's blocks are the VibeVoice
 * engine's (vv.h), called and never changed (CONTRIBUTING rule 6). Float32 on the device; every call is queued on the
 * GPU's queue and returns 0, or -1 (the reason: ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* out[i, :] = table[idx[i], :] (bfloat16 [V, D] -> float32), rows ldo apart; idx int32 device [n] */
int ns_audio_haru_embed(ns_gpu* g, const void* table, int64_t D, const int32_t* idx, int64_t n, float* out, int64_t ldo);

/* The DeltaNet's causal depthwise convolution with SiLU over R rows: x rows ldx apart (C channels), w float32 [C, K],
 * state float32 [K - 1, C] (the inputs before row 0; updated to the last K - 1 inputs). out float32 [R, C] */
int ns_audio_haru_gdn_conv(ns_gpu* g, const float* x, int64_t ldx, int64_t R, int64_t C, int64_t K, const float* w, float* state, float* out);

/* The gated delta rule, row by row: qkv float32 rows ldq apart ([Hk * Dk queries | Hk * Dk keys | Hv * Dv values]),
 * ba rows ldb apart ([Hv b | Hv a]); per value head h (key head h / (Hv / Hk)): q, k L2-normed (eps 1e-6), q / sqrt(Dk);
 * beta = sigmoid(b), decay = exp(-exp(A_log) * softplus(a + dt_bias)); S [Dk, Dv] (float32, state + h * Dk * Dv) =
 * decay * S; S += k (x) beta (v - S^T k); out[r, h * Dv + j] = (S^T q)[j]. Dk = Dv = 128 */
int ns_audio_haru_gdn_rule(ns_gpu* g, const float* qkv, int64_t ldq, const float* ba, int64_t ldb, const float* A_log, const float* dt_bias, float* state,
                           int64_t R, int64_t Hk, int64_t Hv, int64_t Dk, int64_t Dv, float* out);

/* out[r, h * D + d] = w[d] * y / rms(y over the head) * silu(z[r * ldz + h * D + d]); y float32 [R, H * D] */
int ns_audio_haru_gated_norm(ns_gpu* g, const float* y, const float* z, int64_t ldz, const float* w, float eps, int64_t R, int64_t H, int64_t D, float* out);

/* Attention's inputs for R rows at positions p0 + r: qkv float32 rows ldq apart ([Hq x (D query | D gate) | Hkv x D
 * keys | Hkv x D values]); queries and keys RMS-normed per head (weights qw / kw float32 [D], the 1 + w already added)
 * and rotated on their first `rot` values (rotate-half, angle position * inv_freq[i], i < rot / 2); queries to q float32
 * [R, Hq * D], keys and values in half into kc / vc [Hkv, T, D] at p0 + r. D = 256 */
int ns_audio_haru_attn_prep(ns_gpu* g, const float* qkv, int64_t ldq, int64_t R, int64_t Hq, int64_t Hkv, int64_t D, int64_t rot, const float* qw,
                            const float* kw, float eps, const float* inv_freq, int64_t p0, float* q, void* kc, void* vc, int64_t T);

/* Causal attention over the half cache in pieces of `chunk` keys: row r (position p0 + r) attends to keys [0, p0 + r];
 * part float32 [R, Hq, P, D + 2] gets each piece's (max, sum, unnormalized output); P = ceil((p0 + R) / chunk).
 * D = 256, Hq a multiple of Hkv (at most 8 query heads a key head) */
int ns_audio_haru_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t R, int64_t Hq, int64_t Hkv, int64_t D, int64_t T, int64_t p0,
                       int64_t chunk, float* part);
/* The pieces merged: out[r, h * D + d] = (softmax-weighted sum) * sigmoid(gate[r * ldg + h * hs + d]) */
int ns_audio_haru_attn_merge(ns_gpu* g, const float* part, int64_t R, int64_t Hq, int64_t D, int64_t P, const float* gate, int64_t ldg, int64_t hs,
                             float* out);

#ifdef __cplusplus
}
#endif
