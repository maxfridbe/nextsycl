/* skintokens.h: the SkinTokens / TokenRig engine's own kernels (model3d/skintokens), beside the shared diffusion kernels
 * (nsd.h: its linear layers, float32 on oneDNN). Bound by its crate by name (model3d/skintokens/src/ffi.rs). Every
 * buffer is float32 device memory of g unless named int32; every call is queued on g's queue (the caller syncs) and
 * returns 0, or -1 with the reason in ns_last_error(). The model runs in float32 throughout. */
#ifndef NS_MODEL3D_SKINTOKENS_H
#define NS_MODEL3D_SKINTOKENS_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, C] = layer norm of the rows of x (row r at x + r * x_row), (* w + b when given); C at most 8192; out may
 * be x (with x_row = C) */
int ns_model3d_skintokens_ln(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, const float* b, float* out);
/* out [M, C] = x * rsqrt(mean(x^2) + eps) * w (RMS norm, torch's nn.RMSNorm / Qwen3's); out may be x (x_row = C) */
int ns_model3d_skintokens_rms(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, float* out);
/* x in place, n values: kind 0 GELU (erf), 1 sigmoid */
int ns_model3d_skintokens_act(ns_gpu* g, float* x, int64_t n, int kind);
/* x [n] += y [n] */
int ns_model3d_skintokens_add(ns_gpu* g, float* x, const float* y, int64_t n);
/* out [M, I] = silu(gu[m, i]) * gu[m, I + i] (gu rows [gate | up], 2 I wide) */
int ns_model3d_skintokens_swiglu(ns_gpu* g, const float* gu, int64_t M, int64_t I, float* out);
/* out [M] = (x [M, C] . w [C] + b) then sigmoid when act = 1 */
int ns_model3d_skintokens_rowdot(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float b, int act, float* out);
/* out [R, C] = x[idx[r]] (int32 idx), rows of x of stride x_row */
int ns_model3d_skintokens_gather(ns_gpu* g, const float* x, int64_t x_row, const int32_t* idx, int64_t R, int64_t C, float* out);
/* The decoder's attention input: qkv rows [M, (Hq + 2 Hkv) D] (q heads, k heads, v heads). Per head RMS norm (weights qw / kw,
 * D wide) then rotary embedding (rotate-half pairs (i, i + D/2), angle pos[m] * inv_freq[i], inv_freq [D/2]) on q and k, in place;
 * then k and v (after the norm and rotation) copied to row slot[m] of kpool / vpool [*, Hkv D]. */
int ns_model3d_skintokens_qkv_post(ns_gpu* g, float* qkv, int64_t M, int64_t Hq, int64_t Hkv, int64_t D, float eps, const float* qw, const float* kw,
                                   const int32_t* pos, const float* inv_freq, const int32_t* slot, float* kpool, float* vpool);
/* Attention, float32 (B sequences; row and head strides; D at most 1024; Hkv key / value heads shared by H / Hkv query heads each),
 * the scores of a block of query rows through scratch (scratch_n floats, at least Skv) */
int ns_model3d_skintokens_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                    const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv,
                                    int64_t D, int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row);
/* One decode step's attention for B rows (beams) over their histories in a slot pool: row b's query q [B, q_row] (head h at h D),
 * its keys / values the pool rows kpool / vpool [*, Hkv D] named by slots[b * stride + t] (int32), t < len; out [B, H D].
 * len at most 8192, D at most 128. */
int ns_model3d_skintokens_attn_slots(ns_gpu* g, const float* q, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride,
                                     int64_t len, int64_t B, int64_t H, int64_t Hkv, int64_t D, float scale, float* out);

#ifdef __cplusplus
}
#endif
#endif
