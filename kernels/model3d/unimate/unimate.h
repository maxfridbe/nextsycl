/* unimate.h: the UniMate engine's own kernels (model3d/unimate), beside the shared diffusion kernels (nsd.h: its linear
 * layers, float32 on oneDNN). Bound by its crate by name (model3d/unimate/src/ffi.rs). Every buffer is float32 device
 * memory of g; every call is queued on g's queue (the caller syncs) and returns 0, or -1 with the reason in
 * ns_last_error(). The model runs in float32 throughout, as the reference does. */
#ifndef NS_MODEL3D_UNIMATE_H
#define NS_MODEL3D_UNIMATE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* RMS norm (LlamaRMSNorm / T5LayerNorm: x * rsqrt(mean(x^2) + eps), then * w) of the rows (m, h), m < M, h < H, of C
 * values at x + m x_row + h C; the adaLN modulation when scale / shift are given (y * (1 + scale) + shift, [C], only
 * for H = 1). The result goes to out + m out_row + h C (out may be x). w may not be NULL. */
int ns_model3d_unimate_rms(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t H, int64_t C, float eps, const float* w,
                           const float* scale, const float* shift, float* out, int64_t out_row);
/* x [M, C] += gate * y (gate [C], or NULL: a plain sum) */
int ns_model3d_unimate_gated_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, const float* gate);
/* x in place, n values: kind 0 SiLU, 1 GELU (erf) */
int ns_model3d_unimate_act(ns_gpu* g, float* x, int64_t n, int kind);
/* out [M, C] = act(x[:, :C]) * x[:, C:] for x [M, 2C]: kind 0 SiLU (SwiGLU), 1 GELU with the tanh approximation (T5's gated GELU) */
int ns_model3d_unimate_glu(ns_gpu* g, const float* x, int64_t M, int64_t C, int kind, float* out);
/* Spectral / sinusoidal rotary embedding, in place: for the nseg segments s (at x + s seg), each M rows of H heads of
 * D (row m at x + m row, head h at + h D): y = x cos + rotate_half(x) sin with the table row (m / div) % mod of
 * cos / sin [mod, D] */
int ns_model3d_unimate_rope(ns_gpu* g, float* x, int64_t row, int64_t seg, int64_t nseg, int64_t M, int64_t H, int64_t D, int64_t div,
                            int64_t mod, const float* cos, const float* sin);
/* Attention, float32. Groups g < G, each Sq queries and Skv keys; query i of group g is row g q_gs + i q_ss of q (q_row
 * floats a row), key / value j row g kv_gs + j kv_ss of k and v; head h's D features at + h D. out gets the query's row
 * index (g q_gs + i q_ss) at out_row floats a row. Keys j >= nvalid are masked; bias (NULL, or [H, Sq, Skv]) is added to
 * the scaled scores. out = softmax(q.k scale + bias) . v. D at most 128, Skv at most 128. */
int ns_model3d_unimate_attention(ns_gpu* g, const float* q, int64_t q_row, const float* k, int64_t k_row, const float* v, int64_t v_row,
                                 int64_t G, int64_t Sq, int64_t Skv, int64_t H, int64_t D, int64_t q_gs, int64_t q_ss, int64_t kv_gs,
                                 int64_t kv_ss, int64_t nvalid, float scale, const float* bias, float* out, int64_t out_row);
/* out [n, C] = src rows (src_off + r src_stride), r < n (and the same at dst: dst rows dst_off + r dst_stride) */
int ns_model3d_unimate_copy_rows(ns_gpu* g, const float* src, int64_t src_off, int64_t src_stride, float* dst, int64_t dst_off,
                                 int64_t dst_stride, int64_t n, int64_t C);
/* out [M, C1 + C2] = (a [M, C1] | b [M, C2]) */
int ns_model3d_unimate_cat2(ns_gpu* g, const float* a, int64_t C1, const float* b, int64_t C2, int64_t M, float* out);
/* The token grid: out [(F + 1) J, D] with row f J + j = (f = 0: tpos [J, D]; else the frame's joint MLP row (f - 1) J + j, and
 * for j = 0 root [F, D] row f - 1) + depth [J, D] + name [J, D] */
int ns_model3d_unimate_tokens(ns_gpu* g, const float* tpos, const float* joints, const float* root, const float* depth, const float* name,
                              int64_t F, int64_t J, int64_t D, float* out);
#ifdef __cplusplus
}
#endif
#endif
