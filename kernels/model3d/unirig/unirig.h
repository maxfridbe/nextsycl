/* unirig.h: the UniRig engine's own kernels (model3d/unirig), beside the shared diffusion kernels (nsd.h: its linear
 * layers, float32 on oneDNN). Bound by its crate by name (model3d/unirig/src/ffi.rs). Every buffer is float32 device
 * memory of g unless named int32; every call is queued on g's queue (the caller syncs) and returns 0, or -1 with the
 * reason in ns_last_error(). The models run in float32 throughout. */
#ifndef NS_MODEL3D_UNIRIG_H
#define NS_MODEL3D_UNIRIG_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, C] = layer norm of the rows of x (row r at x + r * x_row), (* w + b when given); C at most 8192; out may
 * be x (with x_row = C) */
int ns_model3d_unirig_ln(ns_gpu* g, const float* x, int64_t x_row, int64_t M, int64_t C, float eps, const float* w, const float* b, float* out);
/* Per head layer norm, no affine, in place: rows of x [M, row], head h of n_heads at col + h * D (D at most 64) */
int ns_model3d_unirig_head_norm(ns_gpu* g, float* x, int64_t M, int64_t row, int64_t col, int64_t n_heads, int64_t D, float eps);
/* x in place, n values: each rounded to half and back (what PTv3's flash attention takes: UniRig's qkv.half()) */
int ns_model3d_unirig_round_half(ns_gpu* g, float* x, int64_t n);
/* x in place, n values: kind 0 GELU (erf), 1 ReLU */
int ns_model3d_unirig_act(ns_gpu* g, float* x, int64_t n, int kind);
/* x [n] += y [n] */
int ns_model3d_unirig_add(ns_gpu* g, float* x, const float* y, int64_t n);
/* x [M, C] = x * scale[c] + shift[c] (a folded batch norm) */
int ns_model3d_unirig_affine(ns_gpu* g, float* x, int64_t M, int64_t C, const float* scale, const float* shift);
/* Attention, float32, as the AnimateAnyMesh engine's (B sequences; row and head strides; D at most 1024), the
 * scores of a block of query rows through scratch (scratch_n floats, at least Skv) */
int ns_model3d_unirig_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                                int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row);
/* One decode step's attention for B rows (beams) over their histories in a slot pool: row b's query q [B, H * D],
 * its keys / values the pool rows kpool / vpool [*, H * D] named by slots[b * stride + t] (int32), t < len; out
 * [B, H * D]. len at most 8192, D at most 128. */
int ns_model3d_unirig_attn_slots(ns_gpu* g, const float* q, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride, int64_t len,
                                 int64_t B, int64_t H, int64_t D, float scale, float* out);
/* out [R, C] = x[idx[r]] (int32 idx), rows of x of stride x_row */
int ns_model3d_unirig_gather(ns_gpu* g, const float* x, int64_t x_row, const int32_t* idx, int64_t R, int64_t C, float* out);
/* out [R, C] = x rows written at dst[idx[r]]: out[idx[r]] = x[r] */
int ns_model3d_unirig_scatter(ns_gpu* g, const float* x, const int32_t* idx, int64_t R, int64_t C, float* out);
/* im2col of a sparse convolution: out [R, K * C] row r = (x[nbr[(r0 + r) * K + k]] or zeros when -1) for k < K;
 * x [N, C], nbr int32 [N, K] */
int ns_model3d_unirig_im2col(ns_gpu* g, const float* x, int64_t C, const int32_t* nbr, int64_t K, int64_t r0, int64_t R, float* out);
/* The skin head's per-head bone attention: for rows n < R and heads h < H, softmax over j < J of
 * q[n, h D ..] . k[j, h D ..] * scale; out [R, J, H] */
int ns_model3d_unirig_bone_attn(ns_gpu* g, const float* q, const float* k, int64_t R, int64_t J, int64_t H, int64_t D, float scale, float* out);
/* The skin head's input rows [R * J, 2 H]: (layer norm (aw, ab) of a[n, j, :H]) | (layer norm (vw, vb) of
 * vox[n, j] * ew + eb) - the attention's softmax beside the voxel prior's embedding */
int ns_model3d_unirig_skin_in(ns_gpu* g, const float* a, const float* vox, int64_t RJ, int64_t H, const float* aw, const float* ab,
                              const float* ew, const float* eb, const float* vw, const float* vb, float* out);

#ifdef __cplusplus
}
#endif
#endif
