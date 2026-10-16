/* aam.h: the AnimateAnyMesh engine's own kernels (model3d/animateanymesh), beside the shared diffusion kernels (nsd.h:
 * its linear layers, float32 on oneDNN). Bound by its crate by name (model3d/animateanymesh/src/ffi.rs). Every buffer
 * is float32 device memory of g; every call is queued on g's queue (the caller syncs) and returns 0, or -1 with the
 * reason in ns_last_error(). The model runs in float32 throughout, as the reference does. */
#ifndef NS_MODEL3D_AAM_H
#define NS_MODEL3D_AAM_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Layer norm of x [M, C] (C at most 8192), with two streams: row r is stream 0 when (r % S) < T, else stream 1 (S = T:
 * one stream). Stream s: y = norm(x) (* w_s + b_s when w_s is given) (* (1 + scale_s) + shift_s when scale_s is given:
 * the DiT's adaLN). out may be x. */
int ns_model3d_aam_ln(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, int64_t S, int64_t T,
                      const float* w0, const float* b0, const float* scale0, const float* shift0,
                      const float* w1, const float* b1, const float* scale1, const float* shift1, float* out);
/* x [M, C] += gate_s * y (gate_s float32 [C] of the row's stream, as ns_model3d_aam_ln; NULL: a plain add) */
int ns_model3d_aam_gated_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, int64_t S, int64_t T, const float* gate0,
                             const float* gate1);
/* x in place, n values: kind 0 GELU (erf), 1 SiLU, 2 quick GELU (x sigmoid(1.702 x)) */
int ns_model3d_aam_act(ns_gpu* g, float* x, int64_t n, int kind);
/* out [M, C] = x[:, :C] * gelu(x[:, C:]) for x [M, 2C] (GEGLU, erf) */
int ns_model3d_aam_geglu(ns_gpu* g, const float* x, int64_t M, int64_t C, float* out);
/* Attention, float32: B sequences; sequence b's query rows i < Sq at q + (b Sq + i) q_row, key / value rows j < Skv at
 * k + (b Skv + j) k_row (v alike); head h's features at + h q_head (k_head, v_head): D for queries and keys, Dv for
 * values. out[(b Sq + i) out_row + h Dv + d] = softmax_j(q.k scale (j <= i when causal)) . v. The scores of a block
 * of query rows go through `scratch` (scratch_n floats, at least Skv): the rows a launch takes are what fits.
 * D at most 1024. */
int ns_model3d_aam_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                             const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                             int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row);
/* Attention over a graph's neighbours (one head): row i of q [N, D] over the rows cols[rowptr[i] .. rowptr[i+1]) of k
 * and v [*, D]; a row without neighbours gives zeros (what the reference's fully masked rows give). D at most 1024,
 * at most 1024 neighbours a row. out [N, D]. */
int ns_model3d_aam_csr_attention(ns_gpu* g, const float* q, const float* k, const float* v, int64_t N, int64_t D, const int32_t* rowptr,
                                 const int32_t* cols, float scale, float* out);
/* Farthest point sampling of x [N, D] (pytorch3d's: start at row 0; each next the row farthest, by squared distance
 * summed feature by feature in float32, from those taken; the first such row on ties). idx [K] int32 (K <= N);
 * dist: N floats of scratch. One work-group. */
int ns_model3d_aam_fps(ns_gpu* g, const float* x, int64_t N, int64_t D, int64_t K, float* dist, int32_t* idx);
/* out [R, C] = x[idx[r]] for idx int32 [R] */
int ns_model3d_aam_gather(ns_gpu* g, const float* x, const int32_t* idx, int64_t R, int64_t C, float* out);
/* The flow's model input, both halves of the CFG batch: out [2K, C] rows k and K + k = (f0[k] [F] | y[k][F:]) */
int ns_model3d_aam_flow_in(ns_gpu* g, const float* f0, const float* y, int64_t K, int64_t C, int64_t F, float* out);
/* The guided flow: out = u + g (c - u) for c = flow[:n], u = flow[n:2n] */
int ns_model3d_aam_cfg(ns_gpu* g, const float* flow, int64_t n, float scale, float* out);
/* out = y + a f, n values (a product then a sum: the ODE solver's steps) */
int ns_model3d_aam_axpy(ns_gpu* g, const float* y, const float* f, float a, int64_t n, float* out);

#ifdef __cplusplus
}
#endif
#endif
