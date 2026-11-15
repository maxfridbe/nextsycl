/* b3d.h: the Budget3D-Idol engine's kernels (model3d/budget3d) - its own C ABI, bound by its crate by name
 * (model3d/budget3d/src/ffi.rs). Buffers are device memory of g; dt 0 = float32, 1 = half. Every call is queued on g's
 * queue (the caller syncs); 0, or -1 and ns_last_error(). */
#ifndef NS_MODEL3D_BUDGET3D_H
#define NS_MODEL3D_BUDGET3D_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, N] float32 = x [M, K] . w^T + bias, x and w [N, K] half, bias float32 [N] or NULL: accumulated and written
 * in float32 (oneDNN) - the products that feed a float32 residual stream */
int ns_model3d_budget3d_linear_f32(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, float* out);

/* Layer norm of each row of x [M, C] (x_dt), affine (weight, bias float32 [C]) when given, then * (1 + scale) + shift
 * when given: row r reads scale / shift at (r / group) * stride (group <= 0: the first row for every row) - one
 * modulation row a sequence of `group` rows; out [M, C] in out_dt (it may be x when the types match) */
int ns_model3d_budget3d_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                               const float* scale, const float* shift, int64_t stride, int64_t group, void* out, int out_dt);

/* x [M, C] float32 += h [M, C] (h_dt) * gate, row r's gate at (r / group) * stride (group <= 0: the first row; gate
 * NULL: a plain add) */
int ns_model3d_budget3d_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* h, int h_dt, const float* gate, int64_t stride,
                                 int64_t group);

/* Idol's attention prologue, in place on half rows: each of H heads of D = 64 features of row m (rows `stride` values
 * apart) has its feature pairs (2i, 2i + 1) rotated as complex numbers by angle pi * delta[m, 3 h + a] * freqs[i] - a,
 * the axis of pair i, 0 for i < n0, 1 for i < n0 + n1, else 2 (Idol's learned RePo positions; delta NULL: no rotation)
 * - then RMS-normed: x / max(|x|, 1e-12) * sqrt(D) * gamma[h, :] */
int ns_model3d_budget3d_rope_norm(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* delta, const float* freqs,
                                  int64_t n0, int64_t n1, const float* gamma);

/* Full attention within each of B sequences of S <= 16 rows (Budget3D's refiners: a merged token's children), float32:
 * qkv [B S, 3 H D] (q, k, v of a row side by side, heads of D <= 64), out [B S, H D] */
int ns_model3d_budget3d_small_attn(ns_gpu* g, const float* qkv, int64_t B, int64_t S, int64_t H, int64_t D, float* out);

/* out [n, C] float32 = src[idx[j]] (+ add[aidx[j]] when add is given; aidx NULL: add's first row) */
int ns_model3d_budget3d_gather_add(ns_gpu* g, const float* src, int64_t C, const int32_t* idx, int64_t n, const float* add, const int32_t* aidx,
                                   float* out);

/* out [R, S, C] float32 = x [R, S, C] + pos [S, C] (the slot of each row) */
int ns_model3d_budget3d_slot_add(ns_gpu* g, const float* x, int64_t R, int64_t S, int64_t C, const float* pos, float* out);

/* Budget3D's ScaleMixer3D mixing across a parent's S children, per channel: out [R, t, c] = gelu(sum_s h[R, s, c]
 * w[c, t, s] + b[t, c]) (exact GELU), h and out float32 [R, S, C], w [C, S, S], b [S, C] */
int ns_model3d_budget3d_mix(ns_gpu* g, const float* h, int64_t R, int64_t S, int64_t C, const float* w, const float* b, float* out);

/* out [R, C] float32 = the mean of x [R, S, C] over S */
int ns_model3d_budget3d_group_mean(ns_gpu* g, const float* x, int64_t R, int64_t S, int64_t C, float* out);

/* The refiner's input: out [R S, C] float32 = e[r] + slot[s] + zt[r S + s] (+ carry[r] when given) */
int ns_model3d_budget3d_refine_init(ns_gpu* g, const float* e, int64_t R, int64_t S, int64_t C, const float* slot, const float* zt,
                                    const float* carry, float* out);

/* dst = src over n values, converting between float32 and half */
int ns_model3d_budget3d_convert(ns_gpu* g, const void* src, int src_dt, int64_t n, void* dst, int dst_dt);
/* x = silu(x) in place over n values */
int ns_model3d_budget3d_silu(ns_gpu* g, void* x, int dt, int64_t n);
/* x += y over n values (x_dt, y_dt) */
int ns_model3d_budget3d_add(ns_gpu* g, void* x, int x_dt, const void* y, int y_dt, int64_t n);

/* Group norm of a channels-last half map x [P, C] (G groups of C / G channels over all P pixels), affine (gamma, beta
 * float32 [C]), SiLU after when silu; out half [P, C]. scratch: float32 [G * 66] */
int ns_model3d_budget3d_group_norm(ns_gpu* g, const void* x, int64_t P, int64_t C, int64_t G, float eps, const float* gamma, const float* beta,
                                   int silu, void* out, float* scratch);
/* A channels-last half map x [H, W, C] padded by one zero row at the bottom and one zero column at the right (diffusers'
 * Downsample2D, pad (0, 1, 0, 1)): out [H + 1, W + 1, C] */
int ns_model3d_budget3d_pad_br(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out);
/* out [M, N] half = softmax(x [M, N] float32 * scale) row by row (the VAE's one-head attention, a block of rows at a
 * time) */
int ns_model3d_budget3d_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out);

#ifdef __cplusplus
}
#endif
#endif
