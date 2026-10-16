/* t2.h: the TRELLIS.2 engine's kernels (model3d/trellis2, Pixal3D too) - its own C ABI, bound by its crate by name
 * (model3d/trellis2/src/ffi.rs). Buffers are device memory of g; dt 0 = float32, 1 = half. Every call is queued on g's
 * queue (the caller syncs); 0, or -1 and ns_last_error(). */
#ifndef NS_MODEL3D_TRELLIS2_H
#define NS_MODEL3D_TRELLIS2_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, N] float32 = x [M, K] . w^T + bias, x and w [N, K] half, bias float32 [N] or NULL: accumulated and written
 * in float32 (oneDNN) - the products that feed a float32 residual stream */
int ns_model3d_trellis2_linear_f32(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, float* out);
/* Layer norm of each row of x [M, C] (x_dt), affine (weight, bias float32 [C]) when given, then * (1 + scale) + shift
 * (float32 [C]) when given; out [M, C] in out_dt (it may be x when the types match: a row is read whole first) */
int ns_model3d_trellis2_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                               const float* scale, const float* shift, void* out, int out_dt);
/* TRELLIS's MultiHeadRMSNorm then its rotary embedding, in place on half rows: each of H heads of D features of row m
 * (rows `stride` values apart) -> x / max(|x|, 1e-12) * sqrt(D) * gamma[h, :]; then with cs (float32 [M, D/2, 2]:
 * cos, sin) feature pairs (2i, 2i + 1) rotated as complex numbers by angle i (cs NULL: the norm alone) */
int ns_model3d_trellis2_qk_norm_rope(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* gamma, const float* cs);
/* x [M, C] float32 += h [M, C] (h_dt) * gate (float32 [C]; NULL: a plain add) */
int ns_model3d_trellis2_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* h, int h_dt, const float* gate);
/* x [M, C] float32 += row [C] (float32) on every row */
int ns_model3d_trellis2_add_row(ns_gpu* g, float* x, int64_t M, int64_t C, const float* row);
/* dst = src over n values, converting between float32 and half */
int ns_model3d_trellis2_convert(ns_gpu* g, const void* src, int src_dt, int64_t n, void* dst, int dst_dt);
/* x = silu(x) in place over n values */
int ns_model3d_trellis2_silu(ns_gpu* g, void* x, int dt, int64_t n);
/* x += y over n values (x_dt, y_dt) */
int ns_model3d_trellis2_add(ns_gpu* g, void* x, int x_dt, const void* y, int y_dt, int64_t n);
/* The submanifold convolution's gather: rows [row0, row0 + rows) of out [rows, 27 C] (half) = for each tap t the
 * feature row nmap[r, t] of feats [*, C] (half), zero where nmap is -1 */
int ns_model3d_trellis2_gather27(ns_gpu* g, const void* feats, int64_t C, const int32_t* nmap, int64_t row0, int64_t rows, void* out);
/* out [n, C] = src[idx[j]] row by row (dt both) */
int ns_model3d_trellis2_gather_rows(ns_gpu* g, const void* src, int dt, int64_t C, const int32_t* idx, int64_t n, void* out);
/* The channel-to-space skip connection of a subdividing block: x [*, Cin] float32 repeated r times a channel and seen
 * as 8 children of Cout: out[j, c] = x[parent[j], (sub[j] Cout + c) / r], out float32 [n, Cout] */
int ns_model3d_trellis2_c2s_skip(ns_gpu* g, const float* x, int64_t Cin, const int32_t* parent, const int32_t* sub, int64_t n, int64_t Cout,
                                 int64_t r, float* out);
/* 3-D pixel shuffle, channels-last half: x [D, D, D, 8 C] -> out [2D, 2D, 2D, C], input channel c 8 + s1 4 + s2 2 + s3
 * to output voxel (2 i + s1, 2 j + s2, 2 k + s3) */
int ns_model3d_trellis2_pixel_shuffle3d(ns_gpu* g, const void* x, int64_t D, int64_t C, void* out);
/* Bilinear samples of a feature map (Pixal3D's projection): feat [Hf, Wf, C] float32, uv [n, 2] float32 in
 * grid_sample's normalized [-1, 1] (x, y), align_corners=False, padding_mode border (clipped to the map); out [n, C] float32, rows `stride` apart */
int ns_model3d_trellis2_grid_sample2d(ns_gpu* g, const float* feat, int64_t Hf, int64_t Wf, int64_t C, const float* uv, int64_t n, float* out,
                                      int64_t stride);

#ifdef __cplusplus
}
#endif
#endif
