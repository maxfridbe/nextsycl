/* hi3d.h: the Hi3D engine's kernels (model3d/hi3d) - its own C ABI, bound by its crate by name (model3d/hi3d/src/ffi.rs).
 * Buffers are device memory of g; dt 0 = float32, 1 = half. Every call is queued on g's queue (the caller syncs); 0, or
 * -1 and ns_last_error(). */
#ifndef NS_MODEL3D_HI3D_H
#define NS_MODEL3D_HI3D_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, N] float32 = x [M, K] . w^T + bias, x and w [N, K] half, bias float32 [N] or NULL: accumulated and written
 * in float32 (oneDNN) */
int ns_model3d_hi3d_linear_f32(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, float* out);
/* Layer norm of each row of x [M, C] (x_dt; C up to 8192), affine (weight, bias float32 [C]) when given, then
 * * (1 + scale) + shift (float32 [C]) when given; out [M, C] in out_dt (it may be x when the types match) */
int ns_model3d_hi3d_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                           const float* scale, const float* shift, void* out, int out_dt);
/* Hi3D's self-attention q / k, in place on half rows: each of H heads of D features of row m (rows `stride` values
 * apart) rotated first - feature pairs (2i, 2i + 1) as complex numbers by cs (float32 [M, D/2, 2]: cos, sin) - then
 * RMS-normed: x / max(|x|, 1e-12) * sqrt(D) * gamma[h, :] (float32 [H, D]) */
int ns_model3d_hi3d_rope_rms(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* cs, const float* gamma);
/* x [M, C] float32 += h [M, C] (h_dt) * gate (float32 [C]; NULL: a plain add) */
int ns_model3d_hi3d_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* h, int h_dt, const float* gate);
/* x [M, C] float32 += row [C] (float32) on every row */
int ns_model3d_hi3d_add_row(ns_gpu* g, float* x, int64_t M, int64_t C, const float* row);
/* dst = src over n values, converting between float32 and half */
int ns_model3d_hi3d_convert(ns_gpu* g, const void* src, int src_dt, int64_t n, void* dst, int dst_dt);
/* x = act(x) in place over n values: act 0 silu, 1 quick_gelu (x sigmoid(1.702 x)), 2 tanh */
int ns_model3d_hi3d_act(ns_gpu* g, void* x, int dt, int64_t n, int act);
/* x += y over n values (x_dt, y_dt) */
int ns_model3d_hi3d_add(ns_gpu* g, void* x, int x_dt, const void* y, int y_dt, int64_t n);
/* dst[r, off .. off + C) = src [n, C] row r (src_dt -> dst_dt), dst rows ldd values apart */
int ns_model3d_hi3d_put_cols(ns_gpu* g, const void* src, int src_dt, int64_t n, int64_t C, void* dst, int dst_dt, int64_t ldd, int64_t off);
/* The submanifold convolution's gather: rows [row0, row0 + rows) of out [rows, 27 C] (half) = for each tap t the
 * feature row nmap[r, t] of feats [*, C] (half), zero where nmap is -1 */
int ns_model3d_hi3d_gather27(ns_gpu* g, const void* feats, int64_t C, const int32_t* nmap, int64_t row0, int64_t rows, void* out);
/* out [n, C] = src[idx[j]] row by row (dt both) */
int ns_model3d_hi3d_gather_rows(ns_gpu* g, const void* src, int dt, int64_t C, const int32_t* idx, int64_t n, void* out);
/* The paired downsample's mean as the reference takes it (scatter_reduce mean into zeros, include_self: the zero counted): out
 * [P, C] float32 row p = the sum of x's rows (x_dt) idx[ptr[p] .. ptr[p + 1]) / (their count + 1) */
int ns_model3d_hi3d_seg_mean(ns_gpu* g, const void* x, int x_dt, int64_t C, const int32_t* ptr, const int32_t* idx, int64_t P, float* out);
/* PointGroupNorm's sums: x [N, C] float32 in G groups of C / G channels; part [blocks, G, 2] float32 gets each block of
 * `rows` rows' sum and sum of squares a group (blocks = ceil(N / rows)); the caller adds the blocks up */
int ns_model3d_hi3d_gn_sums(ns_gpu* g, const float* x, int64_t N, int64_t C, int64_t G, int64_t rows, float* part);
/* PointGroupNorm then Swish: out [N, C] (out_dt) = swish((x - mean[g]) * inv[g] * w[c] + b[c]) with g = c / (C / G)
 * (mean, inv float32 [G]; w, b float32 [C]) */
int ns_model3d_hi3d_gn_swish(ns_gpu* g, const float* x, int64_t N, int64_t C, int64_t G, const float* mean, const float* inv, const float* w,
                             const float* b, void* out, int out_dt);
/* Window attention (PTv3's serialized attention): qkv [W S, 3 H D] half (q | k | v), each window of S rows attending
 * within itself, scale `scale`; out [W S, H D] half. D up to 64, S up to 64 */
int ns_model3d_hi3d_win_attn(ns_gpu* g, const void* qkv, int64_t W, int64_t S, int64_t H, int64_t D, float scale, void* out);
/* The children's neighbour maps from their parents': child j (parent[j], sub-cell sub[j] = 4 sx + 2 sy + sz) reads at
 * tap (a, b, c) the child (sx + a - 1, ...) of the parent's neighbour at the matching tap: kids [P, 8] gives each parent's
 * child rows (-1: pruned); out [n, 27] int32 (-1 where absent) */
int ns_model3d_hi3d_child_nmap(ns_gpu* g, const int32_t* pnmap, const int32_t* kids, const int32_t* parent, const int32_t* sub, int64_t n, int32_t* out);
/* Heads copied between widths (half): row m's head h, features 0 .. min(sd, dd) of src (rows `ss` values apart,
 * heads sd wide) times `scale` into dst (rows `ds` apart, heads dd wide), the rest of each dst head zero - the 96-wide
 * heads padded to 128 for the flash attention kernel (q scaled by sqrt(128 / 96) to keep softmax(q k / sqrt(96))) and
 * its output taken back */
int ns_model3d_hi3d_copy_heads(ns_gpu* g, const void* src, int64_t ss, int64_t sd, int64_t M, int64_t H, void* dst, int64_t ds, int64_t dd, float scale);

#ifdef __cplusplus
}
#endif
#endif
