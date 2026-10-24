/* mi.h: the Ming-Image engine's own kernels (kernels/image/mingimage), beside the shared diffusion kernels (nsd.h).
 * Bound by its crate by name (image/mingimage/src/ffi.rs). 16-bit buffers are IEEE half (__half); every call is queued
 * on the GPU's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_MI_H
#define NS_IMAGE_MI_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, C] half = layernorm(x [M, C] float32, no affine, eps) * (1 + scale[c]); scale float32 [C] or NULL: the DiT's
 * final layer. One block a row (C at most 8,192). */
int ns_image_mi_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out);
/* out [M, N] half = softmax(x * scale) per row of x [M, N] float32: the VAE's attention */
int ns_image_mi_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out);
/* x half, in place: silu */
int ns_image_mi_silu(ns_gpu* g, void* x, int64_t n);
/* x float32 += a * y float32 (n values): the sampler's step */
int ns_image_mi_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);
/* conversions: float32 <-> half (n values) */
int ns_image_mi_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
int ns_image_mi_to_float(ns_gpu* g, const void* x, float* out, int64_t n);
/* out [2H, 2W, C] half = nearest 2x of x [H, W, C] half (channels last) */
int ns_image_mi_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out);
/* rgba [H, W, 4] bytes = round((clamp(x, -1, 1) / 2 + 1 / 2) * 255) of x [H, W, 4] half */
int ns_image_mi_to_rgba8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgba);
/* The mixture of experts' rows: out [n, C] half = x[idx[i]] for i < n (x [M, C] half, idx int32 [n]) */
int ns_image_mi_gather_rows(ns_gpu* g, const void* x, int64_t C, const int32_t* idx, int64_t n, void* out);
/* ... and their sum back: x[t] float32 += sum over j < K of w[t, j] * y[slot[t, j]] (y [S, C] float32; slot int32 and
 * w float32 [M, K]; a slot below 0 is skipped). One block a token, in a fixed order: the same sum every run. */
int ns_image_mi_combine(ns_gpu* g, float* x, int64_t M, int64_t C, const void* y, const int32_t* slot, const float* w, int64_t K);
/* The gated activation, scaled down: out[r, i] = silu(x[r, i]) * x[r, C + i] * s (x [M, 2C], out [M, C] half; float32
 * inside). The product reaches past half's range in this model's MLPs (65,504; a layer of the DiT's makes infinities):
 * scaled by 1/256 here, the next linear's float32 output is multiplied back (nsd_scale) */
int ns_image_mi_swiglu_scaled(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out);
/* out [M, C] half, rows `ld` halves apart = x [M, C] float32 * s: a projection's output put in its place of a wider
 * buffer (the DiT's q | k | v), scaled into half's range */
int ns_image_mi_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out, int64_t ld);
/* x [M, C] float32 += other [M, C] half * gate[c] (gate float32 [C]; NULL: 1) - the DiT's tanh-gated residuals */
int ns_image_mi_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* other, const float* gate);
/* out [M, C] half = x [M, C] float32 * (1 + scale[c]) after an rms norm with weight w (float32 [C]), eps: the DiT's
 * modulated input norm, and with scale NULL a plain rms norm to half */
int ns_image_mi_rms_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, void* out);
/* out [rows, Hq * D] half: each query head h of k / v head h / (Hq / Hkv) copied out of x (rows of `stride` halves,
 * the heads starting at element `off`): a grouped key / value made full for the bidirectional attention kernel */
int ns_image_mi_repeat_heads(ns_gpu* g, const void* x, int64_t rows, int64_t stride, int64_t off, int64_t Hkv, int64_t Hq, int64_t D, void* out);

#ifdef __cplusplus
}
#endif
#endif
