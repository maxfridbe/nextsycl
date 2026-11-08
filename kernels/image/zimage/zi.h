/* zi.h: the Z-Image engine's own kernels (kernels/image/zimage), beside the shared diffusion kernels (nsd.h). Bound by
 * its crate by name (image/zimage/src/ffi.rs). 16-bit buffers are IEEE half (__half); every call is queued on the GPU's
 * queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). The DiT's started as copies of
 * Ming-Image's (mi.h: the same transformer), the VAE's as copies of MOSS-Image's (mo.h: the same decoder blocks). */
#ifndef NS_IMAGE_ZI_H
#define NS_IMAGE_ZI_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, C] half = layernorm(x [M, C] float32, no affine, eps) * (1 + scale[c]); scale float32 [C] or NULL: the DiT's
 * final layer. One block a row (C at most 8,192). */
int ns_image_zi_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out);
/* out [M, C] half = rmsnorm(x [M, C] float32) * w (float32 [C], or NULL) * (1 + scale[c]) (scale NULL: 1): the DiT's
 * modulated input norm, the caption embedder's and the text encoder's norms */
int ns_image_zi_rms_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, void* out);
/* x [M, C] float32 += other [M, C] half * gate[c] (gate float32 [C]; NULL: 1) - the DiT's tanh-gated residuals */
int ns_image_zi_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* other, const float* gate);
/* out [M, C] half, rows `ld` halves apart = x [M, C] float32 * s: a product's output scaled into half's range */
int ns_image_zi_to_half_rows(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out, int64_t ld);
/* The gated activation, scaled down: out[r, i] = silu(x[r, i]) * x[r, C + i] * s, out [M, C] half; x [M, 2C] half
 * (swiglu_scaled) or float32 (swiglu_f). The product passes half's range in the DiT's MLPs: the next linear's float32
 * output is multiplied back (nsd_scale). */
int ns_image_zi_swiglu_scaled(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out);
int ns_image_zi_swiglu_f(ns_gpu* g, const float* x, int64_t M, int64_t C, float s, void* out);
/* x float32 += a * y float32 (n values): the sampler's step */
int ns_image_zi_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);
/* conversions: float32 <-> half (n values) */
int ns_image_zi_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
int ns_image_zi_to_float(ns_gpu* g, const void* x, float* out, int64_t n);

/* The VAE's. floats of scratch a group norm of `groups` needs */
int64_t ns_image_zi_group_norm_scratch(int64_t groups);
/* out [px, C] half = group norm of x [px, C] half (channels last; `groups` groups, eps, gamma / beta float32 [C]), SiLU
 * after when `silu`; scratch: ns_image_zi_group_norm_scratch floats */
int ns_image_zi_group_norm(ns_gpu* g, const void* x, int64_t px, int64_t C, int64_t groups, float eps, const float* gamma, const float* beta, int silu, void* out,
                           float* scratch);
/* out [M, N] half = softmax(x * scale) per row of x [M, N] float32: the VAE's attention */
int ns_image_zi_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out);
/* out [2H, 2W, C] half = nearest 2x of x [H, W, C] half (channels last) */
int ns_image_zi_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out);
/* rgb [H, W, 3] bytes = round((clamp(x, -1, 1) / 2 + 1 / 2) * 255) of x [H, W, 3] half */
int ns_image_zi_to_rgb8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgb);

#ifdef __cplusplus
}
#endif
#endif
