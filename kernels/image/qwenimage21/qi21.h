/* qi21.h: the Qwen-Image 2.1 engine's own kernels (image/qwenimage21), beside the shared diffusion kernels (nsd.h).
 * Bound by its crate by name (image/qwenimage21/src/ffi.rs). 16-bit buffers are IEEE half; every call is queued on the
 * GPU's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_QI21_H
#define NS_IMAGE_QI21_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, C] half = layernorm(x [M, C] float32, no affine, eps) * (1 + scale[c]); scale float32 [C] or NULL (plain
 * layernorm). One work-group a row: the row read once into registers, mean and variance by group reductions. */
int ns_image_qi21_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* scale, void* out);
/* x half, in place: gelu (tanh approximation) | silu */
int ns_image_qi21_gelu_tanh(ns_gpu* g, void* x, int64_t n);
int ns_image_qi21_silu(ns_gpu* g, void* x, int64_t n);
/* x float32 += a * y float32 (n values): the sampler's step */
int ns_image_qi21_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);
/* conversions: float32 <-> half (n values) */
int ns_image_qi21_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
int ns_image_qi21_to_float(ns_gpu* g, const void* x, float* out, int64_t n);
/* out [2H, 2W, C] half = nearest 2x of x [H, W, C] half (channels last) */
int ns_image_qi21_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out);
/* the VAE's duplicate-upsample shortcut added in: out [2H, 2W, Co] += x [H, W, Ci] with each output channel o of pixel
 * (y, x) reading input channel (o * F + t * 4 + (y % 2) * 2 + (x % 2)) / R, F = ft * 4 (ft 1 or 2 time duplicates;
 * t = ft - 1, the frame a first chunk keeps), R = Co * F / Ci */
int ns_image_qi21_dupup_add(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t Ci, int64_t Co, int ft, void* out);
/* rgba [H, W, 4] uint8 from x [H, W, 4] half in [-1, 1]: clamped, (x / 2 + 0.5) * 255 rounded */
int ns_image_qi21_to_rgba8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgba);

#ifdef __cplusplus
}
#endif
#endif
