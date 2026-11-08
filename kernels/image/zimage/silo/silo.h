/* silo.h: the Z-Image engine's silo (dist/silo/libnextsycl-zimage.so): the DiT's per-block passes fused and read as
 * vectors - the engine (image/zimage/src/ffi.rs) uses them when the library is there and its own kernels (zi.h, the
 * shared nsd ones) otherwise. Every call is queued on the GPU's queue and returns 0, or -1 with the reason in
 * ns_last_error(). */
#ifndef NS_IMAGE_ZIMAGE_SILO_H
#define NS_IMAGE_ZIMAGE_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether this GPU runs the passes below at these sizes (1) or the engine must use its own kernels (0): sub-groups of 16,
 * C (the row) a multiple of 4 up to 4096, D (the head) 128 */
int ns_image_zimage_silo_supported(ns_gpu* g, int64_t C, int64_t D);

/* The DiT's residual step and the next norm in one pass over a row (x float32 [M, C], C a multiple of 4, at most 4096):
 * with o (float32 [M, C], a product's output): x += rmsnorm(o; wa, eps_a) * ga (ga float32 [C], NULL: 1); then with out:
 * out half = rmsnorm(x; wb, eps_b) * (1 + sb) (sb float32 [C], NULL: 1). zi.h's rms_mod, gate_add, rms_mod took three
 * passes and wrote a half row between them. */
int ns_image_zimage_silo_norm_gate_norm(ns_gpu* g, const float* o, float* x, int64_t M, int64_t C, const float* wa, float eps_a, const float* ga,
                                        const float* wb, float eps_b, const float* sb, void* out);

/* The q and k heads of the packed qkv rows (half, row stride `stride`, q at column 0, k at column `kcol`): each head's
 * RMS norm (weight wq / wk float32 [D], eps) and the rotary phases cs float32 [M, D] (cos, sin pairs; pair j of a token at
 * cs[token * D + 2 j]) applied to the (j, D / 2 + j) pairs - in place, both in one launch. D = 128. (MOSS-Image's
 * silo's, copied.) */
int ns_image_zimage_silo_qk_rope(ns_gpu* g, void* qkv, int64_t M, int64_t H, int64_t D, int64_t stride, int64_t kcol, const float* wq, const float* wk,
                                 float eps, const float* cs);

/* The gated activation, scaled: out[r, c] = silu(x[r, c]) * x[r, C + c] * s, x [M, 2C] half, out [M, C] half, 16-byte
 * vectors (C a multiple of 8) */
int ns_image_zimage_silo_swiglu(ns_gpu* g, const void* x, int64_t M, int64_t C, float s, void* out);

#ifdef __cplusplus
}
#endif
#endif
