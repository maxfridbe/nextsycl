/* silo.h: the upscalers' silo (dist/silo/libnextsycl-upscale.so): convolutions tuned for Real-ESRGAN's and waifu2x's
 * graphs - the engine (image/upscale/src/ffi.rs) uses them when the library is there and the shared convolutions
 * (im2col + GEMM, a padding pass before and an activation pass after) otherwise. Channels-last IEEE half, every call
 * queued on the GPU's queue, 0 or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_UP_SILO_H
#define NS_IMAGE_UP_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether this GPU runs the calls below (1) or the engine must use the shared ones (0) */
int ns_image_up_silo_supported(ns_gpu* g);

/* out [Ho, Wo, Co] = act(conv(x [H, W, Ci] zero-padded by top / left / bottom / right, w [Co, Ci, k, k] half as it is
 * (reordered once into the kernel's layout: keep it alive and unchanged), stride) + bias float32 [Co] or NULL).
 * act: 0 none, 1 relu, 2 leaky relu (slope a), 3 PReLU (slopes float32 [n_slopes]: 1 or Co), 4 sigmoid - fused into
 * the convolution (oneDNN's post-ops), no pass of its own. x_ld / out_ld: halves from one pixel to the next (0: Ci /
 * Co) - x a channel slice of a wider buffer, out written into one (a dense block's concatenation in place). res:
 * NULL, or a residual [Ho, Wo, Co] (dense): out = ra x act(conv) + rb x res, the sum in the convolution too. */
int ns_image_up_silo_conv(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t Ci, int64_t x_ld, const void* w, int64_t Co, int64_t k,
                          int64_t stride, int64_t pt, int64_t pl, int64_t pb, int64_t pr, const float* bias, int act, float a, const float* slopes,
                          int64_t n_slopes, const void* res, float ra, float rb, void* out, int64_t out_ld);

/* dst [P, Cd] channels [off, off + Cs) = src [P, Cs] (channels and offset in eights) */
int ns_image_up_silo_copy_channels(ns_gpu* g, const void* src, int64_t P, int64_t Cs, void* dst, int64_t Cd, int64_t off);

/* out [H r, W r, C] = the nearest of x [H, W, C]: ncnn's Interp 1 by a whole factor (16-byte pieces when C is in eights) */
int ns_image_up_silo_upsample(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t r, void* out);

#ifdef __cplusplus
}
#endif
#endif
