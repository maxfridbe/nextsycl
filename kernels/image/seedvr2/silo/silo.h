/* silo.h: the SeedVR2 engine's silo (dist/silo/libnextsycl-seedvr2.so): kernels tuned for its VAE - the engine
 * (image/seedvr2/src/ffi.rs) uses them when the library is there and its own (svr.h) otherwise. Every call is queued
 * on the GPU's queue and returns 0, or -1 with the reason in ns_last_error(). bfloat16 channels-last frames. */
#ifndef NS_IMAGE_SEEDVR2_SILO_H
#define NS_IMAGE_SEEDVR2_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether this GPU runs the calls below (1) or the engine must use its own kernels (0) */
int ns_image_seedvr2_silo_supported(ns_gpu* g);

/* A causal 3-D convolution of stride 1 in time without copying its frames: out frame t (of nframes - kt + 1) =
 * bias + sum over taps j of the 2-D convolution of frame t + j by w[:, :, j]. frames[i] points at frame i of the
 * padded sequence, [H, W, Ci] bfloat16 (the kept frames of the last slice, or the first frame repeated, then the
 * slice's own): runs of frames that sit one after another go through one batched 2-D convolution a tap. Zeros ph /
 * pw around the picture; w [Co, Ci, kt, kh, kw] bfloat16 (each tap reordered once: keep it alive and unchanged), bias
 * float32 [Co] or NULL; out [nframes - kt + 1, H, W, Co]. */
int ns_image_seedvr2_silo_conv_taps(ns_gpu* g, const void* const* frames, int64_t nframes, int64_t H, int64_t W, int64_t Ci, const void* w,
                                    int64_t Co, int64_t kt, int64_t kh, int64_t kw, int64_t ph, int64_t pw, const float* bias, void* out);
/* The same, added to what out holds (a resnet's residual: its sum fused into the convolution, no pass of its own) */
int ns_image_seedvr2_silo_conv_taps_acc(ns_gpu* g, const void* const* frames, int64_t nframes, int64_t H, int64_t W, int64_t Ci, const void* w,
                                        int64_t Co, int64_t kt, int64_t kh, int64_t kw, int64_t ph, int64_t pw, const float* bias, void* out);

/* out [M, N] = x [M, K] . w^T + bias: bfloat16 in and out, w [N, K] read as it is, bias float32 [N] or NULL (the VAE's
 * pointwise convolutions: one product over all the pixels) */
int ns_image_seedvr2_silo_linear(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out);

/* Attention of B windows of S rows each, bfloat16 throughout (scores float32): q, k, v rows `stride` apart (the
 * packed qkv: k at q + H D, v at q + 2 H D), window b's rows from row b S; out [B S, H D] bfloat16. */
int ns_image_seedvr2_silo_attention(ns_gpu* g, const void* q, const void* k, const void* v, int64_t B, int64_t S, int64_t H, int64_t D,
                                    int64_t stride, void* out);

#ifdef __cplusplus
}
#endif
#endif
