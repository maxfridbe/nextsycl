/* silo.h: the P-H3-Image engine's silo (dist/silo/libnextsycl-ph3image.so): the denoiser's long-sequence attention with
 * its own preparation passes - the engine (image/ph3image/src/ffi.rs) uses it when the library is there and the shared
 * diffusion kernels' (nsd_attention) otherwise. Every call is queued on the GPU's queue and returns 0, or -1 with the
 * reason in ns_last_error(). */
#ifndef NS_IMAGE_PH3_SILO_H
#define NS_IMAGE_PH3_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether `sage` runs these sizes (1) or the engine must use the shared attention (0): D 128, S from 64 to what fits int32
 * indexing, and libnextsycl-flash.so's SageAttention entry found */
int ns_image_ph3_silo_sage_supported(ns_gpu* g, int64_t S, int64_t H, int64_t D);

/* SageAttention v1 over the packed qkv rows, as the shared nsd attention's Sage form computes it (q and k - k's mean over
 * the sequence to int8, one scale per head per 64 rows; v in half; ARK's int8 kernel; the result back in the activations'
 * type) with the passes around the kernel made to stream: k's column sums in parallel slices, q / k / v converted by one
 * launch reading 16-byte vectors, the result written back the same way. qkv bfloat16 [S, stride] with q at column 0,
 * k at H * D, v at 2 H * D; out bfloat16 [S, H * D]. */
int ns_image_ph3_silo_sage(ns_gpu* g, const void* qkv, int64_t S, int64_t H, int64_t D, int64_t stride, void* out);

#ifdef __cplusplus
}
#endif
#endif
