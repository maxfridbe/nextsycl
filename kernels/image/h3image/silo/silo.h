/* silo.h: the MiniMax-H3-Image engine's silo (dist/silo/libnextsycl-h3image.so; P-H3-Image's silo copied: the same H3 denoiser attention): the denoiser's long-sequence attention with
 * its own preparation passes, and the GGUF block matrices' expansion to bfloat16 (Q3_K read as stored) - the engine (image/h3image/src/ffi.rs) uses it when the library is there and the shared
 * diffusion kernels' (nsd_attention) otherwise. Every call is queued on the GPU's queue and returns 0, or -1 with the
 * reason in ns_last_error(). */
#ifndef NS_IMAGE_H3I_SILO_H
#define NS_IMAGE_H3I_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether `sage` runs these sizes (1) or the engine must use the shared attention (0): D 128, S from 64 to what fits int32
 * indexing, and libnextsycl-flash.so's SageAttention entry found */
int ns_image_h3i_silo_sage_supported(ns_gpu* g, int64_t S, int64_t H, int64_t D);

/* SageAttention v1 over the packed qkv rows, as the shared nsd attention's Sage form computes it (q and k - k's mean over
 * the sequence to int8, one scale per head per 64 rows; v in half; ARK's int8 kernel; the result back in the activations'
 * type) with the passes around the kernel made to stream: k's column sums in parallel slices, q / k / v converted by one
 * launch reading 16-byte vectors, the result written back the same way. qkv bfloat16 [S, stride] with q at column 0,
 * k at H * D, v at 2 H * D; out bfloat16 [S, H * D]. */
int ns_image_h3i_silo_sage(ns_gpu* g, const void* qkv, int64_t S, int64_t H, int64_t D, int64_t stride, void* out);

/* Whether `expand` reads GGUF type `qtype` (1): Q8_0 (8), Q3_K (11), Q4_K (12), Q5_K (13) */
int ns_image_h3i_silo_expand_supported(int qtype);

/* n values of GGUF blocks of type `qtype` (as stored in the file; device memory, 16-byte aligned) into out[0 .. n) as
 * bfloat16 (32-byte aligned): the same bits as the shared nsd_dequant (Q3_K: as its Q6_K form), 16 values a
 * work-item from 16-byte loads (expand.cpp) */
int ns_image_h3i_silo_expand(ns_gpu* g, const void* src, int qtype, int64_t n, void* out);

/* A Q8_0 matrix [N, K] (its GGUF blocks, in device memory) rewritten in place as the int8 values [N, K] followed by
 * the half scales [K / 32, N] (the same 34 bytes a block), through `tmp` (at least N K 34 / 32 bytes); waits (q8.cpp) */
int ns_image_h3i_silo_q8_repack(ns_gpu* g, void* blocks, int64_t N, int64_t K, void* tmp);

/* out [M, N] = x [M, K] . W^T, x and out bfloat16, W a repacked Q8_0 matrix (q8_repack): oneDNN's weight
 * decompression (int8, half scales every 32 along K, fpmath bf16) - no expansion pass */
int ns_image_h3i_silo_q8_linear(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, void* out);

#ifdef __cplusplus
}
#endif
#endif
