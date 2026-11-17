/* silo.h: MiniMax H3's video silo (dist/silo/libnextsycl-h3.so): kernels the H3 engine (video/h3, crates/h3) uses for
 * MPEG-Diffusion's two-track steps (crates/h3/src/mpeg.rs). The engine opens it at load when a model card turns the
 * two-track mode on (H3_MPEG) and falls back to one device copy a row without it (or with NS_SILO=0). Every call is
 * queued on the GPU's queue and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_VIDEO_H3_SILO_H
#define NS_VIDEO_H3_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether `gather` takes these sizes (1): rows of `width` 16-bit values, width and both strides multiples of 8 (one
 * 16-byte load a work-item), all of them positive */
int ns_video_h3_silo_gather_supported(int64_t width, int64_t src_stride, int64_t dst_stride);

/* Rows of 16-bit values (bfloat16 or half: copied as bits) picked by index:
 *   dst[r * dst_stride + i] = src[idx[r] * src_stride + i]      r < rows, i < width
 * idx int32 [rows] in device memory; a row whose index is outside [0, src_rows) is left as it was (never read past
 * src). src and dst 16-byte aligned. How the two-track attention assembles the full-resolution queries' keys and
 * values from the qkv buffer (rows of 3 H D, k at H D, v at 2 H D) without one copy a row. */
int ns_video_h3_silo_gather(ns_gpu* g, void* dst, int64_t dst_stride, const void* src, int64_t src_stride, const int32_t* idx, int64_t rows,
                            int64_t width, int64_t src_rows);

#ifdef __cplusplus
}
#endif
#endif
