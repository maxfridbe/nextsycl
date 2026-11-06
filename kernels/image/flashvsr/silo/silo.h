/* silo.h: the FlashVSR engine's silo (dist/silo/libnextsycl-flashvsr.so): its attention on the matrix engines - the
 * block-sparse self-attention and the dense cross-attention to the prompt - as Intel's ARK kernels on sycl-tla (2-D
 * block loads, sub-group matrix products). Built apart from the kind's library with sycl-tla's flags (kernels/build.sh:
 * the silo's tla/ sources), so it does not use the shared part's ns_gpu: calls take the GPU's SYCL queue (Gpu::queue)
 * and say why they failed in ns_image_flashvsr_silo_error(). Every call is queued on that queue; half throughout, heads
 * of 128, rows of H * 128 halves. The engine (image/flashvsr/src/ffi.rs) uses these when the library is there and its
 * own kernels (fv.h) otherwise. */
#ifndef NS_IMAGE_FLASHVSR_SILO_H
#define NS_IMAGE_FLASHVSR_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* ns_image_flashvsr_silo_error(void);

/* Whether the silo's kernels run on this queue's GPU (1) or the engine must use its own (0) */
int ns_image_flashvsr_silo_supported(void* queue);

/* Block-sparse attention: query rows q [Sq, H * 128] over k, v [Skv, H * 128] (their rows `kv_rs` halves apart: the
 * ring's keys and values side by side), softmax(q k^T / sqrt(128)) v over the keys
 * of the blocks each query block's LUT row names: lut int32 [H, Sq / qrow, Skv / 128] delta-encoded (the first block,
 * then each one's step from the one before, ascending), counts int32 [H, Sq / qrow] (qrow query rows a LUT row: 64 or
 * 128, by q_tile). q_tile: the kernel's query tile - 64 (qrow 64), 128 (qrow 128, keys 64 a step; 129: 32 a step: one
 * LUT row a work-group) or 256 (qrow 128: two LUT rows a work-group, the union of their lists walked). A row with no
 * block is not defined (the caller zeroes it). out [Sq, H * 128]. */
int ns_image_flashvsr_silo_sparse(void* queue, const void* q, const void* k, const void* v, void* out, int64_t Sq, int64_t Skv, int64_t H,
                                  int64_t kv_rs, const int32_t* lut, const int32_t* counts, int64_t qrow, int64_t q_tile);

/* Dense attention (no mask): q [Sq, H * 128] over k, v [Skv, H * 128], rows `q_rs`, `kv_rs` and `o_rs` halves apart */
int ns_image_flashvsr_silo_attention(void* queue, const void* q, const void* k, const void* v, void* out, int64_t Sq, int64_t Skv, int64_t H,
                                     int64_t q_rs, int64_t kv_rs, int64_t o_rs);

#ifdef __cplusplus
}
#endif
#endif
