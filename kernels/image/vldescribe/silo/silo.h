/* silo.h: the native describer's silo (dist/silo/libnextsycl-vldescribe.so): one written token's attention over the key /
 * value cache, split over chunks of keys (flash decoding). The engine (image/vldescribe/src/ffi.rs) uses it when the
 * library is there and the shared causal attention by rows (nsd_attention_causal_rows: a work-group a head walking every
 * key, 32 work-groups for a whole card) otherwise. Every call is queued on the GPU's queue and returns 0, or -1 with the
 * reason in ns_last_error(). */
#ifndef NS_IMAGE_VLD_SILO_H
#define NS_IMAGE_VLD_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether `attn1` runs these sizes (1) or the engine must use the shared attention (0): Hq a multiple of Hkv with at most
 * 16 query heads a key / value head, D a multiple of 16 up to 256 */
int ns_image_vld_silo_attn1_supported(int64_t Hq, int64_t Hkv, int64_t D);

/* The float32 scratch `attn1` needs for n cached rows */
int64_t ns_image_vld_silo_attn1_scratch(int64_t n, int64_t Hq, int64_t D);

/* One query row over n cached rows, float32: q [Hq][D]; k, v [n] rows of stride kvs, head g at g * D; out [Hq][D]. The
 * scores a sub-group of 16 a (key / value head, key) for every query head of the group at once; each (head, chunk of
 * 256 keys) its max, sum and weighted values; the chunks combined. */
int ns_image_vld_silo_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                            int64_t D, float* scratch, float* out);

#ifdef __cplusplus
}
#endif
#endif
