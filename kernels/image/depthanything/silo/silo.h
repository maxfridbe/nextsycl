/* silo.h: the Depth Anything V2 engine's silo (dist/silo/libnextsycl-depthanything.so): the ViT's self attention with the
 * keys and values read once a work-group from shared local memory - the engine (image/depthanything/src/ffi.rs) uses it
 * when the library is there and ns_image_da_attention (da.h) otherwise. Queued on the GPU's queue; 0, or -1 with the
 * reason in ns_last_error(). */
#ifndef NS_IMAGE_DA_SILO_H
#define NS_IMAGE_DA_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* 1 when this GPU runs the attention below at head size D (64), 0 when the engine must use its own kernel */
int ns_image_da_silo_supported(ns_gpu* g, int64_t D);

/* Same contract and results as ns_image_da_attention: qkv float32 [N, 3 H D] (q | k | v), out float32 [N, H D];
 * softmax(q k^T / sqrt(D)) v per head, D = 64. A work-group takes 128 queries of a head and walks the keys in tiles of 32
 * held in shared local memory (every lane reads the same key: a broadcast), online softmax in float32. */
int ns_image_da_silo_attention(ns_gpu* g, const float* qkv, int64_t N, int64_t H, int64_t D, float* out);

/* The attention as products (gemm_attn.cpp; the engine's default with this silo): the heads laid out for the shared
 * float32 product - q, k [H, N, D] and v transposed [H, D, N] from qkv [N, 3 H D] */
int ns_image_da_silo_split(ns_gpu* g, const float* qkv, int64_t N, int64_t H, int64_t D, float* q, float* k, float* vt);
/* s [rows, cols] in place: softmax(s * scale) a row */
int ns_image_da_silo_softmax(ns_gpu* g, float* s, int64_t rows, int64_t cols, float scale);
/* o [H, N, D] -> out [N, H D] */
int ns_image_da_silo_merge(ns_gpu* g, const float* o, int64_t N, int64_t H, int64_t D, float* out);

#ifdef __cplusplus
}
#endif
#endif
