/* silo.h: AnimateAnyMesh's silo (dist/silo/libnextsycl-animateanymesh.so): its kernels tuned for its shapes, each with a
 * _supported check; the engine (model3d/animateanymesh/src/ffi.rs) calls them for what they cover and the engine's own
 * kernels (aam.h) otherwise. */
#ifndef NS_MODEL3D_AAM_SILO_H
#define NS_MODEL3D_AAM_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether ns_model3d_aam_silo_attention takes these shapes (1) or the engine's own kernel must (0) */
int ns_model3d_aam_silo_attention_supported(int64_t D, int64_t Dv, int causal);
/* ns_model3d_aam_attention (aam.h), the same arguments and result: heads of 16, 32, 48 or 64 (D = Dv, float4-aligned
 * rows) in one pass, the scores in registers (flash attention's online softmax); other shapes as two float32 oneDNN
 * products (the scores, then the weighted values, both batched over sequences and heads through strides) with a
 * softmax between, the scores of a block of query rows through `scratch` (the rows a pass takes are what fits). */
int ns_model3d_aam_silo_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                  const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                                  int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row);
/* Whether ns_model3d_aam_silo_fps takes these (1) or the engine's own kernel must (0) */
int ns_model3d_aam_silo_fps_supported(int64_t N, int64_t D, int64_t K);
/* ns_model3d_aam_fps (aam.h), the same arguments and the same picks: a launch a pick over the whole GPU, the last
 * work-group reducing the groups' best. Waits for the GPU before it returns (its scratch freed). */
int ns_model3d_aam_silo_fps(ns_gpu* g, const float* x, int64_t N, int64_t D, int64_t K, float* dist, int32_t* idx);

#ifdef __cplusplus
}
#endif
#endif
