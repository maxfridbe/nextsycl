/* silo.h: UniRig's silo (dist/silo/libnextsycl-unirig.so): its kernels tuned for its shapes, each with a
 * _supported check; the engine (model3d/unirig/src/ffi.rs) calls them for what they cover and the engine's own
 * kernels (unirig.h) otherwise. */
#ifndef NS_MODEL3D_UNIRIG_SILO_H
#define NS_MODEL3D_UNIRIG_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether ns_model3d_unirig_silo_attention takes these shapes (1) or the engine's own kernel must (0) */
int ns_model3d_unirig_silo_attention_supported(int64_t D, int64_t Dv, int causal);
/* ns_model3d_unirig_attention (unirig.h), the same arguments and result: heads of 16, 32, 48 or 64 (D = Dv,
 * float4-aligned rows) over up to 4,096 keys in one pass, the scores in registers (flash attention's online softmax); other shapes as two
 * float32 oneDNN products (the scores, then the weighted values, both batched over sequences and heads through
 * strides) with a softmax between, the scores of a block of query rows through `scratch` (what fits a pass). */
int ns_model3d_unirig_silo_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                  const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                                  int64_t Dv, float scale, int causal, float* scratch, int64_t scratch_n, float* out, int64_t out_row);
/* Whether ns_model3d_unirig_silo_attn_slots takes these shapes (1) or the engine's own kernel must (0) */
int ns_model3d_unirig_silo_attn_slots_supported(int64_t D, int64_t len);
/* ns_model3d_unirig_attn_slots (unirig.h), the same arguments and result, for heads of 64: a sub-group of 16 reads a
 * whole key or value row (a float4 a lane), the rows shared among the work-group's sub-groups. */
int ns_model3d_unirig_silo_attn_slots(ns_gpu* g, const float* q, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride,
                                      int64_t len, int64_t B, int64_t H, int64_t D, float scale, float* out);

#ifdef __cplusplus
}
#endif
#endif
