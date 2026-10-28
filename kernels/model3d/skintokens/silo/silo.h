/* silo.h: SkinTokens' silo (dist/silo/libnextsycl-skintokens.so): its kernels tuned for its shapes, each with a
 * _supported check; the engine (model3d/skintokens/src/ffi.rs) calls them for what they cover and the engine's own
 * kernels (skintokens.h) otherwise. */
#ifndef NS_MODEL3D_SKINTOKENS_SILO_H
#define NS_MODEL3D_SKINTOKENS_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether ns_model3d_skintokens_silo_attention takes these shapes (1) or the engine's own kernel must (0): heads of 16, 32,
 * 48 or 64 (D = Dv), H a multiple of Hkv, up to 65,536 keys */
int ns_model3d_skintokens_silo_attention_supported(int64_t D, int64_t Dv, int64_t H, int64_t Hkv, int64_t Skv);
/* ns_model3d_skintokens_attention (skintokens.h), the same arguments and result but the scratch (not needed): one pass,
 * the scores in registers (flash attention's online softmax); q, k, v and out float4-aligned, their rows and heads multiples of 4 */
int ns_model3d_skintokens_silo_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                         const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv,
                                         int64_t D, int64_t Dv, float scale, int causal, float* out, int64_t out_row);

/* Whether ns_model3d_skintokens_silo_gemv takes these shapes (1) or the engine's linear layer (float32, oneDNN) must (0):
 * M at most 16 activation rows, K a multiple of 8 */
int ns_model3d_skintokens_silo_gemv_supported(int64_t M, int64_t N, int64_t K);
/* out [M, N] = x [M, K] . W^T, W [N, K] bfloat16 (as uint16 bit patterns; TokenRig's weights are stored so, which makes this the
 * same product as the float32 one but for the order of the sums): float32 activations and accumulation; x and w 16-byte
 * aligned, x rows contiguous */
int ns_model3d_skintokens_silo_gemv(ns_gpu* g, const float* x, const uint16_t* w, int64_t M, int64_t N, int64_t K, float* out);

/* Whether ns_model3d_skintokens_silo_attn_slots takes these shapes (1) or the engine's own kernel must (0): heads of 128, H = Hkv or
 * 2 Hkv, at most 4,096 positions */
int ns_model3d_skintokens_silo_attn_slots_supported(int64_t D, int64_t len, int64_t H, int64_t Hkv);
/* ns_model3d_skintokens_attn_slots (skintokens.h), the same arguments and result: a work-group a (beam, key / value head) with
 * the query heads that share it; a sub-group of 16 reads a key row whole, the weighted values a thread a dimension */
int ns_model3d_skintokens_silo_attn_slots(ns_gpu* g, const float* q, int64_t q_row, const float* kpool, const float* vpool, const int32_t* slots, int64_t stride,
                                          int64_t len, int64_t B, int64_t H, int64_t Hkv, int64_t D, float scale, float* out);

#ifdef __cplusplus
}
#endif
#endif
