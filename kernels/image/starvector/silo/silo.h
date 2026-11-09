/* silo.h: the StarVector engine's silo (dist/silo/libnextsycl-starvector.so): StarCoder's decode attention as the engine
 * needs it - one new token a beam over a long key / value cache. The engine (image/starvector/src/ffi.rs) calls it when
 * the library is there and this GPU runs it at the model's sizes, and the shared attention (nsd_attention_causal_rows)
 * otherwise. Every call is queued on the GPU's queue and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_STARVECTOR_SILO_H
#define NS_IMAGE_STARVECTOR_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether this silo runs the pass below at these sizes (1) or the engine must use the shared kernel (0): D (the head)
 * 128, H (query heads) 1 to 64, one key / value head, N (keys) 1 to 8192, 1 to 8 beams */
int ns_image_starvector_silo_attn_supported(ns_gpu* g, int64_t nb, int64_t N, int64_t H, int64_t D);

/* Decode attention, multi-query: beam b's query row q + b * H * D (half [H, D]) against its own cache k[b], v[b]
 * (half [N, D]: keys 0 .. N - 1, the new token's last), softmax scale 1 / sqrt(D); out + b * H * D (half [H, D]).
 * The shared kernel walks the N keys one at a time in a single work-group a head (a group-wide reduction a key: 1 us
 * a key, 0.15 s a token at 3,000 keys, 48 launches); this one splits the keys in chunks of 128 across work-groups
 * (flash-decoding) and combines the partial softmaxes - the key / value rows are read once a chunk, by all 16 heads. */
int ns_image_starvector_silo_decode_attn(ns_gpu* g, const void* q, const void* const* k, const void* const* v, int64_t nb, int64_t N, int64_t H, int64_t D, void* out);

/* Whether the layer norm below runs these sizes: C (the row) a multiple of 256, up to 4096 */
int ns_image_starvector_silo_ln_supported(ns_gpu* g, int64_t M, int64_t C);

/* LayerNorm of x float32 [M, C]: out half [M, C] = (x - mean) / sqrt(var + eps) * weight + bias (weight, bias float32 [C]).
 * A work-group a row, the sums as group reductions. The shared nsd_layer_norm sums a row in one work-item (two serial passes
 * over the C values, a launch of M threads): 270 us a call at C = 2048, 48 calls a token - 13 ms of the 18 a token. */
int ns_image_starvector_silo_layer_norm(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* weight, const float* bias, float eps, void* out);

#ifdef __cplusplus
}
#endif
#endif
