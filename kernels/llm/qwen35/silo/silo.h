/* silo.h: the qwen35 engine's silo - kernels tuned for dense Qwen3.8-27B, in their own library
 * (dist/silo/libnextsycl-qwen35.so, opened by the engine at load; without it the engine uses the shared ones). Each op
 * says what it covers (`_supported`); the engine falls back to the shared kernel for everything else. */
#ifndef NS_Q35_SILO_H
#define NS_Q35_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Decode products (mmvq.cpp; the shared one is ns_mmvq): y [ncols, n_out] = W [n_out, n_in] (ggml type, stored
 * blocks) . x (ncols 1..8, the split Q8_1 layout below). Covers Q3_K, Q4_K, Q5_K, IQ4_NL, IQ3_S, IQ4_XS at a width of
 * whole blocks */
int ns_q35_mmvq_supported(int type, int64_t n_in);
/* The silo's split Q8_1 layout: a column's quants [ncols][n_in] int8 (16-byte aligned, contiguous), then the blocks'
 * scales [ncols][n_in / 32] half, then each 16 quants' sum [ncols][n_in / 16] int16 - from standard Q8_1 blocks
 * (column j's at j x n_in / 32) */
int ns_q35_q8_split(ns_gpu* g, const void* q8_1, void* xs, int64_t n_in, int64_t ncols);
size_t ns_q35_q8_split_bytes(int64_t n_in, int64_t ncols);
/* x [ncols, n_in] float32 -> its Q8_1 blocks (nullable; ns_quantize_q8_1's bits) and the split layout, one launch */
int ns_q35_quantize_q8(ns_gpu* g, const float* x, void* q8_1, void* xs, int64_t n_in, int64_t ncols);
int ns_q35_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, const void* x_split, float* y, int64_t n_in, int64_t n_out, int64_t ncols);

/* Prompt attention (attn.cpp; the engine's own is ns_q35_attn): flash attention on XMX. out [T, Hq x 256] = softmax(q k^T
 * / 16) v for rows q [T, Hq x 256] (qs floats apart) at positions p0.., causal, against the half caches [Hkv][cap][256].
 * Covers more than 8 rows, heads of 256, 6 query heads a key head */
int ns_q35_attn_prompt_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t D);
int ns_q35_attn_prompt(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                       int64_t cap, int64_t p0, float* out);
/* Decode attention (attn.cpp): 1..8 rows (each at its own position p0 + t), the same cache and output as above; a lane
 * a key for the scores, the keys split over ~512 work-groups, merged through `scratch` (ns_q35_attn_decode_scratch
 * floats; 0: none) */
int ns_q35_attn_decode_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t D);
int64_t ns_q35_attn_decode_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0);
int ns_q35_attn_decode(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                       int64_t cap, int64_t p0, float* out, float* scratch);
/* Weights to half for a prompt pass's GEMMs (dequant.cpp; the shared one is ns_dequant_f16): n values (a multiple of
 * 256) of a stored matrix into dst. Covers Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, IQ4_XS */
int ns_q35_dequant_f16_supported(int type);
int ns_q35_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst);
/* RMS norm with the Q8_1 blocks of its output (norm.cpp; for decode rows): y [rows, C] = x / rms(x) * w (y may be null),
 * q8_1 the blocks of y as ns_quantize_q8_1 makes them (column r at r x C / 32), xs (nullable) the same in the split
 * layout; C a multiple of 256 */
int ns_q35_rms_q8(ns_gpu* g, const float* x, const float* w, float* y, void* q8_1, void* xs, int64_t rows, int64_t C, float eps);

#ifdef __cplusplus
}
#endif
#endif
