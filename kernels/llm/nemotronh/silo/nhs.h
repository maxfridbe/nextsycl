/* nhs.h: the nemotronh silo (dist/silo/libnextsycl-nemotronh.so, opened at load; without it the engine routes on the
 * host and runs an expert a launch): a MoE layer's decode work on the GPU in a few launches - the routing (sigmoid,
 * the selection bias, top-k, the weights), the activations in a split Q8 layout, and all of a layer's chosen experts
 * in one launch per projection. The products are the deepseek4 silo's scheme (copied, not shared), with the types
 * Nemotron's files hold and expert ids read on the GPU. */
#ifndef NS_NHS_H
#define NS_NHS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Per row of logits [T][E]: p = sigmoid(logit); the K experts of the largest p + bias (the lower index on a tie), in
 * that order, into ids [T][K]; their p (renormalized to sum 1 when norm) times scale into w [T][K]; cols [T][K] = the
 * row */
int ns_nhs_route(ns_gpu* g, const float* logits, const float* bias, int64_t T, int64_t E, int64_t K, int norm, float scale, int32_t* ids, float* w,
                 int32_t* cols);
/* Bytes of the split Q8 layout of ncols columns of n_in (a multiple of 32): the quants, the scales, 16-value sums */
size_t ns_nhs_q8_bytes(int64_t n_in, int64_t ncols);
/* x [ncols, n_in] float32 -> the split Q8 layout */
int ns_nhs_quantize_q8(ns_gpu* g, const float* x, void* xs, int64_t n_in, int64_t ncols);
/* Whether ns_nhs_multi takes ggml type `type` at n_in inputs */
int ns_nhs_multi_supported(int type, int64_t n_in);
/* Whether a type at this width is read packed by ns_nhs_multi (the 32-value step: Q2_0, IQ4_NL, Q8_0 at a width no
 * 256-block divides), the bytes of a packed row, and the packing: rows of the file's blocks -> a row's code bytes in
 * order (16-byte aligned), then its scales, padded to 16 bytes */
int ns_nhs_packed(int type, int64_t n_in);
int64_t ns_nhs_packed_row_bytes(int type, int64_t n_in);
int ns_nhs_pack(ns_gpu* g, int type, const void* src, void* dst, int64_t rows, int64_t n_in);
/* rows of a matrix of a type Strata's expansion lacks (IQ1_S; IQ2_XXS) expanded to fp16 [rows][n_in] */
int ns_nhs_wide_f16(ns_gpu* g, int type, const void* src, uint16_t* dst, int64_t rows, int64_t n_in);
/* rows of a packed matrix (n_in values each) expanded to fp16 [rows][n_in] (the prompt path's products) */
int ns_nhs_packed_f16(ns_gpu* g, int type, const void* src, uint16_t* dst, int64_t rows, int64_t n_in);
/* y [entries][n_out]: entry e = expert ids[e]'s matrix (base + ids[e] * stride bytes, n_out rows of n_in in its blocks)
 * against column cols[e] of xs (ncols columns in the split layout); a packed type's matrices in the packed layout */
int ns_nhs_multi(ns_gpu* g, int type, const void* base, int64_t stride, const int32_t* ids, const int32_t* cols, int64_t entries, const void* xs,
                 int64_t ncols, float* y, int64_t n_in, int64_t n_out);

/* attn.cpp: flash attention on XMX (the qwen35 silo's, copied) - head 128, 16 query heads a key head, no positions,
 * against the half cache [Hkv][cap][128]: prompt rows (more than 8) and decode rows (1..8: the keys split, merged after;
 * scratch floats as ns_nhs_attn_decode_scratch says) */
int ns_nhs_attn_prompt_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh);
int ns_nhs_attn_prompt(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh,
                       int64_t cap, int64_t p0, float* out);
int ns_nhs_attn_decode_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh);
int64_t ns_nhs_attn_decode_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh, int64_t p0);
int ns_nhs_attn_decode(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh,
                       int64_t cap, int64_t p0, float* out, float* scratch);

#ifdef __cplusplus
}
#endif
#endif
