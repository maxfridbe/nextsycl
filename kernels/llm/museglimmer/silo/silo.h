/* silo.h: the museglimmer engine's silo - kernels tuned for Muse Glimmer 30B Q8_0 (every matrix Q8_0), in their own
 * library (dist/silo/libnextsycl-museglimmer.so, opened by the engine at load; without it the engine uses the shared
 * ones). A Q8_0 matrix is repacked once at load into its quants and its scales apart (lossless): 34-byte blocks leave
 * the quants 2-byte aligned, the repacked rows are 16-byte loads. */
#ifndef NS_MG_SILO_H
#define NS_MG_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* A repacked Q8_0 matrix [rows, n_in] (n_in a multiple of 32): its quants [rows][n_in] int8, then (from
 * ns_mg_q8r_scales_at) its scales [rows][n_in / 32] half */
size_t ns_mg_q8r_bytes(int64_t rows, int64_t n_in);
size_t ns_mg_q8r_scales_at(int64_t rows, int64_t n_in);
int ns_mg_q8r_pack(ns_gpu* g, const void* q8_0, int64_t rows, int64_t n_in, void* dst);
/* Float rows [rows, n_in] quantized into that layout (Q8_0's rounding): the DFlash drafter's K-quant matrices, so its
 * 16-row passes take the products below */
int ns_mg_q8r_from_f32(ns_gpu* g, const float* x, int64_t rows, int64_t n_in, void* dst);
/* Activations of a decode pass (1..16 columns) quantized for the XMX products: each 32 by amax / 127 (rounded as
 * ggml's Q8_1 does), block b's quants at b x 512 bytes - word w (quants 4w .. 4w + 3) of column c at w x 64 + c x 4: the
 * packed int8 B operand - then the scales [16][n_in / 32] float - of x = a (op 0), a * sigmoid(b) (op 1: the attention's
 * gate), silu(a) * b (op 2: SwiGLU); a and b [ncols, n_in] */
size_t ns_mg_xq_bytes(int64_t n_in, int64_t ncols);
int ns_mg_xq(ns_gpu* g, const float* a, const float* b, int op, void* xq, int64_t n_in, int64_t ncols);
/* RMS norm straight to that layout (decode rows, at most 16): x [rows, C] / rms(x) * w, its quants into xq (y nullable: the
 * normed rows too) */
int ns_mg_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t rows, int64_t C, float eps);
/* A post norm into the residual, then the next norm straight to the split Q8 layout (decode rows): x [rows, C] +=
 * out / rms(out) * wp (eps ep), then xq = (x / rms(x) * wn) quantized (eps en) - one launch for the two */
int ns_mg_post_rms_xq(ns_gpu* g, const float* out, const float* wp, float ep, float* x, const float* wn, float en, void* xq, int64_t rows, int64_t C);
/* Decode products on XMX: y [ncols, n_out] = W [n_out, n_in] . x (ns_mg_xq), ncols 1..16 (a verify pass), n_out a
 * multiple of 8; a column's sums the same however many columns; W repacked: its quants wq [n_out][n_in]
 * and scales ws [n_out][n_in / 32] (a repacked matrix's rows r0.. are wq + r0 n_in, ws + r0 n_in / 32) */
int ns_mg_mmvq(ns_gpu* g, const void* wq, const void* ws, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
/* Prompt attention (attn.cpp; the engine's own is ns_mg_attn): flash attention on XMX. out [T, Hq x 128] = softmax(q
 * k^T / sqrt(128)) v for rows q [T, Hq x 128] at positions p0.., causal, within the window (win <= 0: all), against
 * the ring caches [Hkv][cap][128] (slot pos % cap). Covers more than 16 rows, heads of 128, 16 query heads a key head */
int ns_mg_attn_prompt_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t D);
int ns_mg_attn_prompt(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                      int64_t win, float* out);
/* Decode attention (attn.cpp: the same flash kernel in its split form): 1..16 rows (a verify pass's) at positions
 * p0.., causal, within the window, the keys cut at absolute positions over ~512 work-groups, merged through `scratch`
 * (ns_mg_attn_decode_scratch floats); a row's result the same whatever rows share its pass */
int ns_mg_attn_decode_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t D);
int64_t ns_mg_attn_decode_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0, int64_t win);
int ns_mg_attn_decode(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                      int64_t win, float* out, float* scratch);
/* The DFlash drafter's block attention (the same kernel, not causal: every row sees to the block's last; 4 query heads a
 * key head), split over the keys and merged through `scratch` (ns_mg_attn_block_scratch floats) */
int64_t ns_mg_attn_block_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0, int64_t win);
int ns_mg_attn_block(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                     int64_t win, float* out, float* scratch);
/* Repacked Q8_0 rows expanded to half (a prompt pass's GEMMs): dst [rows, n_in] */
int ns_mg_q8r_f16(ns_gpu* g, const void* wq, const void* ws, int64_t rows, int64_t n_in, uint16_t* dst);

#ifdef __cplusplus
}
#endif
#endif
