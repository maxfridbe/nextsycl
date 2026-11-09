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

/* Decode products on XMX (xmm.cpp; ns_q35_mmvq routes to it for NS_Q35_XMX or more columns, default 5, on the types and
 * widths it covers: Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_0 at 2,048 or more rows and 4,096 or more columns of x). Reads the split layout's XMX tail (ns_q35_q8_split_bytes
 * includes it; this builds it). Its sums are the dp4a ones' integers, added in another order: a column is not bit for
 * bit a one-column pass's */
int ns_q35_xmx_min_cols(void);
int ns_q35_xmx_supported(int type, int64_t n_in, int64_t n_out);
int ns_q35_xmx_covers(int type, int64_t n_in, int64_t n_out, int64_t ncols);
int ns_q35_xmx_prep(ns_gpu* g, const void* xs, int64_t n_in, int64_t ncols, int halves);
int ns_q35_xmx_mmvq(ns_gpu* g, int type, const void* w, const void* xs, float* y, int64_t n_in, int64_t n_out, int64_t ncols);

/* Prompt attention (attn.cpp; the engine's own is ns_q35_attn): flash attention on XMX. out [T, Hq x 256] = softmax(q k^T
 * / 16) v for rows q [T, Hq x 256] (qs floats apart) at positions p0.., causal, against the half caches [Hkv][cap][256].
 * Covers more than 8 rows, heads of 256, 6 query heads a key head */
int ns_q35_attn_prompt_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t D);
int ns_q35_attn_prompt(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                       int64_t cap, int64_t p0, float* out);
/* Decode attention (attn.cpp): 1..8 rows (each at its own position p0 + t), the same cache and output as above; the
 * flash kernel on XMX with the keys split over ~512 work-groups, merged through `scratch` (ns_q35_attn_decode_scratch
 * floats; 0: none); a row's result is the same whatever rows share its pass */
int ns_q35_attn_decode_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t D);
int64_t ns_q35_attn_decode_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0);
int ns_q35_attn_decode(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                       int64_t cap, int64_t p0, float* out, float* scratch);
/* The two above against a cache of form kt: 0 half (512 bytes a head's row), 1 q8_0 (272: the 256 int8 values, then 8
 * half scales), 2 q4_0 (144: 8 blocks of 16 bytes - byte j value j low, j + 16 high - then 8 half scales); its rows
 * widened to half as they are staged (q35.h: ns_q35_kv_store_q writes them) */
int ns_q35_attn_prompt_kv(ns_gpu* g, int kt, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                          int64_t cap, int64_t p0, float* out);
int ns_q35_attn_decode_kv(ns_gpu* g, int kt, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                          int64_t cap, int64_t p0, float* out, float* scratch);
/* Weights to half for a prompt pass's GEMMs (dequant.cpp; the shared one is ns_dequant_f16): n values (a multiple of
 * 256) of a stored matrix into dst. Covers Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, IQ4_XS */
int ns_q35_dequant_f16_supported(int type);
int ns_q35_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst);
/* RMS norm with the Q8_1 blocks of its output (norm.cpp; for decode rows): y [rows, C] = x / rms(x) * w (y may be null),
 * q8_1 the blocks of y as ns_quantize_q8_1 makes them (column r at r x C / 32), xs (nullable) the same in the split
 * layout; C a multiple of 256 */
int ns_q35_rms_q8(ns_gpu* g, const float* x, const float* w, float* y, void* q8_1, void* xs, int64_t rows, int64_t C, float eps);

/* The DSpine draft's code bridge and refinement (dspine.cpp; llm/qwen35/src/dspine.rs): row-wise steps over at most 8 rows.
 * bridge_norm: y = x / (rms + eps), rms[r] = sqrt(max(mean x^2, tiny)) of each row [rows, C].
 * codes: row 0 = anchor_code (sqrt(C) x the anchor token's codebook row), row r > 0 = sqrt_c x proj / (|proj| + eps).
 * message: packed [rows, 3R] = (q | s | v) rows: msg[0] = 0, msg[i] = sigmoid(q[i] + s[i-1] + bias) x v[i-1].
 * write: out[0] = raw[0], out[i] = raw[i] + rms[i] x delta[i]. */
int ns_q35_dsp_bridge_norm(ns_gpu* g, const float* x, float* y, float* rms, int64_t rows, int64_t C, float eps);
int ns_q35_dsp_codes(ns_gpu* g, const float* proj, const float* anchor_code, float* codes, int64_t rows, int64_t C, float sqrt_c, float eps);
int ns_q35_dsp_message(ns_gpu* g, const float* packed, const float* gate_bias, float* msg, int64_t rows, int64_t R);
int ns_q35_dsp_write(ns_gpu* g, const float* raw, const float* delta, const float* rms, float* out, int64_t rows, int64_t H);
/* The refinement (n draft rows, K candidates each): base[r][k] = (raw[r] . norm_w) . head row of candidate k (rows_f16
 * [n][K][H] half); pick: for each (row r, sender k) - the anchor for row 0, else row r - 1's candidate k - the candidate
 * of row r with the best base + rms[r] x (m . tg), m = sigmoid(q[r] + tsv[sender][:R] + bias) x tsv[sender][R:] (tables
 * [vocab][2R] and [vocab][R] half): idx[r][k] */
int ns_q35_dsp_base(ns_gpu* g, const float* raw, const float* norm_w, const void* rows_f16, float* base, int64_t n, int64_t K, int64_t H);
int ns_q35_dsp_pick(ns_gpu* g, const float* base, const float* rms, const float* q, int64_t qs, const int32_t* cand, int32_t anchor, const void* tsv_f16,
                    const void* tg_f16, const float* gate_bias, int32_t* idx, int64_t n, int64_t K, int64_t R);

#ifdef __cplusplus
}
#endif
#endif
