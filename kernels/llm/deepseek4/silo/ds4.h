/* ds4.h: the deepseek4 engine's own kernels (DeepSeek-V4: attention over a sliding window of raw keys plus compressed
 * ones, 512-wide MQA heads with a rotated tail and learned sinks, mHC streams, MXFP4 experts). Its generic steps (norms,
 * GEMMs, the mHC halves, the indexer's scores and top-k, the dense and IQ types' decode products, the grouped experts)
 * are the glm5next engine's (glm.h); these are what this model needs and no other engine has. */
#ifndef NS_DS4_H
#define NS_DS4_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* YaRN RoPE (ggml's rope_yarn, NORM: pairs of adjacent values) in place on x: T rows `row` floats apart, H heads `head`
 * floats apart, the rotated part `n_dims` values from float `off` of each head; row t at position pos0 + t * pstep.
 * corr0 / corr1 from ggml_rope_yarn_corr_dims; mscale multiplies cos and sin. `inverse`: the rotation back (ggml's
 * rope_back: the sine negated) */
int ns_ds4_rope(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t row, int64_t head, int64_t off, int64_t n_dims, int64_t pos0, int64_t pstep,
                float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1, int inverse);

/* The compressor: the blocks of `ratio` tokens completed by the tokens [pos0, pos0 + T) pooled, each value a softmax
 * over the block's tokens of their scores (+ ape [ratio][W], by position in the block) weighting their kv. kv, sc
 * [T, W] this pass's projections; W = D, or 2D with `overlap` (a block then also pools the previous block's tokens over
 * the first D values of their rows - zero / -inf before the first - and its own over the last D). A token before
 * pos0 comes from `ring` ([ring_n][2W]: kv | sc at slot pos % ring_n); then the ring takes this pass's last tokens.
 * out [blocks][D], the first block pos0 / ratio */
int ns_ds4_pool(ns_gpu* g, float* ring, const float* kv, const float* sc, const float* ape, float* out, int64_t pos0, int64_t T, int64_t D,
                int64_t ratio, int overlap, int64_t ring_n);

/* rows kv [T, D] float32 -> the raw keys' ring (fp16 [R][D], slot pos % R) at positions pos0.. */
int ns_ds4_ring_put(ns_gpu* g, const float* kv, uint16_t* ring, int64_t T, int64_t pos0, int64_t R, int64_t D);

/* rows x [n, D] float32 -> fp16 rows of `cache` from row j0 */
int ns_ds4_rows_f16(ns_gpu* g, const float* x, uint16_t* cache, int64_t n, int64_t j0, int64_t D);

/* Attention at a row's keys, D = 512, MQA (one key head; the key is the value too): q [T, H, D] -> out [T, H, D].
 * Row t (position p = pos0 + t) reads the raw keys of the last `window` positions (ring [R][D] fp16, slot pos % R)
 * and, with `comp` ([n][D] fp16, block j at row j; null: none), the blocks it sees (j < (p + 1) / ratio) - all of them,
 * or with `sel` ([T][K]) those it lists when it sees more than K. Scores x scale, a softmax with a learned sink per
 * head (sinks [H]: in the sum, no value), sum p v */
int ns_ds4_attend(ns_gpu* g, const float* q, const uint16_t* raw, int64_t R, const uint16_t* comp, int64_t ratio, const int32_t* sel, int64_t K,
                  const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window, float scale);

/* attn.cpp: the same attention on XMX (8 heads of a row as the DPAS rows; ns_ds4_attend takes it unless
 * NS_DS4_ATTN_XMX=0) */
int ns_ds4_attend_xmx_supported(int64_t H, int64_t D);
int ns_ds4_attend_xmx(ns_gpu* g, const float* q, const uint16_t* raw, int64_t R, const uint16_t* comp, int64_t ratio, const int32_t* sel, int64_t K,
                      const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window, float scale);

/* The head's mHC: from the mixes m [T, 4] (fn . rms(flatten X)) and X [T, 4, C]: pre = sigmoid(m * scale + base) + eps,
 * h [T, C] = sum_s pre[s] X[s] */
int ns_ds4_hc_head(ns_gpu* g, const float* m, const float* scale, const float* base, const float* X, float* h, int64_t T, int64_t C, float eps);

/* dst [B][A][C] = src [A][B][C] (float32) */
int ns_ds4_permute(ns_gpu* g, const float* src, float* dst, int64_t A, int64_t B, int64_t C);

/* MXFP4 (ggml type 39: blocks of 32 - an E8M0 scale, 16 bytes of 4-bit E2M1 codes): n values expanded to float32 /
 * fp16; decode products y [ncols, n_out] = W [n_out, n_in] . x (Q8_1 blocks, ncols 1..8) */
int ns_ds4_mxfp4_dequant(ns_gpu* g, const void* src, int64_t n, float* dst);
int ns_ds4_mxfp4_dequant_f16(ns_gpu* g, const void* src, int64_t n, uint16_t* dst);
int ns_ds4_mxfp4_mmvq(ns_gpu* g, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols);

/* n bytes (a multiple of 4) from src to dst by a kernel on the GPU's compute queue - not the runtime's memcpy, which
 * goes to a copy engine and waits there behind the expert store's copies (the deepseek2 silo's kcopy, copied): the
 * small tables up and the routers' logits down, through pinned memory. Its ticket */
int ns_ds4_kcopy(ns_gpu* g, void* dst, const void* src, size_t n, int64_t* ticket);
/* The host waits for ticket t (ns_mark, ns_ds4_kcopy) - not the whole queue */
int ns_ds4_ticket_wait(ns_gpu* g, int64_t t);

/* ---- idx.cpp: the lightning indexer at decode widths. score [T][ld] = sum_h w[t][h] relu(q16[t][h] . key_j) for the
 * blocks j < (pos0 + t + 1) / 4 (and < n), -inf for the rest up to ld; q16 [T][64][128] fp16, cache [n][128] fp16 */
int ns_ds4_idx_score_dec(ns_gpu* g, const uint16_t* q16, const float* w, const uint16_t* cache, float* score, int64_t T, int64_t H, int64_t D,
                         int64_t n, int64_t ld, int64_t pos0);
/* the two-level top-k's glue: sel1 [T * nch][K] (positions in chunks of C of score [T][nch * C]) -> the candidates'
 * values and positions [T][nch * K]; sel2 [T][K] (indices into the candidates) -> positions sel [T][K] */
int ns_ds4_topk_gather(ns_gpu* g, const float* score, const int32_t* sel1, float* cval, int32_t* cpos, int64_t T, int64_t nch, int64_t C, int64_t K);
int ns_ds4_topk_map(ns_gpu* g, const int32_t* sel2, const int32_t* cpos, int32_t* sel, int64_t T, int64_t ncand, int64_t K);

/* ---- mmvq.cpp: a layer's experts at decode widths, the deepseek2 silo's scheme (copied): activations in a split Q8
 * layout (q8_bytes: ncols x n_in quants, then their fp16 scales a block of 32, then each 16 quants' sum) */
size_t ns_ds4_q8_bytes(int64_t n_in, int64_t ncols);
int ns_ds4_quantize_q8(ns_gpu* g, const float* x, void* xs, int64_t n_in, int64_t ncols);
/* the types it takes: Q8_0 (8), Q5_K (13), IQ2_XXS (16), IQ3_XXS (18), IQ2_S (22), MXFP4 (39), n_in a multiple of 256 */
int ns_ds4_multi_supported(int type, int64_t n_in);
/* y [entries, n_out]: entry e = the matrix at ptrs[e] (n_out rows of n_in, the type's blocks) against column cols[e] of xs */
int ns_ds4_mmvq_multi(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* cols, int64_t entries, const void* xs, int64_t ncols, float* y,
                      int64_t n_in, int64_t n_out);
/* the same with entries of different heights: entry e's rows [rstart[e], rstart[e + 1]) of y, total_rows in all */
int ns_ds4_mmvq_multi_var(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* cols, const int32_t* rstart, int64_t entries, const void* xs,
                          int64_t ncols, float* y, int64_t n_in, int64_t total_rows);

#ifdef __cplusplus
}
#endif
#endif
