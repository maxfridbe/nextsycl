/* ds41.h: the deepseek41 engine's kernels (DeepSeek-V4.1-Flash, YoungAi's VQ GGUF), its silo
 * (dist/silo/libnextsycl-deepseek41.so). Its generic steps (norms, GEMMs, the mHC mixes and Sinkhorn, top-k, the q4_K
 * decode products) are the glm5next engine's (glm.h); these are what this model needs and no other engine has: the
 * routed experts' vector quantization, the fp8_32x32 projections, the Engram gate, the model's low-precision key
 * simulations, the indexer at compress ratio 1 and 2 with its 32 heads. The attention, the RoPE and the compressor's
 * pooling are deepseek4's silo kernels, copied (ds41.cpp, attn.cpp). There is no shared kernel for the VQ products:
 * the engine needs this library. */
#ifndef NS_DS41_H
#define NS_DS41_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- ds41.cpp (deepseek4's, copied) */
/* YaRN RoPE (ggml's rope_yarn, adjacent pairs) in place: T rows `row` floats apart, H heads `head` apart, n_dims values
 * from float `off` of each head; row t at position pos0 + t * pstep; `inverse`: the rotation back */
int ns_ds41_rope(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t row, int64_t head, int64_t off, int64_t n_dims, int64_t pos0, int64_t pstep,
                 float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1, int inverse);
/* the compressor's pooling: blocks of `ratio` tokens completed by [pos0, pos0 + T), each value a softmax over the
 * block's tokens of score + ape weighting kv (kv, sc [T, D]; the tokens before pos0 from `ring` [ring_n][2D]) */
int ns_ds41_pool(ns_gpu* g, float* ring, const float* kv, const float* sc, const float* ape, float* out, int64_t pos0, int64_t T, int64_t D,
                 int64_t ratio, int overlap, int64_t ring_n);
int ns_ds41_ring_put(ns_gpu* g, const float* kv, uint16_t* ring, int64_t T, int64_t pos0, int64_t R, int64_t D);
int ns_ds41_rows_f16(ns_gpu* g, const float* x, uint16_t* cache, int64_t n, int64_t j0, int64_t D);
/* MQA attention, heads of 512: a row's last `window` raw keys (ring fp16 [R][512]) and the compressed ones it sees
 * (comp fp16 [n][512], j < (p + 1) / ratio; with sel [T][K], those it lists when it sees more than K), a sink a head */
int ns_ds41_attend(ns_gpu* g, const float* q, const uint16_t* raw, int64_t R, const uint16_t* comp, int64_t ratio, const int32_t* sel, int64_t K,
                   const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window, float scale);
int ns_ds41_attend_xmx_supported(int64_t H, int64_t D);
int ns_ds41_attend_xmx(ns_gpu* g, const float* q, const uint16_t* raw, int64_t R, const uint16_t* comp, int64_t ratio, const int32_t* sel, int64_t K,
                       const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window, float scale);
/* dst [B][A][C] = src [A][B][C] */
int ns_ds41_permute(ns_gpu* g, const float* src, float* dst, int64_t A, int64_t B, int64_t C);
/* n bytes by a kernel on the compute queue (not a copy engine): its ticket; the host waits for a ticket */
int ns_ds41_kcopy(ns_gpu* g, void* dst, const void* src, size_t n, int64_t* ticket);
int ns_ds41_ticket_wait(ns_gpu* g, int64_t t);

/* ---- k41.cpp: this model's own */
/* The model's low-precision simulations in place on x [rows][cols], a scale a block of `block` values (kernel.py's
 * act_quant / fp4_act_quant, quantized and back): kind 0 E4M3 with a power-of-two scale (amax >= 1e-4; the window's
 * keys, the attention output), 1 E2M1 with a power-of-two scale (amax >= 6 x 2^-126; the indexer's queries and keys),
 * 2 E2M1 with an E4M3 scale (amax >= 6 x 2^-9; the compressed keys) */
int ns_ds41_quant_sim(ns_gpu* g, float* x, int64_t rows, int64_t cols, int64_t block, int kind);
/* h [T, C] = sum_s pre[t][s] X[t][s] (X [T, 4, C]): a half's input by the previous half's mix */
int ns_ds41_hc_collapse(ns_gpu* g, const float* X, const float* pre, float* h, int64_t T, int64_t C);
/* The Engram's table rows: raw [n][hd + hd / 32] (E4M3, then an E8M0 scale a 32) -> out [n][hd], rounded to bf16 */
int ns_ds41_engram_rows(ns_gpu* g, const uint8_t* raw, float* out, int64_t n, int64_t hd);
/* The Engram's gate (Engram.forward) on the streams X [T, 4, C] in place: kv [T][5C] (a key a stream, then the value),
 * qk [4][C] (q_weight x k_weight); per (t, s): dot = sum X key qk / sqrt(C) x rsqrt(mean X^2 + eps) x rsqrt(mean
 * key^2 + eps), gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))), X += gate x value */
int ns_ds41_engram_gate(ns_gpu* g, float* X, const float* kv, const float* qk, int64_t T, int64_t C, float eps);
/* The indexer's scores: score [T][ld] = sum_h w[t][h] relu(q[t][h] . k_j) for j < min(n, (pos0 + t + 1) / ratio),
 * -inf to ld; q [T][H][D] float, k fp16 [n][D] */
int ns_ds41_idx_score(ns_gpu* g, const float* q, const float* w, const uint16_t* k, float* score, int64_t T, int64_t H, int64_t D, int64_t n,
                      int64_t ld, int64_t pos0, int64_t ratio);
/* The candidate source's blocks: bmax [T][nb] = the best score of each `bs` positions (score [T][ld], n of them), the
 * block of a row's newest position +inf */
int ns_ds41_block_max(ns_gpu* g, const float* score, float* bmax, int64_t T, int64_t n, int64_t ld, int64_t bs, int64_t pos0, int64_t ratio);
/* The index sources after the candidate source: scores only over each row's candidate blocks (cand [T][K] ids of 8
 * keys, ascending) into score [T][K x 8] (-inf where the row sees no key), then the picks' compact indices (selc
 * [T][kp]) mapped back to positions (sel [T][kp]) */
int ns_ds41_idx_score_cand(ns_gpu* g, const float* q, const float* w, const uint16_t* k, const int32_t* cand, int64_t K, float* score, int64_t T, int64_t H,
                           int64_t D, int64_t n, int64_t pos0, int64_t ratio);
int ns_ds41_cand_pos(ns_gpu* g, const int32_t* selc, const int32_t* cand, int32_t* sel, int64_t T, int64_t K, int64_t kp, int64_t bs);
/* score [T][ld] to -inf outside the blocks cand [T][K] lists (mark: scratch of T x nb bytes) */
int ns_ds41_cand_mask(ns_gpu* g, float* score, const int32_t* cand, uint8_t* mark, int64_t T, int64_t n, int64_t ld, int64_t K, int64_t bs);

/* glm5next's ns_hc_pre_fused with h (and its norm) collapsed by pre_in, the half before's mix */
int ns_ds41_hc_pre_in(ns_gpu* g, const float* x, const float* pre_in, const float* fn, const float* scale, const float* base, const float* nw, float* h,
                      float* post, float* comb, float* pre, float* normed, float* part, int64_t T, int64_t C, float eps, float hc_eps, int iters);
/* the attention's output: each head's last rd values rotated back at their positions, then the E4M3 simulation in 32s;
 * x [T][H][D] */
int ns_ds41_att_post(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t D, int64_t rd, int64_t pos0, float freq_base, float freq_scale, float ext_factor,
                     float mscale, float corr0, float corr1);
/* one column against two q4_K matrices of the same x (cols a multiple of 256, rows1 of 4): y1 = W1 . x, y2 = W2 . x */
int ns_ds41_q4k_mv2(ns_gpu* g, const void* w1, int64_t rows1, const void* w2, int64_t rows2, int64_t cols, const float* x, float* y1, float* y2);
/* one float32 row against two matrices: y1 [N1] = W1 [N1][K] . x, y2 [N2] = W2 . x */
int ns_ds41_gemv2(ns_gpu* g, const float* x, const float* w1, int64_t n1, const float* w2, int64_t n2, int64_t K, float* y1, float* y2);
/* the vision sidecar (vision.cpp): the ViT's 2D rotation of q (in place in qkv [N][3][H][64]) and k (into kc [H][N][64]),
 * a row softmax of s * scale, a bias with GELU (act 1), silu(gate) * up of gu [rows][2f] */
int ns_ds41_vit_rope(ns_gpu* g, float* qkv, float* kc, int64_t N, int64_t H, int64_t n_w, float theta);
int ns_ds41_softmax_rows(ns_gpu* g, float* s, int64_t rows, int64_t n, float scale);
int ns_ds41_bias_act(ns_gpu* g, float* y, const float* b, int64_t rows, int64_t c, int act);
int ns_ds41_silu_mul(ns_gpu* g, const float* gu, float* h, int64_t rows, int64_t f);
/* a layer's window keys fused: rms norm (w), the last rd values rotated at their positions, the E4M3 simulation in 32s,
 * into the fp16 ring (slot (pos0 + t) % R); kv [T][D] keeps the simulated values */
int ns_ds41_kv_post(ns_gpu* g, float* kv, const float* w, int64_t T, int64_t D, float eps, int64_t rd, int64_t pos0, float freq_base, float freq_scale,
                    float ext_factor, float mscale, float corr0, float corr1, uint16_t* ring, int64_t R);
/* A decode token's routing on the GPU and its products' tables from the layer's slot table (k41.cpp has the layout) */
int ns_ds41_route1(ns_gpu* g, const float* logits, const float* bias, int64_t ne, int64_t used, float scale, const uint64_t* tab, uint64_t dummy,
                   int64_t off_gate, int64_t off_up, int64_t off_down, uint8_t* gtab, uint8_t* dtab, int32_t* info, int32_t* ctab, const int32_t* tags);
/* dst[pairs[2i]] = pairs[2i + 1], n pairs */
int ns_ds41_scatter_u64(ns_gpu* g, uint64_t* dst, const int64_t* pairs, int64_t n);

/* The draft's Markov chain on the GPU: *tok = argmax(logits + bias) (the lowest index of a tie), n values */
int ns_ds41_markov_step(ns_gpu* g, const float* logits, const float* bias, int64_t n, int32_t* tok);
/* out [count][cols] = table rows idx[0..count) (the indices on the GPU) */
int ns_ds41_gather_rows(ns_gpu* g, const float* table, const int32_t* idx, int64_t count, int64_t cols, float* out);

/* ---- vq.cpp: the routed experts' VQ (YoungAi's DQV3 payloads) and the fp8_32x32 matrices */
/* whether an expert matrix of rows x cols at nbit (12 or 13) index bits is taken */
int ns_ds41_vq_supported(int64_t rows, int64_t cols, int64_t nbit);
/* An expert matrix expanded: payload = its DQV3 payload (the 32-byte header, f16 gains [rows], the 12-bit stream
 * [rows][cols / 8 x 12 / 8 bytes], at 13 bits the plane [rows][cols / 64 bytes]), cb = the layer's E4M3 codebook
 * [2^nbit][8]; out [rows][cols] as fp16 (f16 = 1) or float32 */
int ns_ds41_vq_dequant(ns_gpu* g, const void* payload, const uint8_t* cb, int64_t rows, int64_t cols, int64_t nbit, void* out, int f16);
/* Decode products: entry e = the matrix at ptrs[e] against meta[e] = {ncol, y0, col_0 .. col_7} columns of x [*][cols]:
 * y[(y0 + c) * rows + r] = W . x[col_c]; variant 0 a sub-group a row (cbf: the float32 codebook, or null), 1 a lane a row (the bytes) */
int ns_ds41_vq_mmv(ns_gpu* g, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const uint8_t* cb, const float* cbf, const float* x, float* y,
                   int64_t rows, int64_t cols, int64_t nbit, int variant);
/* The same products by lookup table: for each column c < nx of x, xl[c * 33] = n (1..16) and n (entry, column slot)
 * pairs that read it; maxn the largest n. Each group of 8 values of the column against every codebook entry, in local
 * memory, then a row's sum of its indices' entries; U 64-column units a work-group, the runs' partials added in order.
 * Writes only the listed rows of y [yrows][rows] */
int ns_ds41_vq_lut(ns_gpu* g, const uint64_t* ptrs, const int32_t* meta, const int32_t* xl, int64_t nx, int64_t maxn, const uint8_t* cb, const float* x,
                   float* y, int64_t yrows, int64_t rows, int64_t cols, int64_t nbit, int64_t units_per_group);
/* A prompt pass's expert product on XMX from the indices: y [n][ldy] = x [n][cols] (fp16, rows ldx apart, readable to
 * a multiple of 64 rows) . W^T, W of `rows` rows: rows < split from payload p0, the rest p1's (an expert's gate and up
 * as one); cbh the layer's fp16 codebook (x 2^-8) */
int ns_ds41_vq_gemm(ns_gpu* g, const void* p0, const void* p1, int64_t split, int64_t rows, int64_t cols, int64_t nbit, const void* cbh, const void* x,
                    int64_t ldx, int64_t n, float* y, int64_t ldy);
/* the codebook's n E4M3 bytes as fp16 x 2^-8 (exact) */
int ns_ds41_vq_codebook_f16(ns_gpu* g, const uint8_t* cb, int64_t n, void* out);
/* the codebook's n E4M3 bytes as float32 x 2^-8 (vq_mmv's cbf: no conversion a weight; null: from the bytes) */
int ns_ds41_vq_codebook_f32(ns_gpu* g, const uint8_t* cb, int64_t n, float* out);
/* fp8_32x32 [rows][cols] (w: the E4M3 plane, then the E8M0 tile scales): rows [r0, r0 + n) expanded to float32 / fp16 */
int ns_ds41_fp8_dequant(ns_gpu* g, const uint8_t* w, int64_t rows, int64_t cols, int64_t r0, int64_t n, void* out, int f16);
/* y [t][ycs] = W . x for an fp8_32x32 W [rows][cols], t 1..8 columns of x (column c at x + c xcs); with rpg > 0, row r
 * reads x + (r / rpg) xgs (o_a's groups) */
int ns_ds41_fp8_mmv(ns_gpu* g, const uint8_t* w, int64_t rows, int64_t cols, const float* x, int64_t xcs, int64_t rpg, int64_t xgs, float* y, int64_t ycs,
                    int64_t t);

/* ---- std.cpp: the routed experts in ggml's row types (Unsloth's / llama.cpp's form: merged ffn_*_exps tensors by layer in
 * MXFP4, Q2_K, Q3_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; Q8_0 too). cols a multiple of 256, float32 activations */
int ns_ds41_std_supported(int type, int64_t cols);
/* Decode products, vq_mmv's table: entry e = the matrix of `rows` rows at ptrs[e] against meta[e] = {ncol, y0, col_0 .. col_7}
 * columns of x [*][cols]: y[(y0 + c) * rows + r] = W . x[col_c] */
/* ncmax: the most columns any entry has (1, 2, 4 or 8: the accumulators' count, fixed at compile time) */
int ns_ds41_std_mmv(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const float* x, float* y, int64_t rows, int64_t cols,
                    int64_t ncmax);
/* A matrix [rows][cols] in `type` expanded to out [rows][cols] as fp16 (f16 = 1) or float32 */
int ns_ds41_std_dequant(ns_gpu* g, int type, const void* src, int64_t rows, int64_t cols, void* out, int f16);

/* ---- mmvq.cpp: the same decode products (one column an entry) against activations in a split Q8 layout, dp4a (the
 * deepseek4 silo's scheme): q8_bytes of ncols x n_in, quantize_q8 x [ncols][n_in] float32 into it */
size_t ns_ds41_q8_bytes(int64_t n_in, int64_t ncols);
int ns_ds41_quantize_q8(ns_gpu* g, const float* x, void* xs, int64_t n_in, int64_t ncols);
/* the types it takes: Q8_0, Q2_K, q4_K, Q5_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ2_S, MXFP4; n_in a multiple of 256, at most 8192 */
int ns_ds41_mmvq_supported(int type, int64_t n_in);
/* std_mmv's table (entries with ncol 0 skipped, column col_0 of xs): y[y0 * n_out + r] = W . x[col_0] */
int ns_ds41_std_mmvq(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const void* xs, int64_t ncols, float* y,
                     int64_t n_out, int64_t n_in);

/* ---- q4k.cpp: y [t][ycs] = W . x for a q4_K W [rows][cols], t 1..8 columns of float32 x (column c at x + c xcs); with
 * rpg > 0, row r reads x + (r / rpg) xgs (o_a's groups: a row block its own input) */
int ns_ds41_q4k_mv(ns_gpu* g, const void* w, int64_t rows, int64_t cols, const float* x, int64_t xcs, int64_t rpg, int64_t xgs, float* y, int64_t ycs,
                   int64_t t);

#ifdef __cplusplus
}
#endif
#endif
