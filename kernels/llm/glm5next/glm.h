/* glm.h: the glm5next engine's kernels (glm.cpp, fused.cpp, mmvq.cpp), its own ABI: the engine binds these by name
 * (llm/glm5next/src/ffi.rs). ns_gpu is ns.h's. */
#ifndef NS_GLM_H
#define NS_GLM_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- the bring-up kernels (glm.cpp): float32 activations, row-major, all queued on the GPU's queue ---- */
/* n values of a ggml-typed tensor (0 F32, 1 F16, 30 BF16, 8 Q8_0, 10-14 Q2_K..Q6_K, 16 IQ2_XXS) into float32 */
int ns_dequant(ns_gpu* g, int type, const void* src, size_t n, float* dst);
/* y [T, N] (+)= x [T, K] . w [N, K]^T; rows of x ldx floats apart, rows of y ldy apart */
int ns_gemm(ns_gpu* g, int64_t T, int64_t N, int64_t K, const float* x, int64_t ldx, const float* w, float* y, int64_t ldy,
            int accumulate);
/* batch products: y + b*sy [T, N] (+)= (x + b*sx) [T, K] . (w + b*sw) [N, K]^T */
int ns_gemm_batch(ns_gpu* g, int64_t batch, int64_t T, int64_t N, int64_t K, const float* x, int64_t ldx, int64_t sx, const float* w, int64_t sw,
                  float* y, int64_t ldy, int64_t sy, int accumulate);
/* mHC mixes: m [T, 24] = fn [24, n] . rms(x [T, n]); part: scratch of T * 32 * 25 floats */
int ns_hc_mix(ns_gpu* g, const float* x, const float* fn, float* m, float* part, int64_t T, int64_t n, float eps);
int ns_rms_norm(ns_gpu* g, const float* x, const float* w, float* y, int64_t rows, int64_t C, float eps);
int ns_layer_norm(ns_gpu* g, const float* x, const float* w, const float* b, float* y, int64_t rows, int64_t C, float eps);
/* mHC (docs/glm5next.md): m [T, 24] mixes; X [T, 4, C]; h [T, C]; post [T, 4]; comb [T, 16] */
int ns_hc_pre(ns_gpu* g, const float* m, const float* scale, const float* base, const float* X, float* h, float* post, float* comb,
              float* pre, int64_t T, int64_t C, float eps, int iters);
int ns_hc_pre_fused(ns_gpu* g, const float* x, const float* fn, const float* scale, const float* base, const float* nw, float* h,
                    float* post, float* comb, float* pre, float* normed, float* part, int64_t T, int64_t C, float eps, float hc_eps, int iters);   /* pre [T, 4]: scratch */
int ns_hc_post(ns_gpu* g, const float* y, const float* X, const float* post, const float* comb, float* Xo, int64_t T, int64_t C);
int ns_hc_mean(ns_gpu* g, const float* X, float* y, int64_t T, int64_t C);
/* KDA */
/* snap (nullable): the state after each row but the last, [T-1][k-1][D] (conv) / [T-1][H][d][d] (scan) - speculative rollback */
int ns_conv_silu(ns_gpu* g, const float* x, float* state, const float* w, float* out, int64_t T, int64_t D, int k, float* snap);
int ns_l2_norm(ns_gpu* g, float* x, int64_t rows, int64_t n, float eps);
int ns_kda_gate(ns_gpu* g, float* gate, const float* dt_bias, const float* A, int64_t T, int64_t H, int64_t dh, float low);
int ns_sigmoid(ns_gpu* g, float* x, int64_t n);
int ns_exp(ns_gpu* g, float* x, int64_t n);
int ns_kda_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* S, float* o,
                int64_t T, int64_t H, int64_t d, float* snap);
int ns_kda_out(ns_gpu* g, const float* o, const float* gate, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps);
/* FFN, MLA, experts */
int ns_swiglu_clamp(ns_gpu* g, const float* gate, const float* up, float* out, int64_t n, float limit);
int ns_swiglu_gu_f16(ns_gpu* g, const float* gu, uint16_t* out, int64_t T, int64_t F, float limit);
int ns_mla_attend(ns_gpu* g, const float* qa, const float* c, float* u, int64_t T, int64_t H, int64_t L, int64_t pos0, float scale);
/* the DSA indexer: pooled keys of the pools tokens [pos0, pos0 + T) complete (ring [4][2D]: the last tokens' ik | ig);
 * pool scores (S [T*H, n] = iq . pooled^T for pools [j0, j0 + n); score rows ld apart); top K per row; MLA over a
 * row's selected pools + its incomplete pool (sel_cnt[t] < 0: every earlier token) */
int ns_idx_pool(ns_gpu* g, float* ring, const float* ik, const float* ig, const float* ape, float* pooled, int64_t pos0, int64_t T,
                int64_t D);
int ns_idx_score(ns_gpu* g, const float* S, const float* w, float* score, int64_t T, int64_t H, int64_t j0, int64_t n, int64_t ld,
                 int64_t pos0);
int ns_topk(ns_gpu* g, const float* score, int32_t* sel, int64_t T, int64_t n, int64_t ld, int64_t K);
/* the prompt path's MLA as GEMMs: each row's cells (idx [T][NC], their count n [T]); a masked row softmax; batch
 * products without the transpose (y = x . w, w [K, N]) */
int ns_mla_cells(ns_gpu* g, const int32_t* sel, const int32_t* sel_cnt, int64_t T, int64_t K, int64_t pos0, int32_t* idx, int32_t* n,
                 int64_t NC);
int ns_softmax_masked(ns_gpu* g, float* S, int64_t R, int64_t H, int64_t NC, const int32_t* n, float scale);
int ns_gemm_batch_nn(ns_gpu* g, int64_t batch, int64_t T, int64_t N, int64_t K, const float* x, int64_t ldx, int64_t sx, const float* w,
                     int64_t ldw, int64_t sw, float* y, int64_t ldy, int64_t sy, int accumulate);
int ns_gemm_batch_h(ns_gpu* g, int64_t batch, int trans_w, int64_t T, int64_t N, int64_t K, const uint16_t* x, int64_t ldx, int64_t sx,
                    const uint16_t* w, int64_t ldw, int64_t sw, float* y, int64_t ldy, int64_t sy, int accumulate);
int ns_mla_attend_sel(ns_gpu* g, const float* qa, const uint16_t* c, float* u, int64_t T, int64_t H, int64_t L, int64_t pos0, float scale,
                      const int32_t* sel, const int32_t* sel_cnt, int64_t K);
int ns_scatter_add(ns_gpu* g, float* y, const float* src, const int32_t* idx, const float* w, int64_t n, int64_t C);
int ns_gather(ns_gpu* g, const float* src, const int32_t* idx, float* out, int64_t n, int64_t C);
int ns_add(ns_gpu* g, float* y, const float* x, int64_t n);
/* the prompt path's half GEMMs: a stored matrix expanded to fp16 (types the IQ kernels take), float32 -> fp16, and
 * y [T, N] (+)= x [T, K] . w [N, K]^T with fp16 x / w, float32 y */
int ns_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst);
int ns_to_f16(ns_gpu* g, const float* x, uint16_t* y, int64_t n);
int ns_gather_f16(ns_gpu* g, const float* src, const int32_t* idx, uint16_t* out, int64_t n, int64_t C);
int ns_gather_h(ns_gpu* g, const uint16_t* src, const int32_t* idx, uint16_t* out, int64_t n, int64_t C);
/* the latent cache in q8 (NS_KV=q8): rows [L/32 fp16 scales][L int8], L = 512 (glm.cpp) */
int ns_to_q8row(ns_gpu* g, const float* x, void* y, int64_t n, int64_t L);
int ns_gather_q8_h(ns_gpu* g, const void* src, const int32_t* idx, uint16_t* out, int64_t n, int64_t L);
int ns_mla_attend_sel_q8(ns_gpu* g, const float* qa, const void* c, float* u, int64_t T, int64_t H, int64_t L, int64_t pos0, float scale,
                         const int32_t* sel, const int32_t* sel_cnt, int64_t K);
int ns_gemm_f16(ns_gpu* g, int64_t T, int64_t N, int64_t K, const uint16_t* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                int accumulate);
/* MoE combine per token: y[t] += sum_{j in [t_ptr[t], t_ptr[t+1])} w[j] * rows[ent[j]], rows [entries, C] */
int ns_moe_combine(ns_gpu* g, float* y, const float* rows, const int32_t* t_ptr, const int32_t* ent, const float* w, int64_t T, int64_t C);

/* ---- decode-width products from the stored blocks (mmvq.cpp) ---- */
size_t ns_q8_1_bytes(int64_t n_in, int64_t ncols);
int ns_mmvq_supported(int type);
/* x [ncols, n_in] float32 -> Q8_1 (36 bytes per 32 values, column j at j * n_in / 32 blocks) */
int ns_quantize_q8_1(ns_gpu* g, const float* x, void* q8_1, int64_t n_in, int64_t ncols);
/* y [ncols, n_out] = W [n_out, n_in] . x for 1..8 columns; W in its ggml type */
int ns_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
/* a layer's routed experts in two launches (mmvq.cpp): see there */
int ns_moe_grouped_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
size_t ns_moe_scratch_bytes(int64_t entries, int64_t n_ff);
int ns_moe_grouped(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t groups, int64_t entries, const void* x_q8_1,
                   void* scratch, float* out, float limit, int lanes, int phase);   /* lanes per row: 4/8/16/32, 0 = the default;
                                                                                         phase: 0 both halves, 1 gate/up + activation, 2 down */
/* prompt chunks: an expert's gate | up (IQ2_XXS rows, N = 2 n_ff) or down (Q2_K rows, N = n_embd) times its tokens
 * (x [M, K] fp16, M a multiple of 32, of 64 above 64), the weights decoded inside the matrix-engine GEMM (fused.cpp):
 * out [M, N] float32 */
int ns_moe_fused_gu(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K);
int ns_moe_fused_down(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K);
/* gate | up in ESIMD (fused.cpp): M one of 32, 64, 128, 256; ns_moe_esimd_gu_max: the most rows it takes on g */
int ns_moe_esimd_gu_max(ns_gpu* g);
int ns_moe_fused_gu_esimd(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K);

#ifdef __cplusplus
}
#endif
#endif
