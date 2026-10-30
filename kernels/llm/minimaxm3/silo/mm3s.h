/* mm3s.h: the minimaxm3 silo (dist/silo/libnextsycl-minimaxm3.so, opened at load; without it, or with NS_SILO=0, the engine runs
 * the shared kernels of ../mm3.h). Kernels tuned for this model's shapes alone. */
#ifndef NS_MM3S_H
#define NS_MM3S_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Whether ns_mm3s_attend takes this geometry: heads of 128 values, 16 query heads a key/value group */
int ns_mm3s_attend_supported(int64_t H, int64_t G, int64_t head);

/* ns_llm_minimaxm3_attend (mm3.h) for a prompt pass: the same arguments and result (the scores' sums in another order), a
 * work-group a (query row, key/value group) sharing each staged tile of 64 keys and values among the group's 16 heads */
int ns_mm3s_attend(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T, int64_t H, int64_t G,
                   int64_t pos0, float scale);

/* ns_mm3s_attend on XMX (attnx.cpp): the same arguments and result, q to half, scores and output accumulated in float */
int ns_mm3s_attend_x(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T, int64_t H, int64_t G,
                     int64_t pos0, float scale);

/* ns_mm3s_attend_split on XMX: the same arguments, q to half */
int ns_mm3s_attend_x_split(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T, int64_t H, int64_t G,
                           int64_t pos0, float scale, int64_t S, float* part);

/* The same for a few query rows (decode): the keys split in chunks of 128, a work-group each (S = ceil(keys / 128) chunks,
 * keys = K * 128 selected or pos0 + T), their partial softmax states in `part` (T * G * S * 16 * 130 floats), merged by a second kernel */
int ns_mm3s_attend_split(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T, int64_t H, int64_t G,
                         int64_t pos0, float scale, int64_t S, float* part);

/* The routed experts of a decode token in ggml's row types (IQ1_M, IQ2_XXS, IQ3_XXS, IQ4_XS) on float32 activations. bases [entries]: device
 * addresses of the experts' [gate | up | down] blobs. gu: h [entries][rows] = swiglu_oai(gate . x_tok[e], up . x_tok[e]) (a verify pass's rows are entries with their own tok), the up matrix off_up bytes after the
 * gate's, rows = the expert width, cols = the model width. down: ys[rank[e] * plane + r] = w[e] * (down_e[r] . h_e), the down matrix off_down
 * bytes after the base (rows = the model width, cols = the expert width) */
int ns_mm3s_expert_supported(int type, int64_t cols);
int ns_mm3s_expert_gu(ns_gpu* g, int type, const uint64_t* bases, int64_t off_up, const float* x, const int32_t* tok, float* h, int64_t entries, int64_t rows, int64_t cols, float alpha,
                      float limit);
int ns_mm3s_expert_down(ns_gpu* g, int type, const uint64_t* bases, int64_t off_down, const float* h, const float* w, const int32_t* rank, const int32_t* tok, float* ys, int64_t plane,
                        int64_t entries, int64_t rows, int64_t cols);

/* y_i[r] = m_i[r] . x for up to three matrices of one type and width in a launch (r_i rows; r0 and r1 multiples of 8; float activations) */
int ns_mm3s_mv3(ns_gpu* g, int type, const void* m0, const void* m1, const void* m2, const float* x, float* y0, float* y1, float* y2, int64_t r0, int64_t r1, int64_t r2, int64_t cols);
/* A decode token's attention after its projections: q, k, iq, ik normed and rotated in place, k, v, ik stored as fp16 (at the token's position), a launch; IH = 0: no indexer */
int ns_mm3s_attn_post(ns_gpu* g, float* q, float* k, const float* v, float* iq, float* ik, const float* qn, const float* kn, const float* iqn, const float* ikn, uint16_t* kc, uint16_t* vc, uint16_t* ic,
                      int64_t H, int64_t G, int64_t IH, int64_t HD, int64_t n_rot, int64_t pos, float base, float eps);
/* A matrix's blocks as fp16 for a prompt pass's half GEMM: nv values, a multiple of 256; out 16-byte aligned */
int ns_mm3s_to_half(ns_gpu* g, int type, const void* src, int64_t nv, uint16_t* out);
/* When the work queued before it is done, *flag (pinned host memory) = seq */
int ns_mm3s_doorbell(ns_gpu* g, uint32_t* flag, uint32_t seq);
/* f[r * stride + at + c] = x[r * C + c] */
int ns_mm3s_put_cols(ns_gpu* g, const float* x, float* f, int64_t T, int64_t C, int64_t stride, int64_t at);
/* A small table (whole words) from pinned memory into VRAM in a kernel */
int ns_mm3s_kcopy(ns_gpu* g, uint32_t* dst, const uint32_t* src, int64_t words);
/* *best = the argmax of base + bias over n values */
int ns_mm3s_argmax_add(ns_gpu* g, const float* base, const float* bias, int64_t n, int32_t* best);
/* A draft block on XMX: T rows at pos0.. see the keys from w0 to the block's last row, not causal; S = ceil(keys / 128), part T * G * S * 16 * 130 floats */
int ns_mm3s_attend_block(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, float* out, int64_t T, int64_t H, int64_t G, int64_t pos0, int64_t w0, int64_t S, float* part);
/* att = o + x, hn = rms_norm(att) * w */
int ns_mm3s_add_norm(ns_gpu* g, float* att, const float* o, const float* x, const float* w, float* hn, int64_t C, float eps);
/* out = the `used` planes of ys in rank order + sh + att */
int ns_mm3s_finish(ns_gpu* g, const float* ys, int64_t used, const float* sh, const float* att, float* out, int64_t C);

#ifdef __cplusplus
}
#endif
#endif
