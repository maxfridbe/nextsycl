/* silo.h: the inkling2 engine's silo - kernels tuned for Inkling-2-Mini's decode (dist/silo/libnextsycl-inkling2.so,
 * opened by the engine at load; without it the engine runs on its own kernels in the kind's library, ink.h). The decode
 * products read the weights in their GGUF blocks (Q8_0, NVFP4) against activations quantized as ggml's Q8_0 (as
 * llama.cpp's own products quantize them), an int8 dot a block; a row's arithmetic is the same whatever rows share its
 * pass (a verify row equals its one-row pass bit for bit). */
#ifndef NS_INK2S_SILO_H
#define NS_INK2S_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- the activations ---- */
/* rows x [T, n] quantized as ggml's Q8_0 (a block of 32: d = amax / 127, kept as half; q = round(x / d)): xq holds the
 * quants [T, n] int8, then (from ns_ink2s_xq_scales_at) the scales [T, n / 32] float */
size_t ns_ink2s_xq_bytes(int64_t T, int64_t n);
size_t ns_ink2s_xq_scales_at(int64_t T, int64_t n);
int ns_ink2s_xq(ns_gpu* g, const float* x, void* xq, int64_t T, int64_t n);
/* RMS norm (eps, weight w) straight to that layout: y (nullable) [T, n] the normed rows too */
int ns_ink2s_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t T, int64_t n, float eps);

/* ---- the products ---- */
/* whether the decode products take a type (8 Q8_0, 40 NVFP4; n_in a multiple of 64 for NVFP4, 32 for Q8_0) */
int ns_ink2s_mv_supported(int type, int64_t n_in);
/* y [T, n_out] = W [n_out, n_in] . x (xq: ns_ink2s_xq's layout of x [T, n_in]), T 1..16 */
int ns_ink2s_mv(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t T);
/* several products in one launch (at most 8): job j's matrix w[j] (type ty[j], rows[j] x n_in[j]) against xq[j] (T
 * rows) into y[j] [T, rows[j]] */
int ns_ink2s_mv_many(ns_gpu* g, int n, const int* ty, const void* const* w, const void* const* xq, float* const* y, const int64_t* rows,
                     const int64_t* n_in, int64_t T);

/* ---- the routed experts at decode width (T 1..16, K experts a row) ---- */
/* gate | up of pair p = t K + k (its expert ids[p], at w + ids[p] * stride: rows [2F, n_in]) against row t of xq, x the
 * expert's scale, through SiTU (beta, lb): act [T K, F] quantized straight into ns_ink2s_xq's layout (F a multiple of 32) */
int ns_ink2s_moe_gu(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const void* xq, void* act_q, int64_t T,
                    int64_t K, int64_t n_in, int64_t F, float beta, float lb);
/* the down products of the pairs (act_q [T K, F]) weighted and summed: y [T, n_out] = sum_k wts[t K + k] scale[e] (W_e .
 * act) */
int ns_ink2s_moe_down(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const float* wts, const void* act_q,
                      float* y, int64_t T, int64_t K, int64_t F, int64_t n_out);

#ifdef __cplusplus
}
#endif
#endif
