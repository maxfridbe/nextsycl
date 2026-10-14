/* mg.h: the museglimmer engine's own kernels (Muse Glimmer 30B: llama.cpp's src/models/muse-glimmer.cpp) - what it
 * has that the shared ones do not: the q / k norms with rotary positions on the sliding-window layers only (llama.cpp's
 * NORM rotation: adjacent pairs), the key / value caches as rings (a sliding-window layer keeps its window and a
 * pass's rows), the causal attention over a window of them, and the attention's sigmoid output gate. */
#ifndef NS_MG_H
#define NS_MG_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Each of T x H rows of x (row t, head h at x[t * xs + h * D]) RMS-normed with w [D], then rotated at position p0 + t
 * into y (same layout, ys apart): rope 0 none, 1 llama.cpp's NORM (pairs 2i, 2i + 1 at angle pos theta^(-2i / D)), 2 its
 * NEOX (pairs i, i + D / 2 at the same angles: the DFlash drafter's) */
int ns_mg_qk_norm_rope(ns_gpu* g, const float* x, int64_t xs, float* y, int64_t ys, int64_t T, int64_t H, int64_t D, const float* w, float eps,
                       int rope, float theta, int64_t p0);
/* Rows k, v [T, Hkv x D] (float32) at positions p0.. into the half caches [Hkv][cap][D], slot pos % cap */
int ns_mg_kv_store(ns_gpu* g, const float* k, const float* v, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, void* kc, void* vc);
/* Floats of scratch ns_mg_attn needs (0: none) */
int64_t ns_mg_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0, int64_t win, int causal);
/* Attention of rows q [T, Hq x D] at positions p0.. against the ring caches (cap slots; row p sees positions
 * max(0, p - win + 1) ..= p - causal - or ..= p0 + T - 1, the whole block - not causal: the DFlash drafter's noise block;
 * win <= 0: from 0), softmax(q k^T / sqrt(D)) v: out [T, Hq x D]. D 128, Hq / Hkv a multiple of 4 */
int ns_mg_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
               int64_t win, int causal, float* out, float* scratch);
/* src [T, n] into columns col0 .. col0 + n of dst [T, ld] */
int ns_mg_put_cols(ns_gpu* g, const float* src, float* dst, int64_t T, int64_t n, int64_t ld, int64_t col0);
/* Columns col0 .. col0 + n of src [T, ld] into dst [T, n] */
int ns_mg_get_cols(ns_gpu* g, const float* src, float* dst, int64_t T, int64_t n, int64_t ld, int64_t col0);
/* Each row's draw, on the GPU: the host sampler's distribution (nextsycl-llm sampling.rs: softmax(x / temp) over the
 * 256 likeliest - a higher value first, then a lower index - cut to the smallest prefix holding top_p, renormalized)
 * drawn with one uniform a position (row t at position pos0 + t) from seed: x [T, n] -> out [T] */
int ns_mg_sample(ns_gpu* g, const float* x, int32_t* out, int64_t T, int64_t n, float temp, float top_p, unsigned long long seed, int64_t pos0);
/* Each row's largest: x [T, n] -> out [T] (the first of equals) */
int ns_mg_argmax(ns_gpu* g, const float* x, int32_t* out, int64_t T, int64_t n);
/* The post norm into the residual: res [rows, C] += x / rms(x) * w */
int ns_mg_rms_add(ns_gpu* g, const float* x, const float* w, float* res, int64_t rows, int64_t C, float eps);
/* The logits' output multiplier and soft cap in place: x = cap tanh(x scale / cap) (cap <= 0: x scale) */
int ns_mg_softcap(ns_gpu* g, float* x, int64_t n, float scale, float cap);
/* x [n] *= sigmoid(gate [n]) */
int ns_mg_gate_mul(ns_gpu* g, float* x, const float* gate, int64_t n);

#ifdef __cplusplus
}
#endif
#endif
