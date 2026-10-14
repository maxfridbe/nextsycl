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

/* Each of T x H rows of x (row t, head h at x[t * xs + h * D]) RMS-normed with w [D], then (rope != 0) rotated as
 * llama.cpp's NORM rope does (pairs 2i, 2i + 1 at angle (p0 + t) * theta^(-2i / D)) into y (same layout, ys apart) */
int ns_mg_qk_norm_rope(ns_gpu* g, const float* x, int64_t xs, float* y, int64_t ys, int64_t T, int64_t H, int64_t D, const float* w, float eps,
                       int rope, float theta, int64_t p0);
/* Rows k, v [T, Hkv x D] (float32) at positions p0.. into the half caches [Hkv][cap][D], slot pos % cap */
int ns_mg_kv_store(ns_gpu* g, const float* k, const float* v, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, void* kc, void* vc);
/* Floats of scratch ns_mg_attn needs (0: none) */
int64_t ns_mg_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0, int64_t win);
/* Causal attention of rows q [T, Hq x D] at positions p0.. against the ring caches (cap slots; row p sees positions
 * max(0, p - win + 1) ..= p; win <= 0: all), softmax(q k^T / sqrt(D)) v: out [T, Hq x D]. D 128, Hq / Hkv 4, 8 or 16 */
int ns_mg_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
               int64_t win, float* out, float* scratch);
/* x [n] *= sigmoid(gate [n]) */
int ns_mg_gate_mul(ns_gpu* g, float* x, const float* gate, int64_t n);

#ifdef __cplusplus
}
#endif
#endif
