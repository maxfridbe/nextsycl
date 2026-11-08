/* rc.h: the ricursive engine's own kernels (Ricursive Ripple-65B: dense, sliding-window RoPE layers and global NoPE
 * layers, per-head q / k norms, an elementwise sigmoid gate on the attention output, a value residual from the scaled
 * embeddings), its C ABI, bound by its crate by name (llm/ricursive/src/ffi.rs). The products, the norms, the adds
 * and the clamped SwiGLU are the shared ones (the glm5next engine's glm.h). Float32 activations, row-major, all
 * queued on the GPU's queue; 0 or -1 (the reason: ns_last_error). */
#ifndef NS_RC_H
#define NS_RC_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Per (row t, head h): x = src[t * ss + h * hs .. + D] RMS-normed (weight w [D], eps); with rope != 0 turned by
 * rotate-half (NEOX) rotary positions over all D features (pairs i, i + D/2; angle (p0 + t) * theta^(-2i / D)); into
 * dst [t * ds + h * D] */
int ns_rc_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                       const float* w, float eps, int rope, float theta, int64_t p0);
/* The cache [Hkv][R][D] half, a ring of R rows a head: rows k [T, Hkv x D] (ks floats apart) and v (vs apart) at
 * positions p0.. (position p in slot p % R) */
int ns_rc_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t R, int64_t p0,
                   void* kc, void* vc);
/* Floats of scratch ns_rc_attn needs (0: none) */
int64_t ns_rc_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t W, int64_t p0);
/* Causal attention of rows q [T, Hq x D] (qs floats apart) at positions p0.. against the half cache [Hkv][R][D]: a row
 * at position p sees the keys p - W + 1 .. p (W 0: all of 0 .. p); scale 1 / sqrt(D). D 128, Hq / Hkv 10 (or 8, 6,
 * 4, 2, 1). The keys of a window must still be in the ring: R >= W + T - 1 (W 0: R the context). out [T, Hq x D] */
int ns_rc_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t R,
               int64_t W, int64_t p0, float* out, float* scratch);
/* out [n] *= sigmoid(gate [n]) */
int ns_rc_gate_mul(ns_gpu* g, float* out, const float* gate, int64_t n);
/* x [n] *= s */
int ns_rc_scale(ns_gpu* g, float* x, int64_t n, float s);
/* y [n] = x0 + s * (h - x0): the value residual's mix of the scaled embeddings x0 and the normed block input h */
int ns_rc_vres(ns_gpu* g, const float* x0, const float* h, float* y, int64_t n, float s);
/* y [rows, C] += rms_norm(x) * w: the post norm and the residual add in one launch (y may not alias x) */
int ns_rc_norm_add(ns_gpu* g, float* y, const float* x, const float* w, int64_t rows, int64_t C, float eps);

#ifdef __cplusplus
}
#endif
#endif
