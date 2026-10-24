/* ace.h: the ACE-Step 1.5 engine's own kernels (audio/acestep), beside the shared diffusion kernels (nsd.h). Bound by
 * its crate by name (audio/acestep/src/ffi.rs). 16-bit buffers are bfloat16; every call is queued on the GPU's queue
 * (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_AUDIO_ACE_H
#define NS_AUDIO_ACE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Bidirectional attention inside a window: query row i (of S) reads keys j with |i - j| <= w (all of them when w < 0).
 * q bfloat16, row i's Hq heads x 128 at q + i * qs; k / v bfloat16, Hkv heads x 128 at + j * kvs (query head h reads
 * key head h / (Hq / Hkv)); scores scaled by `scale`, the softmax in float32; out bfloat16 rows of Hq x 128, os apart. */
int ns_audio_ace_attn_window(ns_gpu* g, const void* q, int64_t qs, const void* k, const void* v, int64_t kvs, int64_t S, int64_t Hq, int64_t Hkv,
                             int64_t w, float scale, void* out, int64_t os);

/* The same attention over sequences of `seg` rows side by side: row i reads the keys of rows [seg (i / seg), seg (i / seg)
 * + seg) alone (an attention pooler's or detokenizer's patches, each a sequence of its own). */
int ns_audio_ace_attn_segments(ns_gpu* g, const void* q, int64_t qs, const void* k, const void* v, int64_t kvs, int64_t S, int64_t Hq, int64_t Hkv,
                               int64_t seg, float scale, void* out, int64_t os);

/* Oobleck's snake, in place allowed: out[c, t] = x[c, t] + inv_beta[c] * sin(alpha[c] x[c, t])^2 for x float32 [C, L]
 * (alpha = exp of the checkpoint's, inv_beta = 1 / (exp(beta) + 1e-9): made once at load) */
int ns_audio_ace_snake(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, const float* inv_beta, float* out);

#ifdef __cplusplus
}
#endif
#endif
