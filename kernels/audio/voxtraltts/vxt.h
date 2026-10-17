/* vxt.h: the Voxtral TTS engine's own kernels (audio/voxtraltts) - its codec decoder's causal attention with ALiBi
 * over a short window, an RMS norm of strided rows (the codec's q / k norms over a whole row of heads), the left
 * padding of its causal convolutions (replicate, reflect), and the flow transformer's six input rows. The backbone's
 * cached decode is the MiniMax Music engine's (mm3.h), the per-head RoPE and the flow's bidirectional attention the
 * Breeze TTS 2 engine's (bz2.h), layer-scaled residuals the Qwen3-TTS engine's (q3t.h), the norms and wide linear
 * layers the shared diffusion kernels (nsd.h). Float32 on the device; every call is queued on the GPU's queue and
 * returns 0, or -1 (the reason: ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Causal attention with ALiBi over a window: query i sees keys j in [i - W, i] (W + 1 of them), score
 * q.k / sqrt(D) + slope_h (j - i) with slope_h = 2^(-8 h / H) (upstream's geometric slopes, H a power of two);
 * q, k, v rows of H heads x D, qs / kvs floats apart; out [L, H x D]. D a multiple of 16, at most 256. */
int ns_audio_vxt_alibi_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t L, int64_t H, int64_t D, int64_t qs, int64_t kvs,
                            int64_t W, float* out);
/* RMS norm of each row's C values in place: x rows xs floats apart, x = x / sqrt(mean(x^2) + eps) * w (w float32 [C]) */
int ns_audio_vxt_rms_rows(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t C, const float* w, float eps);
/* out [C, P + L] = x [C, L] with P values ahead of each channel: mode 0 its first value repeated (replicate), mode 1
 * its mirror without the edge, x[P], ..., x[1] (reflect; P < L) */
int ns_audio_vxt_pad_left(ns_gpu* g, const float* x, int64_t C, int64_t L, int64_t P, int mode, float* out);
/* The flow transformer's input for both guidance passes, out [6, D]: [inp, tproj, hp] (conditioned) and [inp, tproj, 0]
 * (not) */
int ns_audio_vxt_fm_rows(ns_gpu* g, const float* inp, const float* tproj, const float* hp, int64_t D, float* out);

#ifdef __cplusplus
}
#endif
