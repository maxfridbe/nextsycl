/* ov.h: the OmniVoice engine's own kernels (audio/omnivoice) - the masked-diffusion step's pieces around the shared
 * products: the audio rows' input embeddings (the eight codebooks' rows summed), the grouped heads' keys and values laid
 * beside the queries for the full attention, and the step's scoring (classifier-free guidance on the two passes' log
 * probabilities, each position's best code and its log probability); and the codec encoder's per-channel norm and ELU.
 * The linear layers, norms, rotary positions, attention, convolutions and Snake are the shared diffusion kernels
 * (nsd.h) and the MiniMax Music engine's convolutions (mm3.h). Every call is queued on the GPU's queue and returns 0, or
 * -1 (the reason: ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Audio rows' input embeddings: out[rows[i] * ldo + j] = sum_c table[(ids[i * C + c] + c * V) * D + j], for i < n.
 * table float32 [C * V, D]; ids int32 [n, C] (a codebook's code, 0 <= code < V); rows int32 [n]; out float32. */
int ns_audio_ov_audio_embed(ns_gpu* g, const float* table, int64_t C, int64_t V, int64_t D, const int32_t* ids, const int32_t* rows, int64_t n,
                            float* out, int64_t ldo);
/* The fused projection's rows [q (Hq heads) | k (Hkv) | v (Hkv)] of D halfs each, R rows, into [q | k' | v'] with Hq
 * heads each (query head h reads key / value head h / (Hq / Hkv)): out half [R, 3 * Hq * D]. Hq a multiple of Hkv. */
int ns_audio_ov_expand_qkv(ns_gpu* g, const void* qkv, int64_t R, int64_t Hq, int64_t Hkv, int64_t D, void* out);
/* The step's scores. logits float32 [2 T, C * V]: the conditional pass's T target rows, then the unconditional pass's
 * T; for each codebook c and position t, over the V codes:
 *   lc = log_softmax(cond), lu = log_softmax(uncond), lp = log_softmax(lc + guidance (lc - lu))   (guidance 0: lp = lc)
 *   lp[mask] = -inf; tok[c * T + t] = argmax lp (the first on a tie), score[c * T + t] = max lp
 * tok int32 [C, T], score float32 [C, T]. */
int ns_audio_ov_score(ns_gpu* g, const float* logits, int64_t T, int64_t C, int64_t V, int64_t mask, float guidance, int32_t* tok, float* score);
/* Per channel over time (GroupNorm with a group a channel): x float32 [C, L], out = (x - mean) / sqrt(var + eps) * w + b
 * (w, b float32 [C]; the variance biased). out may be x. */
int ns_audio_ov_chan_norm(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* b, float eps, float* out);
/* ELU (alpha 1) over n float32 values: out = x > 0 ? x : exp(x) - 1. out may be x. */
int ns_audio_ov_elu(ns_gpu* g, const float* x, int64_t n, float* out);

#ifdef __cplusplus
}
#endif
