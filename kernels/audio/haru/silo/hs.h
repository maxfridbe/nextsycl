/* hs.h: Haru-4B's silo (dist/silo/libnextsycl-haru.so): the kernels its prompt's wide passes are tuned with. The
 * engine opens it at load and falls back to its own kernels (haru.h) without it (or with NS_SILO=0). 0 or -1
 * (ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when ns_haru_silo_gdn_rule takes heads of these sizes (Dk = Dv = 128, Hv a multiple of Hk) */
int ns_haru_silo_gdn_supported(int64_t Hk, int64_t Hv, int64_t Dk, int64_t Dv);

/* The gated delta rule over R rows, as ns_audio_haru_gdn_rule (the same float32 operations in the same order), with
 * each item's column of the state [Dk, Dv] held in registers across the rows - read once and written once a call
 * instead of twice a row (a prompt's 128 rows a pass: the engine's kernel waits on memory each row) */
int ns_haru_silo_gdn_rule(ns_gpu* g, const float* qkv, int64_t ldq, const float* ba, int64_t ldb, const float* A_log, const float* dt_bias, float* state,
                          int64_t R, int64_t Hk, int64_t Hv, int64_t Dk, int64_t Dv, float* out);

/* 1 when ns_haru_silo_attn takes these heads (Dh 256, at most 4 query heads a key head) */
int ns_haru_silo_attn_supported(int64_t Hq, int64_t Hkv, int64_t Dh);

/* A wide pass's causal attention over the half cache, gated: out[r, h * Dh + d] = softmax(q k / sqrt(Dh)) v *
 * sigmoid(gate[r * ldg + h * hs + d]) for rows r at positions p0 + r (keys [0, p0 + r]) - what ns_audio_haru_attn and
 * its merge give, in one pass a row (a sub-group a row: no pieces, no scratch). q float32 [R, Hq * Dh], kc / vc half
 * [Hkv, T, Dh] */
int ns_haru_silo_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t R, int64_t Hq, int64_t Hkv, int64_t Dh, int64_t T, int64_t p0,
                      const float* gate, int64_t ldg, int64_t hs, float* out);

#ifdef __cplusplus
}
#endif
