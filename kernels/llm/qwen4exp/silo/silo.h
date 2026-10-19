/* silo.h: the qwen4exp engine's silo - kernels for Qwen3.8-Flash-Next (Coder, Swift) alone, in their own library
 * (dist/silo/libnextsycl-qwen4exp.so, opened by the engine at load; without it the engine uses the shared kernels).
 * The engine hands these to its glue (ns_qw_set_silo); each says what it covers, the shared path takes the rest. */
#ifndef NS_QW_SILO_H
#define NS_QW_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Prompt experts as fused int8 DPAS products (i8moe.cpp: Strata 0.2.0's ESIMD kernels, Battlemage): the weights decoded in
 * registers into the s8 DPAS operand, the activations quantized per 32 (q8_1's grid) - no FP16 copy of an expert.
 * ns_qw_i8_available: the queue's device runs them (1). ns_qw_i8_supported: these gate/up and down formats and shapes
 * (gate/up IQ3_XXS 18, IQ3_S 21, IQ2_S 22; down IQ4_NL 20, Q2_0 42). */
int ns_qw_i8_available(void* queue);
int ns_qw_i8_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/* One group of up to 16 experts on `queue` (a sycl::queue*, in order): X [rows][n_embd] FP16, the experts' rows back to back
 * (cnt[i] rows of expert i, whose blob blobs[i] holds gate rows at 0, up rows at up_off, down rows at down_off); GU [rows][2
 * n_ff] FP32 interleaved (gate, up), Dm [rows][n_embd] FP32. 0 done, 1 a format it does not take, -1 an error. */
int ns_qw_i8_group(void* queue, int gu_type, int d_type, const uint8_t* const* blobs, uint64_t up_off, uint64_t down_off,
                   const int32_t* cnt, int n, const uint16_t* X, int64_t n_embd, int64_t n_ff, float* GU, float* Dm);

#ifdef __cplusplus
}
#endif
#endif
