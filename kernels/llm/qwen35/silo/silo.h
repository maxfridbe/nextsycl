/* silo.h: the qwen35 engine's silo - kernels tuned for dense Qwen3.8-27B, in their own library
 * (dist/silo/libnextsycl-qwen35.so, opened by the engine at load; without it the engine uses the shared ones). Each op
 * says what it covers (`_supported`); the engine falls back to the shared kernel for everything else. */
#ifndef NS_Q35_SILO_H
#define NS_Q35_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Decode products (mmvq.cpp; the shared one is ns_mmvq): y [ncols, n_out] = W [n_out, n_in] (ggml type, stored
 * blocks) . x (Q8_1, ncols 1..8). Covers Q3_K, Q4_K, Q5_K, IQ4_NL, IQ3_S, IQ4_XS at a width of whole blocks */
int ns_q35_mmvq_supported(int type, int64_t n_in);
int ns_q35_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols);

#ifdef __cplusplus
}
#endif
#endif
