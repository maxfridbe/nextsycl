/* mm3q.h: decode products against Q8_1 activations for the types this engine's attention and shared experts hold (mmvq.cpp, a copy of the qwen35 silo's) */
#ifndef NS_MM3Q_H
#define NS_MM3Q_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

int ns_mm3q_mmvq_supported(int type, int64_t n_in);
int ns_mm3q_q8_split(ns_gpu* g, const void* q8_1, void* xs, int64_t n_in, int64_t ncols);
size_t ns_mm3q_q8_split_bytes(int64_t n_in, int64_t ncols);
int ns_mm3q_quantize_q8(ns_gpu* g, const float* x, void* q8_1, void* xs, int64_t n_in, int64_t ncols);
int ns_mm3q_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, const void* x_split, float* y, int64_t n_in, int64_t n_out, int64_t ncols);

#ifdef __cplusplus
}
#endif
#endif
