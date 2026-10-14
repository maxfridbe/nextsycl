/* silo.h: the museglimmer engine's silo - kernels tuned for Muse Glimmer 30B Q8_0 (every matrix Q8_0), in their own
 * library (dist/silo/libnextsycl-museglimmer.so, opened by the engine at load; without it the engine uses the shared
 * ones). A Q8_0 matrix is repacked once at load into its quants and its scales apart (lossless): 34-byte blocks leave
 * the quants 2-byte aligned, the repacked rows are 16-byte loads. */
#ifndef NS_MG_SILO_H
#define NS_MG_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* A repacked Q8_0 matrix [rows, n_in] (n_in a multiple of 32): its quants [rows][n_in] int8, then (from
 * ns_mg_q8r_scales_at) its scales [rows][n_in / 32] half */
size_t ns_mg_q8r_bytes(int64_t rows, int64_t n_in);
size_t ns_mg_q8r_scales_at(int64_t rows, int64_t n_in);
int ns_mg_q8r_pack(ns_gpu* g, const void* q8_0, int64_t rows, int64_t n_in, void* dst);
/* Activations x [ncols, n_in] float32 in the same split layout: quants [ncols][n_in] int8 (each 32 by amax / 127,
 * rounded as ggml's Q8_1 does), then the scales [ncols][n_in / 32] float */
size_t ns_mg_xq_bytes(int64_t n_in, int64_t ncols);
int ns_mg_xq(ns_gpu* g, const float* x, void* xq, int64_t n_in, int64_t ncols);
/* Decode products: y [ncols, n_out] = W [n_out, n_in] (repacked Q8_0) . x (ns_mg_xq), ncols 1..8 */
int ns_mg_mmvq(ns_gpu* g, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
/* A repacked Q8_0 matrix expanded to half (a prompt pass's GEMMs): dst [rows, n_in] */
int ns_mg_q8r_f16(ns_gpu* g, const void* w, int64_t rows, int64_t n_in, uint16_t* dst);

#ifdef __cplusplus
}
#endif
#endif
