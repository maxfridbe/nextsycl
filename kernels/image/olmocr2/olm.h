/* olm.h: the olmOCR-2 engine's kernels (image/olmocr2: Qwen2.5-VL) - its own C ABI, bound by its crate by name
 * (image/olmocr2/src/ffi.rs). Every buffer is device memory of g; the work is queued on g's queue (the caller
 * syncs). Each returns 0, or -1 and ns_last_error(). */
#ifndef NS_IMAGE_OLM_H
#define NS_IMAGE_OLM_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* float32 -> half, n values */
int ns_image_olm_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
/* dst [rows][..] (row stride ds) = src [rows][..] (row stride ss), cols values a row */
int ns_image_olm_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols);
/* out [n][C] = src[idx[i]] */
int ns_image_olm_gather(ns_gpu* g, const float* src, int64_t C, const int32_t* idx, int64_t n, float* out);
/* Attention within segments: token i (of N, rows `stride` apart in q / k / v) attends to the tokens [lo[i], hi[i]);
 * H heads of D (D a multiple of 16, at most 256); out [N][H * D] */
int ns_image_olm_seg_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t stride, int64_t N, int64_t H, int64_t D,
                          const int32_t* lo, const int32_t* hi, float* out);
/* y = W x products on Q8_0 matrices repacked at load: qs int8 [n][k] (rows 16-byte aligned), d half [n][k / 32] (a scale a
 * block of 32); k a multiple of 256. q8r_gemv: y [n] = W x (+ bias); q8r_half: the matrix expanded to half [n][k] */
int ns_image_olm_q8r_gemv(ns_gpu* g, const int8_t* qs, const void* d, const float* x, int64_t n, int64_t k, const float* bias, float* y);
int ns_image_olm_q8r_half(ns_gpu* g, const int8_t* qs, const void* d, int64_t n, int64_t k, void* out);
/* One query row over n cached rows (flash decoding, chunks of 256 keys), as pocr's: q [Hq][D], k / v [n][Hkv * D]
 * (row stride kvs); out [Hq][D]; scratch of ns_image_olm_attn1_scratch(n, Hq, D) floats */
int64_t ns_image_olm_attn1_scratch(int64_t n, int64_t Hq, int64_t D);
int ns_image_olm_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                       int64_t D, float* scratch, float* out);

#ifdef __cplusplus
}
#endif
#endif
