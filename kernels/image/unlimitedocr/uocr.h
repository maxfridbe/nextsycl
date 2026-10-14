/* uocr.h: the Unlimited-OCR engine's kernels (image/unlimitedocr) - its own C ABI, bound by its crate by name
 * (image/unlimitedocr/src/ffi.rs). Every buffer is device memory of g; the work is queued on g's queue (the caller
 * syncs). Each returns 0, or -1 and ns_last_error(). */
#ifndef NS_IMAGE_UOCR_H
#define NS_IMAGE_UOCR_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* float32 -> half, n values */
int ns_image_uocr_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
/* x = x * sigmoid(1.702 x) in place (CLIP's quick GELU) */
int ns_image_uocr_quick_gelu(ns_gpu* g, float* x, int64_t n);
/* Pictures [B][3][H][W] -> their P x P patches [B][H/P][W/P][3 * P * P], a patch's values (channel, y, x): a
 * stride-P convolution as a linear */
int ns_image_uocr_patchify(ns_gpu* g, const float* img, int64_t B, int64_t H, int64_t W, int64_t P, float* out);
/* A 3 x 3 convolution's columns (zero padding 1, stride s): x [B][H][W][C] -> [B][Ho][Wo][C * 9], each (channel, ky,
 * kx), Ho = (H + 2 - 3) / s + 1 */
int ns_image_uocr_im2col3(ns_gpu* g, const float* x, int64_t B, int64_t H, int64_t W, int64_t C, int64_t s, float* out);
/* Windows of w x w (zero padded to multiples of w): back = 0: x [B][H][W][C] -> [B][Hp / w][Wp / w][w][w][C];
 * back = 1: the reverse, the padding dropped (x the windows, out [B][H][W][C]) */
int ns_image_uocr_window(ns_gpu* g, const float* x, int64_t B, int64_t H, int64_t W, int64_t C, int64_t w, int back, float* out);
/* SAM's attention with decomposed relative positions over a Gh x Gw grid of N = Gh Gw tokens, B grids: qkv
 * [B][N][3][heads][D] (its qkv linear's output), out [B][N][heads][D] =
 *   softmax_j(q.k_j / sqrt(D) + q.Rh[qy - ky + Gh - 1] + q.Rw[qx - kx + Gw - 1]) v_j
 * rh [2 Gh - 1][D], rw [2 Gw - 1][D] (the tables resampled to the grid). N at most 4096 */
int ns_image_uocr_sam_attn(ns_gpu* g, const float* qkv, int64_t B, int64_t Gh, int64_t Gw, int64_t heads, int64_t D, const float* rh,
                           const float* rw, float* out);
/* Q5_0 blocks (32 values: half d, 32 high bits, 16 bytes of low nibbles) -> n halfs: (q - 16) d */
int ns_image_uocr_q5_0_half(ns_gpu* g, const void* src, int64_t n, void* out);
/* dst [rows][..] (row stride ds) = src [rows][..] (row stride ss), cols values a row */
int ns_image_uocr_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols);
/* out [n][C] = src[idx[i]] */
int ns_image_uocr_gather(ns_gpu* g, const float* src, int64_t C, const int32_t* idx, int64_t n, float* out);
/* dst[idx[i]] += w[i] src[i], src [n][C] (the indices distinct) */
int ns_image_uocr_scatter_add(ns_gpu* g, float* dst, int64_t C, const float* src, const int32_t* idx, const float* w, int64_t n);

#ifdef __cplusplus
}
#endif
#endif
