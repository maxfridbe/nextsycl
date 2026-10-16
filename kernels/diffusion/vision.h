/* vision.h: picture operations the foundation's vision networks share (crates/remover: BiRefNet's Swin-L and its
 * decoder) - linked, like the diffusion kernels, into every kind's library but the llm one, and bound by name by
 * the crate that uses them. Every buffer is float32 device memory of g, pictures channels-last [H, W, C] (a row of C
 * values a pixel); every call is queued on g's queue (the caller syncs). 0, or -1 and ns_last_error(). */
#ifndef NS_VISION_H
#define NS_VISION_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* im2col for a kh x kw convolution (stride s, zero padding p, dilation d) of x [H, W, C]: output pixels
 * [row0, row0 + rows) of the Ho x Wo result, each a row of kh * kw * C values in (ky, kx, c) order (the weight is
 * [Co, kh, kw, C]) */
int ns_vision_im2col(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, int64_t kh, int64_t kw, int64_t s, int64_t p, int64_t d,
                     int64_t Ho, int64_t Wo, int64_t row0, int64_t rows, float* out);
/* The modulated deformable im2col of torchvision's deform_conv2d (stride 1, one offset group, k x k taps, padding p,
 * output H x W): om [H * W, 3 k k] holds each pixel's tap offsets (dy, dx) pairs then the taps' mask logits (the mask
 * is 2 sigmoid(logit), as BiRefNet's DeformableConv2d); samples are bilinear, zero outside. Rows as ns_vision_im2col. */
int ns_vision_deform_im2col(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, int64_t k, int64_t p, const float* om,
                            int64_t row0, int64_t rows, float* out);
/* Bilinear resize with align_corners=True: x [Hi, Wi, C] -> out [Ho, Wo, C], out's rows `out_stride` values apart
 * (a column block of a wider buffer: a concatenation) */
int ns_vision_resize_ac(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t out_stride);
/* The picture split into G x G tiles stacked as channels (einops 'c (hg h) (wg w) -> (c hg wg) h w'):
 * x [H, W, C] -> out [H / G, W / G, C G G], channel c G G + gy G + gx = x[gy H/G + y, gx W/G + x, c] */
int ns_vision_tiles(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, int64_t G, float* out);
/* Swin's windows: x [H, W, C] zero-padded to Hp x Wp (multiples of ws), rolled by -shift, cut into ws x ws windows:
 * out [nW * ws * ws, C], windows row-major. scatter_add is the way back: x += the windows' rows (the padding
 * dropped) */
int ns_vision_window_gather(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, float* out);
int ns_vision_window_scatter_add(ns_gpu* g, const float* win, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, float* x);
/* Window attention of Swin (head size 32): qkv [nW * N, 3 C] (N = ws * ws; q | k | v, heads of 32), each window's
 * softmax(q k^T * scale + bias[h] + mask) v with bias [heads, N, N]; shift > 0: positions of different regions of the
 * rolled padded frame (Hp x Wp) get -100 (Swin's mask). out [nW * N, C]. */
int ns_vision_window_attention(ns_gpu* g, const float* qkv, int64_t nwh, int64_t nww, int64_t ws, int64_t heads, const float* bias, int64_t shift,
                               float scale, float* out);
/* Swin's patch merging gather: x [H, W, C] -> out [ceil(H/2), ceil(W/2), 4 C] = (x[2y, 2x], x[2y+1, 2x], x[2y, 2x+1],
 * x[2y+1, 2x+1]), zero past the edge */
int ns_vision_patch_merge(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out);
/* x = max(x, 0) over n values */
int ns_vision_relu(ns_gpu* g, float* x, int64_t n);
/* x [M, C] *= sigmoid(logit[r]) a row (logit [M]) */
int ns_vision_mul_sigmoid(ns_gpu* g, float* x, int64_t M, int64_t C, const float* logit);
/* dst[r, c] = src[r, c] for r < M, c < C; rows of src `ss` values apart, of dst `ds` (column blocks: concatenation) */
int ns_vision_copy_cols(ns_gpu* g, const float* src, int64_t M, int64_t C, int64_t ss, float* dst, int64_t ds);
/* out[c] = the mean of x[:, c] over M rows */
int ns_vision_col_mean(ns_gpu* g, const float* x, int64_t M, int64_t C, float* out);
/* dst[r, c] = v[c] for every row r < M (rows `ds` apart) */
int ns_vision_broadcast_row(ns_gpu* g, const float* v, int64_t M, int64_t C, float* dst, int64_t ds);
/* x += y over n values */
int ns_vision_add(ns_gpu* g, float* x, const float* y, int64_t n);

#ifdef __cplusplus
}
#endif
#endif
