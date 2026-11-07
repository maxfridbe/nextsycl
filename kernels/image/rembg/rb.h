/* rb.h: the background removers' own kernels (kernels/image/rembg), beside the shared vision kernels (vision.h) and
 * the diffusion kernels' GEMMs. Bound by name by image/rembg/src/ffi.rs. Pictures are float32 channels-last [H, W, C];
 * every call is queued on g's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_RB_H
#define NS_IMAGE_RB_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ns_gpu ns_gpu;
/* max-pool 2 x 2, stride 2, ceil mode (a last odd row or column pooled alone): x [H, W, C] -> out [ceil(H/2), ceil(W/2), C] */
int ns_image_rb_maxpool2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out);
/* bilinear resize, half-pixel centres (align_corners false: PyTorch's interpolate, ONNX's half_pixel), the source
 * coordinate clamped at the edges: x [Hi, Wi, C] -> out [Ho, Wo, C], out's rows `out_stride` values apart */
int ns_image_rb_resize(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t out_stride);
/* x += bias[c] over [M, C]; then ReLU when relu */
int ns_image_rb_bias_act(ns_gpu* g, float* x, int64_t M, int64_t C, const float* bias, int relu);
/* x = 1 / (1 + exp(-x)) over n values */
int ns_image_rb_sigmoid(ns_gpu* g, float* x, int64_t n);
#ifdef __cplusplus
}
#endif
#endif
