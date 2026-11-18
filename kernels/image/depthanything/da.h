/* da.h: Depth Anything V2's own kernels (kernels/image/depthanything), beside the shared vision kernels (vision.h:
 * im2col, the align-corners resize, add) and the diffusion kernels' GEMMs. Bound by name by image/depthanything/src/
 * ffi.rs. Token matrices are float32 row-major [M, C]; feature maps float32 channels-last [H, W, C]. Every call is
 * queued on g's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_DA_H
#define NS_IMAGE_DA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ns_gpu ns_gpu;
/* out [1 + N, C]: row 0 = cls + pos[0]; row 1 + i = patches[i] + bias + pos[1 + i] (patches [N, C] the patch
 * convolution's GEMM, its bias not yet added) */
int ns_image_da_embed(ns_gpu* g, const float* patches, const float* bias, const float* cls, const float* pos, int64_t N, int64_t C, float* out);
/* out[r] = (x[r] - mean) / sqrt(var + eps) * w + b, a row of C values (biased variance, as torch's LayerNorm) */
int ns_image_da_layernorm(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, const float* b, float eps, float* out);
/* x [M, C] += bias[c] (bias may be NULL), then act: 0 none, 1 ReLU, 2 GELU (the exact erf one) */
int ns_image_da_bias_act(ns_gpu* g, float* x, int64_t M, int64_t C, const float* bias, int act);
/* out = act(x) over n values (act 1 ReLU), x untouched */
int ns_image_da_act(ns_gpu* g, const float* x, int64_t n, int act, float* out);
/* x [M, C] += (y [M, C] + bias[c]) * ls[c]  (a block's output with its LayerScale, into the residual stream) */
int ns_image_da_ls_add(ns_gpu* g, float* x, const float* y, const float* bias, const float* ls, int64_t M, int64_t C);
/* Multi-head self attention without a mask: qkv [N, 3 H D] (q | k | v, heads of D side by side), out [N, H D];
 * softmax(q k^T / sqrt(D)) v per head, in float32 (the running maximum, as an online softmax) */
int ns_image_da_attention(ns_gpu* g, const float* qkv, int64_t N, int64_t H, int64_t D, float* out);
/* ConvTranspose2d with kernel = stride = k: y [h w, k k Co] (row p, column (ky k + kx) Co + co: the GEMM of x [h w, Ci]
 * with the weight [k k Co, Ci]) -> out [h k, w k, Co] + bias[co] */
int ns_image_da_shuffle(ns_gpu* g, const float* y, int64_t h, int64_t w, int64_t k, int64_t Co, const float* bias, float* out);
/* bilinear resize x [Hi, Wi, C] -> out [Ho, Wo, C], half-pixel centres (PyTorch's align_corners=False: the source
 * coordinate clamped at 0, the neighbour at the edge), out's rows `os` values apart */
int ns_image_da_resize(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t os);
#ifdef __cplusplus
}
#endif
#endif
