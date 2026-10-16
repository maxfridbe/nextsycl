/* rignet.h: the RigNet engine's own kernels (model3d/rignet; GPLv3 as RigNet), beside the shared diffusion kernels
 * (nsd.h: its linear layers, float32 on oneDNN). Bound by its crate by name (model3d/rignet/src/ffi.rs). Buffers are
 * float32 (indices int32) device memory of g; every call is queued on g's queue (the caller syncs) and returns 0, or
 * -1 with the reason in ns_last_error(). */
#ifndef NS_MODEL3D_RIGNET_H
#define NS_MODEL3D_RIGNET_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* x [M, C] in place: x = relu(x), then * scale + shift when scale is given (a BatchNorm after the ReLU, in eval form,
 * folded to one affine per channel) */
int ns_model3d_rignet_relu_affine(ns_gpu* g, float* x, int64_t M, int64_t C, const float* scale, const float* shift);
/* x in place, n values: kind 0 tanh, 1 sigmoid */
int ns_model3d_rignet_act(ns_gpu* g, float* x, int64_t n, int kind);
/* EdgeConv's messages: out [E, 2C] row e = [x[dst[e]], x[src[e]] - x[dst[e]]] for x [*, C] */
int ns_model3d_rignet_edge_feat(ns_gpu* g, const float* x, int64_t C, const int32_t* src, const int32_t* dst, int64_t E, float* out);
/* The max of each segment of rows: out[i * out_stride + out_off + c] = max over rows r in [rowptr[i], rowptr[i+1]) of
 * x[r * C + c], for i < N (an empty segment gives 0, as PyTorch Geometric's scatter does) */
int ns_model3d_rignet_seg_max(ns_gpu* g, const float* x, int64_t C, const int32_t* rowptr, int64_t N, float* out, int64_t out_stride,
                              int64_t out_off);
/* Columns into a wider buffer (a concatenation): dst[r * dst_stride + dst_off + c] = src[r * src_stride + c] for r < M,
 * c < C (src_stride 0: one row broadcast to every row) */
int ns_model3d_rignet_copy_cols(ns_gpu* g, const float* src, int64_t src_stride, int64_t M, int64_t C, float* dst, int64_t dst_stride,
                                int64_t dst_off);

#ifdef __cplusplus
}
#endif
#endif
