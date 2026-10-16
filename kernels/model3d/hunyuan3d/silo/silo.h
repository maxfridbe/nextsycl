/* silo.h: Hunyuan3D's silo (dist/silo/libnextsycl-hunyuan3d.so): its bandwidth-bound kernels written for its shapes
 * (the paint UNet's and the SD VAE's feature maps of 36 x 4,096 or 262,144 pixels and 128 - 2,560 channels, the DiT's
 * and decoders' rows), each with a _supported check. The engine (model3d/hunyuan3d/src/ffi.rs) calls them for what
 * they take and its own kernels (h3d.h) otherwise; the results are the same up to the order of float32 sums. */
#ifndef NS_MODEL3D_HY_SILO_H
#define NS_MODEL3D_HY_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ns_model3d_hunyuan3d_group_norm (h3d.h) with its statistics from many work-groups a frame (the engine's own: one
 * work-group a frame and group, 32 for a VAE picture) and the normalization two channels a lane. scratch: float32
 * [ns_model3d_hy_silo_group_norm_scratch(...)] */
int ns_model3d_hy_silo_group_norm_supported(int64_t F, int64_t N, int64_t C, int64_t G);
int64_t ns_model3d_hy_silo_group_norm_scratch(int64_t F, int64_t N, int64_t C, int64_t G);
int ns_model3d_hy_silo_group_norm(ns_gpu* g, const void* x, int64_t F, int64_t N, int64_t C, int64_t G, const float* w, const float* b, float eps,
                                  int silu, float* scratch, void* out);

/* ns_model3d_hunyuan3d_ln (h3d.h), a sub-group a row (the engine's own: a 256-lane work-group a row, most of it idle
 * at 320 channels) */
int ns_model3d_hy_silo_ln_supported(int64_t C);
int ns_model3d_hy_silo_ln(ns_gpu* g, const void* x, int x_half, int64_t M, int64_t C, float eps, const float* w, const float* b, const float* scale,
                          const float* shift, void* out, int out_half);

/* ns_model3d_hunyuan3d_geglu and ns_model3d_hunyuan3d_add (h3d.h), four values a lane, 32-bit indices */
int ns_model3d_hy_silo_geglu_supported(int64_t M, int64_t F);
int ns_model3d_hy_silo_geglu(ns_gpu* g, const void* x, int64_t M, int64_t F, void* out);
int ns_model3d_hy_silo_add_supported(int64_t M, int64_t C);
int ns_model3d_hy_silo_add(ns_gpu* g, float* x, const void* y, int y_half, int64_t M, int64_t C, const float* gate);

#ifdef __cplusplus
}
#endif
#endif
