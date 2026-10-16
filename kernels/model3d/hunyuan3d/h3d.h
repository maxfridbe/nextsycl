/* h3d.h: the Hunyuan3D engine's own kernels (model3d/hunyuan3d), beside the shared diffusion kernels (nsd.h): the norms
 * as Hunyuan3D runs them, the small elementwise steps, the shape decoder's query points, the MoE's routing. Bound by
 * its crate by name (model3d/hunyuan3d/src/ffi.rs). 16-bit buffers are IEEE half; every call is queued on the GPU's
 * queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_MODEL3D_HUNYUAN3D_H
#define NS_MODEL3D_HUNYUAN3D_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* out [M, C] (half when out_half, else float32) = (layernorm(x [M, C]) * w + b) * (1 + scale) + shift, per row;
 * x float32 (x_half 0) or half (1); w, b, scale, shift float32 [C] or NULL (absent: 1, 0, 0, 0). C <= 8192. */
int ns_model3d_hunyuan3d_ln(ns_gpu* g, const void* x, int x_half, int64_t M, int64_t C, float eps, const float* w, const float* b,
                            const float* scale, const float* shift, void* out, int out_half);
/* Per-head norm in place: x half holds M rows of H heads x D features, row m at element m * stride; each head's D
 * features normalized - RMS (rms 1: x / sqrt(mean x^2 + eps) * w) or layer norm (rms 0: (x - mean) / sqrt(var + eps)
 * * w + b) - w, b float32 [D] (b NULL: 0). D a multiple of 32, at most 256. */
int ns_model3d_hunyuan3d_head_norm(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* w, const float* b,
                                   float eps, int rms);
/* out half [n] = silu(x float32 [n]) */
int ns_model3d_hunyuan3d_silu(ns_gpu* g, const float* x, void* out, int64_t n);
/* conversions over n values: float32 -> half, half -> float32 */
int ns_model3d_hunyuan3d_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
int ns_model3d_hunyuan3d_to_float(ns_gpu* g, const void* x, float* out, int64_t n);
/* x float32 [M, C] += y [M, C] (float32, or half when y_half) * gate[c] (gate float32 [C], NULL: 1) */
int ns_model3d_hunyuan3d_add(ns_gpu* g, float* x, const void* y, int y_half, int64_t M, int64_t C, const float* gate);
/* x float32 += a * y float32 over n values (a sampler's step) */
int ns_model3d_hunyuan3d_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);
/* out = u + s * (c - u) over n float32 values (classifier-free guidance) */
int ns_model3d_hunyuan3d_cfg(ns_gpu* g, float* out, const float* c, const float* u, int64_t n, float s);
/* The shape decoder's query embedding of N points: point i is pts[i] (float32 [N, 3]) when pts is given, else grid
 * point p = idx[i] (idx int32 [N]; NULL: p = start + i) of an R1 x R1 x R1 lattice (x slowest, z fastest) at
 * lo + (ix, iy, iz) * step. out half [N, 3 (2 nf + 1)] = (x, sin(x_d f_j) for d, j, cos(x_d f_j) for d, j), f_j = 2^j
 * (times pi when pi). */
int ns_model3d_hunyuan3d_fourier(ns_gpu* g, const float* pts, const int32_t* idx, int64_t start, int64_t N, int64_t R1, float lo,
                                 float step, int nf, int pi, void* out);
/* The MoE's routing: per row of logits float32 [M, E] (E <= 64), softmax, then the k (<= 8) largest: idx int32 [M, k]
 * and their probabilities w float32 [M, k] (not renormalized), largest first. */
int ns_model3d_hunyuan3d_route(ns_gpu* g, const float* logits, int64_t M, int64_t E, int64_t k, int32_t* idx, float* w);
/* out half [n, C] = rows idx[i] (int32 [n]) of x half [M, C] */
int ns_model3d_hunyuan3d_gather(ns_gpu* g, const void* x, const int32_t* idx, int64_t n, int64_t C, void* out);
/* out float32 [M, C]: row idx[i] += w[i] * y[i] (y half [n, C], w float32 [n]); the idx distinct within a call */
int ns_model3d_hunyuan3d_scatter_add(ns_gpu* g, const void* y, const int32_t* idx, const float* w, int64_t n, int64_t C, float* out);

/* The paint model's (a Stable Diffusion 2 UNet and VAE, channels last):
 * group_norm: F frames of N pixels x C channels in G groups, per frame and group (x - mean) / sqrt(var + eps) * w + b,
 *   then SiLU when silu; x, out half (out may be x); stats float32 scratch [F * G * 2]
 * geglu: out half [M, F] = x[:, :F] * gelu(x[:, F:]) (erf), x half [M, 2F]
 * pad2d: out half [N, H + t + b, W + l + r, C] = x half [N, H, W, C] with zeros around
 * concat: out half [M, Ca + Cb] = [a [M, Ca] | b [M, Cb]] (half)
 * up2: out half [N, 2H, 2W, C] = nearest 2x of x half [N, H, W, C]
 * add_half: x half += s * y half over n values (in float32) */
int ns_model3d_hunyuan3d_group_norm(ns_gpu* g, const void* x, int64_t F, int64_t N, int64_t C, int64_t G, const float* w, const float* b,
                                    float eps, int silu, float* stats, void* out);
int ns_model3d_hunyuan3d_geglu(ns_gpu* g, const void* x, int64_t M, int64_t F, void* out);
int ns_model3d_hunyuan3d_pad2d(ns_gpu* g, const void* x, int64_t N, int64_t H, int64_t W, int64_t C, int64_t t, int64_t b, int64_t l,
                               int64_t r, void* out);
int ns_model3d_hunyuan3d_concat(ns_gpu* g, const void* a, int64_t Ca, const void* b, int64_t Cb, int64_t M, void* out);
int ns_model3d_hunyuan3d_up2(ns_gpu* g, const void* x, int64_t N, int64_t H, int64_t W, int64_t C, void* out);
int ns_model3d_hunyuan3d_add_half(ns_gpu* g, void* x, const void* y, int64_t n, float s);

#ifdef __cplusplus
}
#endif
#endif
