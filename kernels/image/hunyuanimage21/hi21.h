/* hi21.h: the HunyuanImage 2.1 engine's own kernels (image/hunyuanimage21), beside the shared diffusion kernels (nsd.h).
 * Bound by its crate by name (image/hunyuanimage21/src/ffi.rs). `dt` names a buffer's type as nsd.h does (0 float32,
 * 1 half); every call is queued on the GPU's queue (the caller syncs) and returns 0, or -1 with the reason in
 * ns_last_error(). */
#ifndef NS_IMAGE_HI21_H
#define NS_IMAGE_HI21_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Q4_0 blocks (an fp16 scale, 16 bytes of 4-bit values: value j < 16 the low nibble of byte j, j >= 16 the high
 * nibble of byte j - 16, each minus 8) -> n values of half; n a multiple of 32 */
int ns_image_hi21_q4_0_half(ns_gpu* g, const void* src, int64_t n, void* out);
/* out [M, C] (out_dt) = layernorm(x [M, C] float32, eps) (* weight + bias, when given) * (1 + scale) + shift (each
 * float32 [C] or NULL); C at most 8192. The DiT's modulated norms, the token refiner's and byT5 mapper's affine ones. */
int ns_image_hi21_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                         const float* scale, const float* shift, void* out, int out_dt);
/* x (dt) in place, n values: kind 0 SiLU, 1 GELU (tanh approximation), 2 GELU (erf) */
int ns_image_hi21_act(ns_gpu* g, void* x, int dt, int64_t n, int kind);
/* out [M, C] (dt) = gelu_tanh(x[:, :C]) * x[:, C:] for x [M, 2C] (dt): T5's gated GELU */
int ns_image_hi21_geglu(ns_gpu* g, const void* x, int dt, int64_t M, int64_t C, void* out);
/* the single block's linear2 input, half: out [M, A + F] = [att [M, A] | gelu_tanh(mlp)], mlp row r at
 * mlp + r * mlp_stride (F values) */
int ns_image_hi21_cat_gelu(ns_gpu* g, const void* att, int64_t M, int64_t A, const void* mlp, int64_t mlp_stride, int64_t F, void* out);
/* Attention for short sequences, float32: query rows q (row stride qs, Hq heads of D), key / value rows k, v (stride
 * kvs, Hkv heads: head h reads h / (Hq / Hkv)), L rows each; scores * scale (+ bias [Hq, L, L] when given), a causal
 * mask when causal; out [L, Hq * D]. One work-group a (head, query), its scores in local memory: L at most 4096,
 * D at most 256. */
int ns_image_hi21_attn_small(ns_gpu* g, const float* q, int64_t qs, const float* k, const float* v, int64_t kvs, int64_t L, int64_t Hq,
                             int64_t Hkv, int64_t D, float scale, const float* bias, int causal, float* out);
/* x float32 += a * y float32 (n values): the sampler's step */
int ns_image_hi21_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);
/* conversions, n values: float32 -> half, half -> float32 (times s) */
int ns_image_hi21_to_half(ns_gpu* g, const float* x, void* out, int64_t n, float s);
int ns_image_hi21_to_float(ns_gpu* g, const void* x, float* out, int64_t n);
/* out [N, C] half += x [N, Ci] half with each output channel c reading input channel c / rep (repeat_interleave) */
int ns_image_hi21_rep_add(ns_gpu* g, void* out, const void* x, int64_t N, int64_t C, int64_t Ci, int64_t rep);
/* The VAE's upsample: out [2H, 2W, C] half = conv [H, W, 4C] rearranged "(r1 r2 c) h w -> c (h r1) (w r2)" plus the
 * input x [H, W, Ci] repeated rep times per channel (repeat_interleave), rearranged the same way */
int ns_image_hi21_up_shuffle(ns_gpu* g, const void* conv, const void* x, int64_t H, int64_t W, int64_t C, int64_t Ci, int64_t rep, void* out);
/* GroupNorm of x [N, C] half (channels last) in G groups, affine weight / bias float32 [C], then SiLU when silu;
 * out [N, C] half. scratch: at least ns_image_hi21_gn_scratch(G) floats. */
int64_t ns_image_hi21_gn_scratch(int64_t G);
int ns_image_hi21_group_norm(ns_gpu* g, const void* x, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias, float eps,
                             int silu, float* scratch, void* out);
/* out [C, R] half = x [R, C] half transposed */
int ns_image_hi21_transpose(ns_gpu* g, const void* x, int64_t R, int64_t C, void* out);
/* out [M, N] half = softmax(x * scale) per row of x [M, N] float32 */
int ns_image_hi21_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out);
/* rgb [n, 3] uint8 from x [n, 3] half in [-1, 1]: (x / 2 + 0.5) clamped to [0, 1], times 255, truncated (the
 * pipeline's .byte()) */
int ns_image_hi21_to_rgb8(ns_gpu* g, const void* x, int64_t n, uint8_t* rgb);

#ifdef __cplusplus
}
#endif
#endif
