/* mo.h: the MOSS-Image engine's own kernels (kernels/image/mossimage), beside the shared diffusion kernels (nsd.h).
 * Bound by its crate by name (image/mossimage/src/ffi.rs). 16-bit buffers are IEEE half (__half); every call is queued
 * on the GPU's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_IMAGE_MO_H
#define NS_IMAGE_MO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* The DiT's gated activation: out[r, c] half = 4 tanh(gate / 4) sigmoid(gate) * 25 tanh(hid / 25), with hid and gate
 * float32 rows `ldh` and `ldg` floats apart (a fused up/gate matrix's two halves, or two buffers), out rows `ldo`
 * halves apart. */
int ns_image_mo_situ(ns_gpu* g, const float* hid, int64_t ldh, const float* gate, int64_t ldg, int64_t M, int64_t C, void* out, int64_t ldo);
/* conversions: float32 <-> half (n values) */
int ns_image_mo_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
int ns_image_mo_to_float(ns_gpu* g, const void* x, float* out, int64_t n);
/* x float32 += a * y float32 (n values): the sampler's Euler step */
int ns_image_mo_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);
/* out = uncond + s * (cond - uncond), float32 (n values): classifier-free guidance */
int ns_image_mo_cfg(ns_gpu* g, float* out, const float* cond, const float* uncond, int64_t n, float s);

/* The text encoder's (Qwen3.5-9B: gated DeltaNet layers and gated full attention), float32 activations; the kernels of
 * kernels/llm/qwen35 and glm5next, after llama.cpp's qwen35 graph, as this engine's own (te.cpp). */
/* eg [T, H, d] = exp(softplus(alpha[t, h] + dt[h]) * a[h]); beta [T, H] = sigmoid(beta) in place */
int ns_image_mo_gdn_gates(ns_gpu* g, const float* alpha, float* beta, const float* dt, const float* a, float* eg, int64_t T, int64_t H, int64_t d);
/* causal depthwise conv (k taps, w [D, k]) over x [T, D] after the k-1 earlier inputs in state [k-1, D] (updated to the
 * last k-1 inputs), then SiLU; out [T, D] */
int ns_image_mo_conv_silu(ns_gpu* g, const float* x, float* state, const float* w, float* out, int64_t T, int64_t D, int k);
/* dst [T, Hv, d] from src rows ss apart: head h of dst is src head h % Hk */
int ns_image_mo_expand(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t T, int64_t Hk, int64_t Hv, int64_t d);
/* x / sqrt(sum x^2 + eps) per row of n, in place */
int ns_image_mo_l2_norm(ns_gpu* g, float* x, int64_t rows, int64_t n, float eps);
/* the delta rule over T rows: q, k, v, eg [T, H, 128], beta [T, H], state S [H, 128, 128] updated; o [T, H, 128] */
int ns_image_mo_kda_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* eg, const float* beta, float* S, float* o,
                         int64_t T, int64_t H);
/* y [T, H, d] = rms_norm(o) * w * silu(z), z rows zs apart */
int ns_image_mo_gdn_out(ns_gpu* g, const float* o, const float* z, int64_t zs, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps);
/* per-head RMS norm (w [D]) then rotate-half RoPE over the first n_rot features at positions p0 + t, theta: src rows ss
 * apart, head h at h * hs; dst [T, H, D] */
int ns_image_mo_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t T, int64_t H, int64_t D, const float* w,
                             float eps, int64_t n_rot, float theta);
/* out [T, H, D] *= sigmoid(gate), the gate at q_full[t * gs + h * 2D + D + d] */
int ns_image_mo_gate_mul(ns_gpu* g, float* out, const float* q_full, int64_t gs, int64_t T, int64_t H, int64_t D);
/* out [T, F] half = silu(gate) * up, float32 [T, F] each */
int ns_image_mo_silu_mul(ns_gpu* g, const float* gate, const float* up, void* out, int64_t n);

/* The VAE decoder's (the FLUX.2 VAE: channels-last half pictures [px, C]). */
/* floats of scratch a group norm of `groups` needs */
int64_t ns_image_mo_group_norm_scratch(int64_t groups);
/* out = group norm of x [px, C] (groups of C / groups channels over all pixels, eps) * gamma[c] + beta[c], then SiLU when
 * `silu`; scratch: ns_image_mo_group_norm_scratch floats */
int ns_image_mo_group_norm(ns_gpu* g, const void* x, int64_t px, int64_t C, int64_t groups, float eps, const float* gamma, const float* beta, int silu, void* out,
                           float* scratch);
/* probs [M, N] half = softmax(x [M, N] float32 * scale) per row */
int ns_image_mo_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out);
/* out [2H, 2W, C] half = nearest 2x of x [H, W, C] half */
int ns_image_mo_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, void* out);
/* rgb [H, W, 3] bytes = round((clamp(x, -1, 1) / 2 + 1 / 2) * 255) of x [H, W, 3] half */
int ns_image_mo_to_rgb8(ns_gpu* g, const void* x, int64_t H, int64_t W, uint8_t* rgb);

#ifdef __cplusplus
}
#endif
#endif
