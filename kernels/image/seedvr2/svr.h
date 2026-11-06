/* svr.h: the SeedVR2 engine's own kernels (kernels/image/seedvr2), beside the shared diffusion kernels (nsd.h). Bound
 * by its crate by name (image/seedvr2/src/ffi.rs). 16-bit buffers are bfloat16 (the reference runs in bfloat16:
 * autocast for the DiT, the VAE's dtype); every call is queued on the GPU's queue (the caller syncs) and returns 0, or
 * -1 with the reason in ns_last_error(). Volumes are channels-last [T, H, W, C]. */
#ifndef NS_IMAGE_SEEDVR2_H
#define NS_IMAGE_SEEDVR2_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- the VAE (conv.cpp: oneDNN; svr.cpp) ------------------------------------------------------------------------- */

/* A 3-D convolution of x [T, H, W, Ci] into out [To, Ho, Wo, Co], no padding along T (a causal convolution's frames
 * before the first are the caller's: its memory, or the first frame repeated), zeros ph0 / pw0 above and left and
 * ph1 / pw1 below and right; w [Co, Ci, kt, kh, kw] bfloat16 (reordered once per buffer: keep it alive and unchanged),
 * bias float32 [Co] or NULL. To = (T - kt) / st + 1, Ho = (H + ph0 + ph1 - kh) / sh + 1, likewise Wo. */
int ns_image_seedvr2_conv3d(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t kt,
                            int64_t kh, int64_t kw, int64_t st, int64_t sh, int64_t sw, int64_t ph0, int64_t ph1, int64_t pw0, int64_t pw1,
                            const float* bias, void* out);
/* GroupNorm of each of F frames of N voxels [F, N, C] (statistics per frame and group, as the VAE's causal_norm_wrapper
 * folds time into the batch), G groups, affine weight / bias float32 [C], then SiLU when silu. out may be x. */
int ns_image_seedvr2_group_norm(ns_gpu* g, const void* x, int64_t F, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias,
                                float eps, int silu, void* out);
/* The upsampler's shuffle (rearrange "b (x y z c) f h w -> b c (f z) (h x) (w y)"): x [T, H, W, (sx sy sz C)] into
 * out [T sz, H sx, W sy, C]; drop_second: the second output frame left out (remove_head: the first latent frame's
 * duplicate), so out has T sz - 1 frames. */
int ns_image_seedvr2_shuffle_up(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t sx, int64_t sy, int64_t sz,
                                int drop_second, void* out);
/* x += y over n bfloat16 values (the resnets' and the attention's residual) */
int ns_image_seedvr2_add(ns_gpu* g, void* x, const void* y, int64_t n);
/* softmax(scale * s) of each of R rows of L float32 scores, into bfloat16 probabilities */
int ns_image_seedvr2_softmax_rows(ns_gpu* g, const float* s, int64_t R, int64_t L, float scale, void* out);

/* ---- the frames ------------------------------------------------------------------------------------------------- */

/* Frames of RGB bytes [N, Hi, Wi, 3] resized to [N, Ho, Wo, 3] bfloat16 in [-1, 1]: torchvision's resize of the
 * float frames / 255 (bicubic, antialiased: PyTorch's separable filter, a = -0.5, half-pixel centres, edge weights
 * renormalised), clamped to [0, 1], then * 2 - 1. A centre crop of (oy, ox) after it (DivisibleCrop): out pixel (y, x)
 * is resized pixel (y + oy, x + ox) of the Hr x Wr resize. */
int ns_image_seedvr2_resize(ns_gpu* g, const uint8_t* rgb, int64_t N, int64_t Hi, int64_t Wi, int64_t Hr, int64_t Wr, int64_t oy, int64_t ox,
                            int64_t Ho, int64_t Wo, void* out);
/* The wavelet colour fix of N frames [N, H, W, 3] bfloat16: out = clamp(content + blur5(style - content), -1, 1), blur5
 * the five dilated 3 x 3 blurs (dilation 1, 2, 4, 8, 16, each at most min(H, W) / 8; borders replicated) - numz's
 * wavelet_reconstruction, which is linear in its two pictures. tmp: float32 scratch of 2 N H W 3. */
int ns_image_seedvr2_colorfix(ns_gpu* g, const void* content, const void* style, int64_t N, int64_t H, int64_t W, float* tmp, void* out);
/* bfloat16 [n] in [-1, 1] to bytes: round(clamp(x, -1, 1) * 127.5 + 127.5) */
int ns_image_seedvr2_to_rgb8(ns_gpu* g, const void* x, int64_t n, uint8_t* out);

/* ---- the DiT ---------------------------------------------------------------------------------------------------- */

/* conversions: float32 <-> bfloat16 (n values) */
int ns_image_seedvr2_to_bf16(ns_gpu* g, const float* x, void* out, int64_t n);
int ns_image_seedvr2_to_f32(ns_gpu* g, const void* x, float* out, int64_t n);
/* out bfloat16 [M, C] = x / rms(x) (* weight when given) * scale + shift: x float32 [M, C], weight / scale / shift
 * float32 [C] (the same for every row: one timestep) */
int ns_image_seedvr2_norm_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* weight, float eps, const float* scale,
                              const float* shift, void* out);
/* x float32 [M, C] += y bfloat16 [M, C] * gate float32 [C] */
int ns_image_seedvr2_gate_add(ns_gpu* g, float* x, const void* y, int64_t M, int64_t C, const float* gate);
/* (half: the qkv rows and the attention's output are IEEE half - the shared attention's fast path - and gather's
 * half_to_bf16 turns them back as it gathers; text_mean reads half.)
 * The q and k heads of the packed qkv rows (bfloat16, row stride `stride`; q at column 0, k at column H D): each head's
 * RMS norm (weights wq / wk float32 [D]), rounded to bfloat16 as the reference's norm leaves it, then the rotary
 * rotation of its first 2 P features in interleaved pairs (2 j, 2 j + 1) - rotary_embedding_torch's rotate_half - by
 * cs float32 [M, P, 2] (cos, sin), cs NULL: no rotation. In place. */
int ns_image_seedvr2_qk_norm_rope(ns_gpu* g, void* qkv, int64_t M, int64_t H, int64_t D, int64_t stride, const float* wq, const float* wk,
                                  float eps, const float* cs, int64_t P, int half);
/* out[r] = row idx[r] of a (idx < na) or of b (idx - na): rows of `cols` bfloat16, sources' rows `lda` / `ldb` apart,
 * out's `cols` apart */
int ns_image_seedvr2_gather(ns_gpu* g, const void* a, int64_t na, int64_t lda, const void* b, int64_t ldb, const int32_t* idx, int64_t rows,
                            int64_t cols, void* out, int half_to_bf16);
/* The text rows of the windows coalesced: out[l] = the mean over the n windows of att[base[w] + l] (bfloat16 rows of
 * C), l < L - the reference's unconcat_coalesce */
int ns_image_seedvr2_text_mean(ns_gpu* g, const void* att, const int32_t* base, int64_t n, int64_t L, int64_t C, void* out, int half);

#ifdef __cplusplus
}
#endif
#endif
