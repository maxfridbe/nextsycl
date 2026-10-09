/* nsd.h: the shared diffusion kernels' C ABI (kernels/diffusion/nsd.cpp, from H3's h3sycl.h at bd0f481), bound by
 * name by nextsycl-diffusion (crates/diffusion/src/kernels.rs). */
/* h3sycl.h - the C ABI of libnextsycl-image / libnextsycl-video: SYCL kernels for MiniMax H3 on Intel Arc (Xe2).
 *
 * The Rust engine (engine/h3-sys) binds exactly this file. Every data pointer is a USM device pointer of the
 * context's queue. Kernels are asynchronous: a call returns once its work is queued, and the queue is in order, so
 * a later call sees an earlier call's result; nsd_read and nsd_wait are the points that wait.
 *
 * Functions returning int give 0, or -1 with the reason in nsd_last_error() (per thread). Functions returning a
 * pointer give NULL on failure, same.
 */
#ifndef H3SYCL_H
#define H3SYCL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* element types of floating-point tensors */
enum { NSD_F32 = 0, NSD_F16 = 1, NSD_BF16 = 2 };

const char* nsd_last_error(void);

/* ---- context -------------------------------------------------------------------------------------------------- */

/* A context on the first GPU, with its own in-order queue. The Rust engine's way in. */
void* nsd_open(void);
/* The GPUs the runtime sees (ONEAPI_DEVICE_SELECTOR=level_zero:* for every card), and a context on one of them.
 * nsd_gpu_info fills the name, the memory size and the PCI address ("0000:0b:00.0", empty when unknown). */
int nsd_gpu_count(void);
int nsd_gpu_info(int index, char* name, int name_len, uint64_t* mem_bytes, char* pci, int pci_len);
void* nsd_open_gpu(int index);
/* A context around an existing `sycl::queue*` (PyTorch: torch.xpu.current_stream().sycl_queue); the queue handle is
 * copied. The reference pipeline's way in: its tensors' device pointers are then valid here. */
void* nsd_create(void* sycl_queue);
/* Waits for the queue, frees the scratch buffers and everything still allocated with nsd_alloc. */
void nsd_destroy(void* ctx);

const char* nsd_device_name(void* ctx);

/* ---- device memory -------------------------------------------------------------------------------------------- */

/* The xe driver has no out-of-memory error: an over-commit stalls the whole machine. So allocations are counted, and
 * one that would pass the cap (0.94 of the card; NSD_MEM_FRACTION overrides) is refused here, with an error. The
 * kernels' own scratch buffers count against the same cap: a kernel call that needs more fails the same way. */
void* nsd_alloc(void* ctx, uint64_t bytes);
void nsd_free(void* ctx, void* p);              /* waits for the queue first */
uint64_t nsd_mem_used(void* ctx);
uint64_t nsd_mem_cap(void* ctx);
/* What the whole card has free right now, every process counted (0 when the driver cannot tell: it needs
 * ZES_ENABLE_SYSMAN=1, which the image sets). For deciding whether to load beside whatever else holds the card. */
uint64_t nsd_mem_free(void* ctx);

/* Host <-> device copies; both wait for the copy (the host buffer may be released on return). Thread-safe. */
int nsd_write(void* ctx, void* dst, const void* src_host, uint64_t bytes);
int nsd_read(void* ctx, void* dst_host, const void* src, uint64_t bytes);
/* Device to device, queued like a kernel (it does not wait). */
int nsd_copy(void* ctx, void* dst, const void* src, uint64_t bytes);
/* Waits until everything queued has run; reports an asynchronous error if there was one. */
int nsd_wait(void* ctx);

/* Pinned host memory, for uploads that overlap the kernels: the GPU copies from it by DMA at the link's full rate.
 * Not counted against the device cap. nsd_free_host waits for the uploads first. */
void* nsd_alloc_host(void* ctx, uint64_t bytes);
void nsd_free_host(void* ctx, void* p);
/* Pinned host -> device, queued on the context's copy queue beside the kernels (it does not wait): the kernels do not
 * wait for it either, so call nsd_upload_wait before using dst. src must come from nsd_alloc_host (a queue copy with
 * no device end can hang the B70's copy engine); anything else is refused. */
int nsd_upload(void* ctx, void* dst, const void* src_pinned, uint64_t bytes);
int nsd_upload_wait(void* ctx);

/* ---- kernels -------------------------------------------------------------------------------------------------- */

/* A linear layer with int8 weights and activations quantized on the fly (comfy-kitchen's int8_linear):
 *
 *   x_rot = x . H           per group of `group` features, H the normalized regular Hadamard matrix ("ConvRot")
 *   s_r   = max|x_rot[r]| / 127                          per row, at least 1e-30
 *   q     = clamp(round(x_rot / s_r), -128, 127)         int8
 *   acc   = q . W^T                                      int32 - the card's matrix engine
 *   out   = acc * s_r * wscale + bias                    in out_dt
 *
 * x [M, K] in x_dt; w int8 [N, K]; wscale float32, n_wscale = 1 (one scale) or N (one per output); bias float32 [N]
 * or NULL; out [M, N] in out_dt. group = 0: no rotation, else a power of 4, at most 256, dividing K. */
int nsd_int8_linear(void* ctx, const void* x, int x_dt, int64_t M, int64_t K, const int8_t* w, int64_t N,
                    const float* wscale, int64_t n_wscale, const float* bias, void* out, int out_dt, int group);

/* A plain linear layer: out = x . w^T + bias. x [M, K] and w [N, K] in the same type dt (float32, half or
 * bfloat16); bias float32 [N] or NULL; out [M, N] in out_dt. */
/* out [M, N] += x [M, K] . W^T (W [N, K]), all in type dt: a LoRA's second factor, accumulated into the layer's output. */
int nsd_linear_acc(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out);
/* x [M, C] *= s[row] in place (s float32 [M]): folds a per-output scale into a weight matrix. */
int nsd_scale_rows(void* ctx, void* x, int dt, int64_t M, int64_t C, const float* s);
int nsd_linear(void* ctx, const void* x, int dt, int64_t M, int64_t K, const void* w, int64_t N, const float* bias,
               void* out, int out_dt);

/* Row-wise RMS norm, optionally followed by a per-row scale and shift picked from a table:
 *
 *   n   = x[r] / sqrt(mean(x[r]^2) + eps) * weight
 *   out = n * (1 + scale[rows[r]]) + shift[rows[r]]        (or out = n when rows, scale or shift is NULL)
 *
 * x, out [M, C]; weight float32 [C]; rows int32 [M]; scale, shift float32 [R, C]. out may be x. */
/* The text encoder's operations.
 * dequant: n values of a GGUF type into out_dt: 12 = Q4_K, 14 = Q6_K (multiples of 256), 8 = Q8_0 (of 32), 30 = BF16.
 * attention_causal: softmax(q k^T / sqrt(D)) v with a causal mask, Hq query heads over Hkv key/value heads (head h
 *   reads kv head h / (Hq / Hkv)); q rows of stride qs, k/v rows of stride kvs, out [L, Hq * D]; for short L. */
int nsd_dequant(void* ctx, const void* src, int qtype, int64_t n, void* out, int out_dt);
int nsd_attention_causal(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t L, int64_t Hq, int64_t Hkv, int64_t D,
                         int64_t qs, int64_t kvs, void* out);
/* The latent upscaler's operations, on channels-last volumes [T, H, W, C] in a 16-bit type `dt`.
 * conv3d: a k x k x k convolution (k odd, zero padding k / 2), w [Co, Ci, k, k, k] in dt (reordered once per buffer:
 *   keep it alive and unchanged), bias float32 [Co] or NULL; out [T, H, W, Co].
 * group_norm_silu: GroupNorm with G groups over [N, C], affine weight/bias float32 [C], then * (1 + scale) + shift
 *   when scale/shift (float32 [C]) are given, then SiLU.
 * temporal_dwconv: per channel along T, w float32 [C, K], zero padding K / 2; x, out [T, P, C].
 * trilinear: resize to [To, Ho, Wo, C], align_corners=False. */
int nsd_conv3d(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t k,
               const float* bias, void* out);
int nsd_group_norm_silu(void* ctx, const void* x, int dt, int64_t F, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias,
                        float eps, const float* scale, const float* shift, void* out);
/* pad3d: zeros in front along T, the spatial border reflected; conv3d_ex: strided, no padding (out dims (D - k) / s + 1).
 * group_norm_silu takes F frames of N voxels: statistics per frame and group (F = 1: over the whole volume). */
int nsd_pad3d(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t C, int64_t front, int64_t top, int64_t bottom,
              int64_t left, int64_t right, void* out);
/* The pixel upscaler (ESRGAN-type, frames channels-last [N, H, W, C] in 16-bit dt):
 * conv2d: a k x k convolution of N frames (k odd, zero padding k / 2), w [Co, Ci, k, k] in dt (reordered once per
 *   buffer), bias float32 [Co] or NULL; out [N, H, W, Co].
 * prelu: x = x > 0 ? x : alpha[c] x, in place; alpha float32 [C].
 * pixel_shuffle_add: out [N, H r, W r, C] = PixelShuffle(r) of x [N, H, W, C r r] + base [N, H, W, C] repeated r x r
 *   (nearest), as PyTorch's NCHW PixelShuffle orders the channels (c r r + i r + j).
 * resize_area: x [N, Hi, Wi, C] -> out float32 PLANAR [C, N, Ho, Wo], each output pixel the mean of its input area
 *   (PyTorch's mode="area", adaptive average pooling), clamped to [0, 1]. */
int nsd_conv2d(void* ctx, const void* x, int dt, int64_t N, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co, int64_t k,
               const float* bias, void* out);
int nsd_prelu(void* ctx, void* x, int dt, int64_t M, int64_t C, const float* alpha);
int nsd_pixel_shuffle_add(void* ctx, const void* x, int dt, int64_t N, int64_t H, int64_t W, int64_t C, int64_t r,
                          const void* base, void* out);
int nsd_resize_area(void* ctx, const void* x, int dt, int64_t N, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo,
                    float* out);
int nsd_conv3d_ex(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t Ci, const void* w, int64_t Co,
                  int64_t kt, int64_t kh, int64_t kw, int64_t st, int64_t sh, int64_t sw, const float* bias, void* out);
int nsd_temporal_dwconv(void* ctx, const void* x, int dt, int64_t T, int64_t P, int64_t C, const float* w, int64_t K,
                        const float* bias, void* out);
int nsd_trilinear(void* ctx, const void* x, int dt, int64_t T, int64_t H, int64_t W, int64_t C, int64_t To, int64_t Ho, int64_t Wo,
                  void* out);
/* The audio decoder's 1-D operations, float32, signals [B, C, L] row-major.
 * conv1d: w [Co, Ci, K], zero padding `pad` both sides; out [B, Co, Lo], Lo = (L + 2 pad - dil (K - 1) - 1) / stride + 1.
 * conv_transpose1d: w [Ci, Co, K]; out [B, Co, Lo], Lo = (L - 1) stride - 2 pad + K.
 * aa_snake: BigVGAN's anti-aliased SnakeBeta (2x up, x + sin^2(e^a x) / e^b, 2x down), 12-tap filters; out [B, C, L].
 * scale: x *= s over n values. */
int nsd_conv1d(void* ctx, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
               const float* bias, int64_t stride, int64_t dil, int64_t pad, float* out, int64_t Lo);
int nsd_conv_transpose1d(void* ctx, const float* x, int64_t B, int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
                         const float* bias, int64_t stride, int64_t pad, float* out, int64_t Lo);
int nsd_aa_snake(void* ctx, const float* x, int64_t B, int64_t C, int64_t L, const float* log_alpha, const float* log_beta,
                 const float* up, const float* down, float* out);
int nsd_scale(void* ctx, float* x, int64_t n, float s);
/* snake: x + sin^2(alpha x) / alpha per channel of [B, C, L] (the audio encoder's activation). */
int nsd_snake(void* ctx, const float* x, int64_t B, int64_t C, int64_t L, const float* alpha, float* out);
/* Layer norm of x [M, C] per row: (x - mean) / sqrt(var + eps) * weight + bias (weight, bias float32 [C] or NULL). */
int nsd_layer_norm(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, const float* bias, float eps,
                   void* out, int out_dt);
int nsd_rms_norm_mod(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, const float* weight, float eps,
                     const int32_t* rows, const float* scale, const float* shift, void* out, int out_dt);

/* Per-head RMS norm, then the rotary position rotation, in place. x holds M token rows of H heads x D features;
 * row m starts at element m * stride (stride >= H * D: the rows may sit inside a wider buffer, as q and k do inside
 * the qkv linear's output). Each head's D features are normalized (weight float32 [D]); then feature pairs
 * (i, rot_dim/2 + i), i < rot_dim/2, are rotated by the token's angle:
 *
 *   a' = a * cos - b * sin,   b' = b * cos + a * sin
 *
 * cs float32 [M, rot_dim/2, 2] holds (cos, sin) per token and pair. Features from rot_dim on are only normalized. */
int nsd_rms_rope(void* ctx, void* x, int x_dt, int64_t M, int64_t H, int64_t D, int64_t stride, const float* weight,
                 float eps, const float* cs, int rot_dim);

/* The gated activation between an MLP's two linears: out[r, i] = silu(x[r, i]) * x[r, C + i].
 * x [M, 2 * C], out [M, C]. */
int nsd_swiglu(void* ctx, const void* x, int x_dt, int64_t M, int64_t C, void* out, int out_dt);

/* x[r] += other[r] * gate[rows[r]], in place. x, other [M, C]; rows int32 [M]; gate float32 [R, C].
 * gate NULL: a plain add. */
int nsd_gate_add(void* ctx, void* x, int x_dt, int64_t M, int64_t C, const void* other, int other_dt,
                 const int32_t* rows, const float* gate);

/* Attention: out = softmax(q . k^T / sqrt(D)) . v, per head. q, k, v: S token rows of H heads x D features, row s at
 * element s * stride (as in nsd_rms_rope), in type dt; out [S, H * D] in out_dt. The S x S scores are never
 * held whole: the work is done a block of query rows at a time, sized so that a block's score table is at most
 * 1.5 GiB (NSD_ATTN_TABLE_MB in the environment overrides; with oneDNN's fused kernel no table is written at all, and
 * the bound is what is at stake if oneDNN silently falls back - see h3sycl.cpp). */
int nsd_attention(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t S, int64_t H, int64_t D,
                  int64_t stride, void* out, int out_dt);
/* B independent sequences of S tokens each in one call (the video decoder's tiles): sequence b's rows start at row
 * b * S of q, k and v (stride as above) and of out [B * S, H * D]. One oneDNN fused call for the whole batch; with
 * 16-bit half in and out it reads q, k, v and writes out in place through strides (no copies). Otherwise, and when
 * oneDNN refuses, the sequences one at a time through nsd_attention. */
int nsd_attention_batch(void* ctx, const void* q, const void* k, const void* v, int dt, int64_t B, int64_t S, int64_t H,
                        int64_t D, int64_t stride, void* out, int out_dt);
/* nextsycl: Sq query rows (stride qs) over Skv key / value rows (stride kvs) - a cached prefix and the queries' own
 * block, no mask; out [Sq, H * D]. oneDNN's fused kernel only (an error when it refuses). */
int nsd_attention_qk(void* ctx, const void* q, int64_t Sq, int64_t qs, const void* k, const void* v, int64_t Skv,
                     int64_t kvs, int dt, int64_t H, int64_t D, void* out, int out_dt);

#ifdef __cplusplus
}
#endif
#endif
