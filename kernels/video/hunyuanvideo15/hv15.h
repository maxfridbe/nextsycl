/* hv15.h: the HunyuanVideo-1.5 engine's own kernels (video/hunyuanvideo15), beside the shared diffusion kernels (nsd.h).
 * Bound by its crate by name (video/hunyuanvideo15/src/ffi.rs). `dt` names a buffer's type as nsd.h does (0 float32,
 * 1 half); every call is queued on the GPU's queue (the caller syncs) and returns 0, or -1 with the reason in
 * ns_last_error(). The text encoders' kernels are HunyuanImage 2.1's (kernels/image/hunyuanimage21/hi21.h), the same
 * code under this engine's names: each kind's library carries its own engines' kernels. */
#ifndef NS_VIDEO_HV15_H
#define NS_VIDEO_HV15_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- the text encoders' (as hi21.h) ---------------------------------------------------------------------------- */
/* Q4_0 blocks -> n values of half (n a multiple of 32) */
int ns_video_hv15_q4_0_half(ns_gpu* g, const void* src, int64_t n, void* out);
/* out [M, C] (out_dt) = layernorm(x [M, C] float32, eps) (* weight + bias) * (1 + scale) + shift; C at most 8192 */
int ns_video_hv15_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                         const float* scale, const float* shift, void* out, int out_dt);
/* x (dt) in place, n values: kind 0 SiLU, 1 GELU (tanh), 2 GELU (erf) */
int ns_video_hv15_act(ns_gpu* g, void* x, int dt, int64_t n, int kind);
/* out [M, C] (dt) = gelu_tanh(x[:, :C]) * x[:, C:] */
int ns_video_hv15_geglu(ns_gpu* g, const void* x, int dt, int64_t M, int64_t C, void* out);
/* attention for short float32 sequences (L <= 4096, D <= 256): hi21.h's ns_image_hi21_attn_small */
int ns_video_hv15_attn_small(ns_gpu* g, const float* q, int64_t qs, const float* k, const float* v, int64_t kvs, int64_t L, int64_t Hq,
                             int64_t Hkv, int64_t D, float scale, const float* bias, int causal, float* out);
/* out [M, N] float32 = x [M, K] half . w^T (w [N, K] half, rows ldw apart; 0: K) + bias (float32 [N] or NULL), accumulated
 * and written in float32 (oneDNN); out rows ldo apart (0: N) */
int ns_video_hv15_linear_f32(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t ldw, int64_t N, const float* bias, float* out,
                             int64_t ldo);
/* out [M, N] float32 = x [M, K] float32 . w^T (w [N, K] float32) + bias (float32 [N] or NULL), deterministic (oneDNN) */
int ns_video_hv15_linear32(ns_gpu* g, const float* x, int64_t M, int64_t K, const float* w, int64_t N, const float* bias, float* out);
/* x float32 += a * y float32, n values */
int ns_video_hv15_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a);
/* float32 -> half (times s), half -> float32 */
int ns_video_hv15_to_half(ns_gpu* g, const float* x, void* out, int64_t n, float s);
int ns_video_hv15_to_float(ns_gpu* g, const void* x, float* out, int64_t n);

/* ---- the DiT's ----------------------------------------------------------------------------------------------- */
/* Q8_0 blocks (an fp16 scale, 32 int8) of n values -> float32 (n a multiple of 32): the modulations' matrices */
int ns_video_hv15_q8_0_float(ns_gpu* g, const void* src, int64_t n, float* out);

/* ---- the VAE's: channels-last volumes [T, H, W, C] of half ----------------------------------------------------- */
/* out [T + 2, H + 2, W + 2, C] = x padded for a causal 3x3x3 convolution: two frames before the first - copies of
 * it (CausalConv3d's replicate padding), or with prev ([2, H + 2, W + 2, C], already padded: the last two frames of the
 * previous chunk's padded input) those, a stream carried on - and one pixel around each frame (the border repeated) */
int ns_video_hv15_pad_causal(ns_gpu* g, const void* x, const void* prev, int64_t T, int64_t H, int64_t W, int64_t C, void* out);
/* out [N, C] = x / max(|x|, 1e-12) * sqrt(C) * gamma (F.normalize over the channels, RMS_norm), then SiLU when silu;
 * C at most 1024 */
int ns_video_hv15_rms_channels(ns_gpu* g, const void* x, int64_t N, int64_t C, const float* gamma, int silu, void* out);
/* out [N, C] half += x [N, Ci] half with output channel c reading input channel c / rep (repeat_interleave) */
int ns_video_hv15_rep_add(ns_gpu* g, void* out, const void* x, int64_t N, int64_t C, int64_t Ci, int64_t rep);
/* x [n] half += y [n] half */
int ns_video_hv15_add(ns_gpu* g, void* x, const void* y, int64_t n);
/* The decoder's Upsample: conv [T, H, W, Cc] (its convolution's output) and x [T, H, W, Ci] (the block's input) ->
 * out [To, 2H, 2W, C].
 *   temporal (Cc = 8 C): frame 0 -> one frame, "(r2 r3 c) f h w -> c f (h r2) (w r3)" of conv's first 4 x 2C channels
 *     keeping c < C, plus x's "(r2 r3 c)" (c of Ci / 4) repeated (8 C / Ci) / 2 times per channel; frame t >= 1 -> two
 *     frames, "(r1 r2 r3 c)" of conv (c of C) plus x's "(r1 r2 r3 c)" (c of Ci / 8) repeated 8 C / Ci times. To = 2T - 1.
 *   spatial (Cc = 4 C): "(r2 r3 c)" of conv plus x repeated 4 C / Ci times per channel, then rearranged the same way.
 *     To = T. */
int ns_video_hv15_upsample(ns_gpu* g, const void* conv, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t Ci, int temporal,
                           void* out);
/* ns_video_hv15_upsample with first = 0: a chunk after the stream's first, whose frame 0 is not the video's: every frame
 * becomes two (To = 2T) */
int ns_video_hv15_upsample_ex(ns_gpu* g, const void* conv, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t Ci, int temporal,
                              int first, void* out);
/* The encoder's Downsample of one frame (I2V's picture): conv [H, W, Cc] (its convolution's output, Cc = Cout / 8 in
 * time and space, Cout / 4 in space) and x [H, W, Ci] -> out [H / 2, W / 2, Cout]: conv's "(r2 r3 c)" space-to-depth
 * (in time: twice over, the first frame's channels repeated), plus x's "(r2 r3 c)" averaged over groups of 4 Ci / Cout
 * channels */
int ns_video_hv15_downsample(ns_gpu* g, const void* conv, const void* x, int64_t H, int64_t W, int64_t Cc, int64_t Ci, int64_t Cout, void* out);
/* out [N, C] half += the mean of x [N, C r] half over each run of r channels ("(c r)": channel c r + j): the encoder's
 * last shortcut */
int ns_video_hv15_group_mean_add(ns_gpu* g, void* out, const void* x, int64_t N, int64_t C, int64_t r);
/* out [N, C] half = x [N, C] half transposed to [C, N] */
int ns_video_hv15_transpose(ns_gpu* g, const void* x, int64_t R, int64_t C, void* out);
/* The VAE's frame-causal softmax: x float32 [M, Nk] holds scores of query rows row0 .. row0 + M against keys 0 .. Nk; a
 * query of frame f = (row0 + i) / hw sees keys j < (f + 1) hw. out [M, Nk] half = softmax(x * scale) over those, 0
 * beyond */
int ns_video_hv15_softmax_causal(ns_gpu* g, const float* x, int64_t M, int64_t Nk, int64_t row0, int64_t hw, float scale, void* out);

#ifdef __cplusplus
}
#endif
#endif
