/* yue.h: the YuE2 engine's own kernels (audio/yue2), beside the shared diffusion kernels (nsd.h). Bound by its crate by
 * name (audio/yue2/src/ffi.rs). 16-bit buffers are IEEE half; every call is queued on the GPU's queue (the caller
 * syncs) and returns 0, or -1 with the reason in ns_last_error(). */
#ifndef NS_AUDIO_YUE2_H
#define NS_AUDIO_YUE2_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* The decode's product read straight from a Q8_0 matrix (ggml's blocks of 32: a half scale, 32 int8):
 *   out[m, n] (+)= sum_k x[m, k] w[n, k]        m < M <= 8, w [N, K] Q8_0 (K a multiple of 32), x float32 [M, K]
 * out float32 rows `ldo` apart; acc 1: added to what out holds (the residual's own add) */
int ns_audio_yue2_gemv_q8(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* w, int64_t N, float* out, int64_t ldo, int acc);

/* Rows of a Q8_0 table [V, K] as float32: out[i, :] = table[ids[i], :] (ids int32 on the GPU, n of them) */
int ns_audio_yue2_embed_q8(ns_gpu* g, const void* table, int64_t K, const int32_t* ids, int64_t n, float* out);

/* The attention's inputs from the fused projection's rows qkv float32 [T, 4096] (q: 16 heads x 128, k and v: 8 each):
 * q and k RMS-normed per head (weights qn, kn float32 [128], eps), then rotate-half RoPE (pairs i, 64 + i; cs float32
 * [positions, 64, 2] the (cos, sin) of upstream's float32 angles, made on the host); q to q_out half [T, 2048], k and v
 * to the cache half [cap, 8 x 128] at rows row0 + t */
int ns_audio_yue2_qkv(ns_gpu* g, const float* qkv, int64_t T, const float* qn, const float* kn, float eps, const float* cs, int64_t pos0, void* q_out,
                      void* kc, void* vc, int64_t row0);

/* One query row's attention against the cache: q half [16 x 128], keys / values half [n, 8 x 128] (query head h reads
 * key head h / 2), scale 1 / sqrt(128); out float32 [16 x 128]. part: float32 scratch of ns_audio_yue2_attn_scratch(n)
 * floats (none when that is 0) */
int64_t ns_audio_yue2_attn_scratch(int64_t n);
int ns_audio_yue2_attn(ns_gpu* g, const void* q, const void* kc, const void* vc, int64_t n, float* out, float* part);

/* The flow's query rows for the shared attention, whose query and key heads are one to one: q half [S, 16 x 128] <->
 * half [2 S, 8 x 128] - query heads 2 h + g become head h of row 2 r + g, the row reading the key heads straight
 * (to_grouped 1: that way; 0: back) */
int ns_audio_yue2_heads(ns_gpu* g, const void* in, void* out, int64_t S, int to_grouped);

/* x[r, c] += a[c] + rows[r, c] (x, rows float32 [R, C]; a float32 [C]): the NAR rows' time embedding and positions */
int ns_audio_yue2_add_vec_rows(ns_gpu* g, float* x, int64_t R, int64_t C, const float* a, const float* rows);

/* out = a + s b (n floats; out may be a): the midpoint solver's steps */
int ns_audio_yue2_axpy(ns_gpu* g, float* out, const float* a, const float* b, int64_t n, float s);

/* SheetSage2's encoder (MERT-v2's ConvNeXt subsampler and conformer blocks; kernels/audio/yue2/ss2.cpp): float32 rows
 * [T, C], channels-last.
 * dwconv: a depthwise convolution along T, zero padded K / 2 either side: out[t, c] = bias[c] + sum_j w[c, j] x[t + j - K/2, c]
 *   (w float32 [C, K], torch's [C, 1, K]; bias float32 [C] or NULL; K odd).
 * grn: ConvNeXt-V2's global response norm over the whole sequence, out (half) = gamma (x n) + beta + x, n[c] = |x[:, c]|
 *   / (mean over c of |x[:, c]| + 1e-6) (the L2 norm along T); part: float32 scratch of C + 1.
 * to_half: n floats to half (nsd_dequant has no float32 input here).
 * glu: out[t, c] = x[t, c] sigmoid(x[t, C + c]) for x [T, 2C] (the conformer's convolution module). */
int ns_audio_yue2_dwconv(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out);
int ns_audio_yue2_grn(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* gamma, const float* beta, void* out, float* part);
int ns_audio_yue2_to_half(ns_gpu* g, const float* x, int64_t n, void* out);
int ns_audio_yue2_glu(ns_gpu* g, const float* x, int64_t T, int64_t C, float* out);

#ifdef __cplusplus
}
#endif
#endif
