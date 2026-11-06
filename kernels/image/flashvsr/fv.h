/* fv.h: the FlashVSR engine's own kernels (kernels/image/flashvsr), beside the shared diffusion kernels (nsd.h). Bound
 * by its crate by name (image/flashvsr/src/ffi.rs). 16-bit buffers are IEEE half (sycl::half: nextcuda's numerics, with
 * float32 accumulation and float32 residuals); every call is queued on the GPU's queue (the caller syncs) and returns
 * 0, or -1 with the reason in ns_last_error().
 *
 * The DiT's tokens are kept in window order: a chunk's latent frames in pairs (a "group"), each group's 2 x h x w patch
 * grid cut into blocks of 2 x 8 x 8 = 128 tokens - row-major over the blocks, then (frame, y, x) inside one - which is
 * the order FlashVSR's window partition and block-sparse attention read them in. Every row-wise step reads them in any
 * order, so they stay in it from the patches in to the head out. */
#ifndef NS_IMAGE_FLASHVSR_H
#define NS_IMAGE_FLASHVSR_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- the DiT's tokens (fv.cpp) ----------------------------------------------------------------------------------- */

/* patches: lat float32 [16, f, h8, w8] (f even, h8 and w8 multiples of 16) -> out half [L, 64] in window order, a
 * token's features (c, ky, kx) as the patch embedding's Conv3d weight [1536, 16, 1, 2, 2] flattens them */
int ns_image_flashvsr_patchify(ns_gpu* g, const float* lat, int64_t f, int64_t h8, int64_t w8, void* out);
/* the head's output back to latents: out [16, f, h8, w8] = noise - unpatchify(pred [L, 64] float32, window order; a
 * token's features (ky, kx, c), as Wan's unpatchify reads them) */
int ns_image_flashvsr_unpatchify_sub(ns_gpu* g, const float* pred, const float* noise, int64_t f, int64_t h8, int64_t w8, float* out);
/* x [L, C] float32 (window order) += lq [f * hp * wp, C] float32 (natural (frame, y, x) order) */
int ns_image_flashvsr_add_lq(ns_gpu* g, float* x, const float* lq, int64_t f, int64_t hp, int64_t wp, int64_t C);
/* out [M, C] half = (x - mean) / sqrt(var + eps) * a[c] + b[c] per row of x float32 (a, b float32 [C]; NULL: 1, 0) */
int ns_image_flashvsr_ln(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* a, const float* b, void* out);
/* x [M, C] float32 += y [M, C] float32 * gate[c] (gate NULL: 1) */
int ns_image_flashvsr_gate_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, const float* gate);
/* Each row of x half [M, C] (rows `ld` halves apart) in place: x / rms(x) * w[c] (eps inside the root), then - ts not
 * NULL - its heads of 128 rotated in interleaved pairs (2 i, 2 i + 1) by Wan's 3-D RoPE: Wan's RMSNorm over the whole
 * width, then rope_apply. The rows are tokens in window order of a grid of hp x wp patches (frame t, y, x from the row);
 * pairs 0-21 rotate by the frame's (cos, sin) in ts float32 [frames, 22, 2], pairs 22-63 by the patch's in hw float32
 * [hp wp, 42, 2] - tables of a few KB a chunk instead of one a token. */
int ns_image_flashvsr_rms_rope(ns_gpu* g, void* x, int64_t M, int64_t C, int64_t ld, const float* w, float eps, const float* ts, const float* hw,
                               int64_t hp, int64_t wp);

/* ---- the block-sparse attention's draft (fv.cpp) ----------------------------------------------------------------- */

/* out [n, C] float32: the mean of each block of 128 rows of x half (rows `stride` halves apart) */
int ns_image_flashvsr_block_mean(ns_gpu* g, const void* x, int64_t n, int64_t stride, int64_t C, float* out);
/* The draft attention between blocks: probs [H, nq, nk] = softmax over j of (qm[i] . km[j] per head of 128) / sqrt(128),
 * key blocks outside the query block's local window (lr x lr blocks of the nbh x nbw grid, block index mod nb; not
 * clamped at the edges: FlashVSR's normal_slide) at -inf; a key block with valid[j / nb] == 0 too (a ring slot not yet
 * filled; valid NULL: every one). qm [nq, H * 128], km [nk, H * 128] float32. */
int ns_image_flashvsr_block_probs(ns_gpu* g, const float* qm, const float* km, int64_t nq, int64_t nk, int64_t H, int64_t nbh, int64_t nbw,
                                  int64_t lr, const int32_t* valid, float* probs);
/* thr[s] = the rank-th largest of the n values of segment s of p float32 [S, n] (values >= 0); rank from 1 */
int ns_image_flashvsr_topk_thr(ns_gpu* g, const float* p, int64_t S, int64_t n, int64_t rank, float* thr);
/* The selected key blocks: for each head h and query block i, the j with probs[h, i, j] > thr[h * G + i / nb], in
 * ascending order: lists int32 [H, nq rep, nk] (delta: the first index, then each one's step from the one before - the
 * silo's LUT; else the indices; each block's row `rep` times over: LUT rows of 128 / rep queries), counts int32
 * [H, nq rep] (G = nq / nb query groups) */
int ns_image_flashvsr_block_lists(ns_gpu* g, const float* probs, const float* thr, int64_t H, int64_t nq, int64_t nk, int64_t nb, int delta,
                                  int64_t rep, int32_t* lists, int32_t* counts);
/* out half [nr qrow, H * 128]: head h's rows of LUT row r zeroed where counts[h * nr + r] == 0 (no key block: the
 * reference's kernel gives 0) */
int ns_image_flashvsr_zero_empty(ns_gpu* g, void* out, int64_t H, int64_t nr, int64_t qrow, const int32_t* counts);
/* Block-sparse attention, head size 128, half in and out (the engine's own: the silo's is the fast one): query rows
 * q [Sq, H * 128] (query block = row / 128) over the 128-key blocks its list names (plain indices), block j's keys at
 * k / v + j * 128 rows (rows `kv_rs` halves apart: the ring's keys and values side by side); a row with no block gives
 * 0. out [Sq, H * 128]. */
int ns_image_flashvsr_sparse_attn(ns_gpu* g, const void* q, int64_t Sq, int64_t H, const void* k, const void* v, int64_t kv_rs, const int32_t* lists,
                                  const int32_t* counts, int64_t nk, void* out);

/* ---- the LQ projection (fv.cpp) ---------------------------------------------------------------------------------- */

/* frames half [T, H, W, 3] -> out half [T / ff, H / r, W / r, 3 * ff * r * r], channel ((c * ff + t) * r + y) * r + x */
int ns_image_flashvsr_unshuffle(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t ff, int64_t r, void* out);
/* A causal conv3d's input as one 2-D convolution's: x half [T, H, W, C] (its frames before already in front) -> out
 * half [To, H + 2, W + 2, kt * C], To = (T - kt) / st + 1: output frame t's pixel holds input frames t st .. t st + kt - 1
 * side by side (channel dt * C + c), the border replicated (the reference pads with 'replicate') */
int ns_image_flashvsr_stack_taps(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t kt, int64_t st, void* out);
/* x half [M, C] in place: x / max(|x|, 1e-12) * sqrt(C) * gamma[c], then SiLU (gamma float32 [C]) */
int ns_image_flashvsr_rms_silu(ns_gpu* g, void* x, int64_t M, int64_t C, const float* gamma);

/* ---- the tiny decoder (fv.cpp) ----------------------------------------------------------------------------------- */

/* the decoder's input for one latent frame: out half [h8, w8, ld] (ld >= 784, zeros after 784) = 3 tanh(v / 3) of (the 4
 * cond frames [4, 8 h8, 8 w8, 3] half unshuffled 8 x 8 in space and 4 in time to 768 channels, then the latent's 16
 * channels lat[c, t, y, x] of float32 [16, f, h8, w8]) */
int ns_image_flashvsr_tcd_input(ns_gpu* g, const void* cond, const float* lat, int64_t f, int64_t t, int64_t h8, int64_t w8, int64_t ld, void* out);
/* nearest 2x: x half [H, W, C] (pixels `ldx` halves apart, channels from `c0`) -> rows [r0, r0 + rows) of its 2 H x 2 W
 * upsample, out half [rows, 2 W, C] (a band of it: the decoder's last convolutions go a band of rows at a time) */
int ns_image_flashvsr_up2(ns_gpu* g, const void* x, int64_t H, int64_t W, int64_t C, int64_t ldx, int64_t c0, int64_t r0, int64_t rows, void* out);

/* ---- frames, colour, noise (fv.cpp) ------------------------------------------------------------------------------ */

/* bytes [n] -> half x / 255 * 2 - 1 */
int ns_image_flashvsr_u8_to_half(ns_gpu* g, const uint8_t* x, int64_t n, void* out);
/* float32 <-> half, n values */
int ns_image_flashvsr_to_half(ns_gpu* g, const float* x, int64_t n, void* out);
int ns_image_flashvsr_to_float(ns_gpu* g, const void* x, int64_t n, float* out);
/* The colour fix's parameters (the reference's AdaIN, per channel over the box (x0, y0, w, h) of two [H, W, 3] half
 * pictures): ms float32 [12] = per channel mean_d, std_d, mean_l, std_l of dec * 2 - 1 (the decoder's frame) and of
 * lq (the upscaled input), std = sqrt(var + 1e-5) with the variance biased. scratch: float32 [2 * 256 * 6] (partial
 * sums, no double precision on the card) */
int ns_image_flashvsr_adain(ns_gpu* g, const void* dec, const void* lq, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, float* scratch,
                            float* ms);
/* The colour fix and the bytes: for the box of dec half [H, W, 3] (the decoder's [0, 1]-ish output), v = dec * 2 - 1,
 * y = clamp((v - mean_d) / std_d * std_l + mean_l, -1, 1) per channel (ms float32 [12]; NULL: y = clamp(v, -1, 1)),
 * out bytes [h, w, 3] = trunc(clamp((y + 1) * 127.5, 0, 255)) */
int ns_image_flashvsr_to_u8(ns_gpu* g, const void* dec, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, const float* ms, uint8_t* out);
/* out float32 [n]: standard normal numbers from (seed, the index + offset) - a counter-based generator, so a latent
 * frame's noise is the same whatever came before (and every tile of a clip draws the same) */
int ns_image_flashvsr_randn(ns_gpu* g, float* out, int64_t n, uint64_t seed, uint64_t offset);

/* ---- products on oneDNN (dnn.cpp) -------------------------------------------------------------------------------- */

/* Forget the weights' reorders and the primitives on this card: an engine's weights went (a later buffer at the same
 * address must not find their reorders). Waits for the queue. */
int ns_image_flashvsr_dnn_clear(ns_gpu* g);

/* out [M, N] = epilogue(x [M, K] . w^T): x half (rows `lda` apart), w half [N, K]; then + bias (float32 [N] or NULL),
 * act (0 none, 1 ReLU, 2 GELU tanh), * gate (float32 [N] or NULL), + out's own values when `accum` (a residual: out
 * float32), in that order; out half or float32 (out_f32), rows `ldo` apart. Float32 accumulation. */
int ns_image_flashvsr_gemm(ns_gpu* g, const void* x, int64_t M, int64_t K, int64_t lda, const void* w, int64_t N, const float* bias, int act,
                           const float* gate, int accum, void* out, int64_t ldo, int out_f32);
/* A 2-D convolution of N frames, channels last: x half [N, H, W, Ci] (pixels `ldx` halves apart), w half [Co, Ci, k, k]
 * (reordered once per buffer: keep it alive and unchanged), stride 1, zero padding `pad`; out half [N, Ho, Wo, Co] =
 * act(conv + bias (+ out's own values when `accum`)), act 0 none, 1 ReLU */
int ns_image_flashvsr_conv(ns_gpu* g, const void* x, int64_t N, int64_t H, int64_t W, int64_t Ci, int64_t ldx, const void* w, int64_t Co, int64_t k,
                           int64_t pad, const float* bias, int accum, int act, void* out);

#ifdef __cplusplus
}
#endif
#endif
