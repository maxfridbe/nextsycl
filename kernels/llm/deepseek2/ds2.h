/* ds2.h: the deepseek2 engine's own kernels (DeepSeek-V3 / R1: MLA with a decoupled RoPE part, 256 experts). Its
 * generic steps (norms, GEMMs, the masked softmax, gather / scatter, the dense types' decode products) are the
 * glm5next engine's (glm.h); these are what this model needs and no other engine has: IQ1_S (the 1.58-bit experts
 * of unsloth's dynamic quants - no shared kernel takes the type), the products of a layer's experts in one launch,
 * YaRN's RoPE, the latent cache's rows, and a host-side wait for a ticket (the disk tier reads into pinned slots the
 * copy engine may still be reading). */
#ifndef NS_DS2_H
#define NS_DS2_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- ds2.cpp */

/* YaRN RoPE (ggml's rope_yarn, NORM: pairs of adjacent values) in place on x: T rows `row` floats apart, H heads
 * `head` floats apart, the rotated part `n_dims` values from float `off` of each head; row t at position pos0 + t.
 * `out` (nullable): the rotated part also written there, rows `out_row` / heads `out_head` floats apart (from its
 * front). corr0 / corr1 from ggml_rope_yarn_corr_dims; mscale multiplies cos and sin (YaRN's magnitude after the
 * attention factor) */
int ns_ds2_rope(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t row, int64_t head, int64_t off, int64_t n_dims, int64_t pos0,
                float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1, float* out, int64_t out_row,
                int64_t out_head);
/* The latent cache's rows pos0 .. pos0 + T - 1 (fp16, [L + R] each): kv [T, L + R] float32 (kv_a's output) -> the
 * first L RMS-normed with w (eps), the last R rotated (as ns_ds2_rope at the row's position) */
int ns_ds2_kv_store(ns_gpu* g, const float* kv, const float* w, float eps, uint16_t* cache, int64_t T, int64_t L, int64_t R, int64_t pos0,
                    float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1);
/* out [rows, f] = silu(gu[r][0, f)) * gu[r][f, 2f) (a layer's experts' gate | up rows) */
int ns_ds2_swiglu_rows(ns_gpu* g, const float* gu, float* out, int64_t rows, int64_t f);
/* n values (a multiple of 256) of type `type` (IQ1_S) to float32 / fp16 */
int ns_ds2_dequant_supported(int type);
int ns_ds2_dequant(ns_gpu* g, int type, const void* src, int64_t n, float* dst);
int ns_ds2_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst);
/* n bytes (a multiple of 16) from src to dst by a kernel on the GPU's compute queue (not a copy engine: a copy down to
 * pinned memory this way runs beside a copy engine's copy up), after ticket `after` (< 0: none); its ticket */
int ns_ds2_kcopy(ns_gpu* g, void* dst, const void* src, size_t n, int64_t after, int64_t* ticket);
/* n floats of x set to v by a kernel (memset goes to a copy engine, behind the store's copies) */
int ns_ds2_fill(ns_gpu* g, float* x, float v, int64_t n);
/* The host waits for ticket t (ns_mark, ns_stream_copy); `done` asks without waiting (1 when it has run) */
int ns_ds2_ticket_wait(ns_gpu* g, int64_t t);
int ns_ds2_ticket_done(ns_gpu* g, int64_t t);

/* ---- mmvq.cpp: decode products from the stored blocks against the split Q8 layout (the qwen35 silo's: a column's
 * quants [ncols][n_in] int8, then the scales [ncols][n_in / 32] half, then each 16 quants' sum [ncols][n_in / 16]
 * int16) */
size_t ns_ds2_q8_bytes(int64_t n_in, int64_t ncols);
/* x [ncols, n_in] float32 -> the split layout */
int ns_ds2_quantize_q8(ns_gpu* g, const float* x, void* xs, int64_t n_in, int64_t ncols);
/* IQ1_S, IQ2_XXS at a width of whole 256-blocks */
int ns_ds2_mmvq_supported(int type, int64_t n_in);
/* y [ncols, n_out] = W [n_out, n_in] . x (ncols 1..8) */
int ns_ds2_mmvq(ns_gpu* g, int type, const void* w, const void* xs, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
/* A layer's experts at once: entry e multiplies the matrix [n_out, n_in] at ptrs[e] (u64 device addresses) by
 * column cols[e] of x (ncols columns in the split layout): y [entries, n_out] */
int ns_ds2_mmvq_multi(ns_gpu* g, int type, const uint64_t* ptrs, const int32_t* cols, int64_t entries, const void* xs, int64_t ncols, float* y,
                      int64_t n_in, int64_t n_out);

#ifdef __cplusplus
}
#endif
#endif
