/* silo.h: HunyuanImage 2.1's silo (dist/silo/libnextsycl-hunyuanimage21.so): its DiT's products and its VAE's group norm tuned
 * for its shapes, each with a _supported check; the engine (image/hunyuanimage21/src/ffi.rs) uses them when the check
 * passes and the engine's own kernels (hi21.h, the shared nsd ones) otherwise. */
#ifndef NS_IMAGE_HI21_SILO_H
#define NS_IMAGE_HI21_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* A Q8_0 matrix [N, K] (rows of K / 32 blocks: an fp16 scale, 32 int8 values) split for the products below: w int8
 * [N, K] (row-major, the blocks' values) and s half [K / 32, N] (block b of row n at s[b * N + n]). K a multiple of 32. */
int ns_image_hi21_silo_q8_split(ns_gpu* g, const void* blocks, int64_t N, int64_t K, int8_t* w, void* s);
/* The same matrix requantized for the int8 products (approximate - an opt-in): each row turned by the orthonormal
 * Hadamard matrix in blocks of 256 along K, then w int8 [N, K] over one scale a row, scale float32 [N] (the turned
 * row's largest magnitude / 127). K a multiple of 256. */
int ns_image_hi21_silo_q8_rows(ns_gpu* g, const void* blocks, int64_t N, int64_t K, int8_t* w, float* scale);

/* One product of the DiT, on oneDNN with the weights read as int8 and widened to half in the kernel (no expanded copy
 * of the matrix: the same values the Q8_0 expansion gives - scale x value rounded to half -, accumulated in float32):
 *
 *   for b < B:  out_b [M, N] (rows ldo apart) (post) = x_b [M, kn] (half, rows lda apart) . W[:, k0 : k0 + kn]^T + bias
 *
 * W from ns_image_hi21_silo_q8_split (its K columns; k0 and kn multiples of 32), bias float32 [N] or NULL, x_b at
 * x + b * x_bs halves, out_b at out + b * o_bs values. post: 0 out = the product (out_dt 0 float32 or 1 half),
 * 1 out = gelu_tanh(product) (half), 2 out += gate * product (float32; gate float32 [N]) - the gated residual.
 *
 * With wscale (float32 [N]) the product is int8 x int8 instead (approximate - an opt-in): W from
 * ns_image_hi21_silo_q8_rows (wscale its row scales), each row of x turned the same way and quantized over its
 * largest magnitude into the scratch xq (int8 [B M, kn]) and xs (float32 [B M]), the int32 sums rescaled by both
 * scales; k0 and kn multiples of 256, kn at most 16384, lda and x_bs multiples of 8. */
typedef struct ns_hi21_mm {
    const void* x;
    int64_t lda, x_bs;
    int64_t B, M, K, k0, kn, N;
    const int8_t* w;
    const void* s;
    const float* bias;
    void* out;
    int64_t ldo, o_bs;
    int32_t out_dt, post;
    const float* gate;
    const float* wscale;
    int8_t* xq;
    float* xs;
} ns_hi21_mm;
/* Whether this build can run these products (1) or the engine must expand the matrices itself (0): oneDNN takes int8
 * weights with grouped half scales on this GPU; int8: whether it runs the int8 x int8 ones */
int ns_image_hi21_silo_linear_supported(ns_gpu* g, int int8);
int ns_image_hi21_silo_linear(ns_gpu* g, const ns_hi21_mm* a);

/* ns_image_hi21_group_norm (hi21.h), the same arguments and result, read as a stream of 8-half vectors: C a power of
 * two from 8 to 2048 (the VAE's 128 to 1024), a multiple of G. scratch: ns_image_hi21_silo_gn_scratch(G) floats. */
int ns_image_hi21_silo_group_norm_supported(int64_t C, int64_t G);
int64_t ns_image_hi21_silo_gn_scratch(int64_t G);
int ns_image_hi21_silo_group_norm(ns_gpu* g, const void* x, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias, float eps,
                                  int silu, float* scratch, void* out);

#ifdef __cplusplus
}
#endif
#endif
