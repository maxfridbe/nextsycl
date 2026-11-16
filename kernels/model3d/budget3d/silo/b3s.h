/* b3s.h: the Budget3D-Idol engine's silo (dist/silo/libnextsycl-budget3d.so, opened at load; without it, or with
 * NS_SILO=0, the engine runs on its own kernels in libnextsycl-model3d.so and the shared ones - b3d.h). Kernels tuned for
 * its denoiser and decoders: the products with their epilogues fused (bias, GELU, the gated residual add), the
 * modulated layer norm a sub-group a row. Each op says what it covers (`_supported`); the engine uses its own kernel
 * for everything else. Buffers are device memory of g; dt 0 = float32, 1 = half; every call is queued on g's queue;
 * 0, or -1 and ns_last_error(). */
#ifndef NS_B3S_H
#define NS_B3S_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* The epilogues of ns_b3s_linear */
enum {
    NS_B3S_PLAIN = 0,   /* out (out_dt) = x w^T + bias */
    NS_B3S_GELU = 1,    /* out (out_dt) = gelu_tanh(x w^T + bias), computed before the rounding to out_dt */
    NS_B3S_RESID = 2,   /* out (float32) += (x w^T + bias) * gate (float32 [N]; NULL: a plain add) */
    NS_B3S_SILU = 3,    /* out (out_dt) = silu(x w^T + bias), computed before the rounding to out_dt */
};

/* Whether ns_b3s_linear takes this epilogue and output type */
int ns_b3s_linear_supported(int epi, int out_dt);
/* x [M, K] half . w [N, K] half ^T + bias (float32 [N] or NULL), accumulated in float32, then the epilogue `epi` into
 * out [M, N] (oneDNN, the whole M a primitive: no row chunks) */
int ns_b3s_linear(ns_gpu* g, const void* x, int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out, int out_dt, int epi,
                  const float* gate);

/* out [M, N] float32 = x [M, K] float32 . w [N, K] float32 ^T + bias, multiplied in TF32 on the matrix engines and
 * accumulated in float32 (RePo's products) */
int ns_b3s_linear_tf32(ns_gpu* g, const float* x, int64_t M, int64_t K, const float* w, int64_t N, const float* bias, float* out);

/* Whether ns_b3s_ln_mod takes rows of C at these types (C 1..2048) */
int ns_b3s_ln_mod_supported(int64_t C, int x_dt, int out_dt);
/* b3d.h's ns_model3d_budget3d_ln_mod (the same arithmetic and arguments: row r's scale / shift at (r / group) * stride,
 * group <= 0: the first row), a sub-group of 32 a row held in registers */
int ns_b3s_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias, const float* scale,
                  const float* shift, int64_t stride, int64_t group, void* out, int out_dt);

#ifdef __cplusplus
}
#endif
#endif
