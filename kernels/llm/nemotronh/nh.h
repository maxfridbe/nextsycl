/* nh.h: the nemotronh engine's own kernels (Nemotron-H: Mamba-2 mixers, latent MoE, attention without positions), its
 * C ABI, bound by its crate by name (llm/nemotronh/src/ffi.rs). The products from the stored blocks, the RMS norm and
 * the MoE combine are the glm5next engine's generic ones (glm.h). Float32 activations, row-major, all queued on the
 * GPU's queue; 0 or -1 (the reason: ns_last_error). After llama.cpp's src/models/nemotron-h.cpp and mamba-base.cpp. */
#ifndef NS_NH_H
#define NS_NH_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* The Mamba-2 conv: rows x [T] of C channels (xs floats apart) after the k-1 inputs before them (state [k-1][C],
 * oldest first; updated to the last k-1): out [T][C] = silu(sum_j in[t + j][c] * w[c][j] + b[c]). snap (nullable): the
 * state after each row but the last, [T-1][k-1][C] (a verify pass's rollback) */
int ns_nh_conv(ns_gpu* g, const float* x, int64_t xs, float* state, const float* w, const float* b, float* out, int64_t T, int64_t C, int k,
               float* snap);
/* The Mamba-2 scan (ggml's ssm_scan, d_state N, one A a head): rows xbc [T] = x [H][P] | B [G][N] | C [G][N] (xs floats
 * apart), dt [T][H] (dts apart, before its bias); per row and head dt' = softplus(dt + dt_bias), the state S [H][P][N]
 * decayed by exp(dt' A) plus dt' B x, y [T][H][P] = C . S + Dp x. A head h reads group h / (H / G). snap (nullable): S
 * after each row but the last, [T-1][H][P][N] */
int ns_nh_ssm(ns_gpu* g, const float* xbc, int64_t xs, const float* dt, int64_t dts, const float* dt_bias, const float* A, const float* Dp, float* S,
              float* y, int64_t T, int64_t H, int64_t P, int64_t N, int64_t G, float* snap);
/* The same scan for prompt rows in Mamba-2's chunked form (64-row chunks: within one as small products, the state
 * carried over them after; no snapshots), head dim 64 and d_state 128 only; scratch: ns_nh_ssm_chunk_scratch floats */
int64_t ns_nh_ssm_chunk_scratch(int64_t T, int64_t H, int64_t P, int64_t N);
int ns_nh_ssm_chunk(ns_gpu* g, const float* xbc, int64_t xs, const float* dt, int64_t dts, const float* dt_bias, const float* A, const float* Dp, float* S,
                    float* y, int64_t T, int64_t H, int64_t P, int64_t N, int64_t G, float* scratch);
/* The mixer's gate and norm: out [T][C] = rms_norm(y * silu(z)) * w over groups of `group` values (z rows zs apart) */
int ns_nh_gated_norm(ns_gpu* g, const float* y, const float* z, int64_t zs, const float* w, float* out, int64_t T, int64_t C, int64_t group, float eps);
/* x = max(x, 0)^2 */
int ns_nh_relu2(ns_gpu* g, float* x, int64_t n);
/* The cache [Hkv][cap][D] half: rows k [T, Hkv x D] (ks floats apart) and v (vs apart) at positions p0.. */
int ns_nh_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                   void* kc, void* vc);
/* Floats of scratch ns_nh_attn needs for T rows of Hq heads (Hkv key heads) x D at positions p0.. (0: none) */
int64_t ns_nh_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0);
/* Causal attention (no positions; scale 1/sqrt(D)) of rows q [T, Hq x D] (qs floats apart) at positions p0.. against
 * the cache: out [T, Hq x D]. D 128, Hq a multiple of 8 and of Hkv */
int ns_nh_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap,
               int64_t p0, float* out, float* scratch);

/* n values of a Q6_K tensor (n a multiple of 256) expanded to fp16 (the prompt path's products) */
int ns_nh_q6k_f16(ns_gpu* g, const void* src, int64_t n, uint16_t* dst);

#ifdef __cplusplus
}
#endif
#endif
