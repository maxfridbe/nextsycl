/* q35.h: the qwen35 engine's own kernels (dense Qwen3.5 / Qwen3.8: gated DeltaNet and gated full attention), its C
 * ABI, bound by its crate by name (llm/qwen35/src/ffi.rs). The products, norms, the conv + SiLU, the L2 norm and the
 * delta-rule scan are the glm5next engine's generic ones (glm.h). Float32 activations, row-major, all queued on the
 * GPU's queue; 0 or -1 (the reason: ns_last_error). */
#ifndef NS_Q35_H
#define NS_Q35_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Per (row t, head h): x = src[t * ss + h * hs .. + D] RMS-normed (weight w [D], eps), then rotate-half rotary positions
 * on its first n_rot features (pairs i, i + n_rot/2; angle (p0 + t) * theta^(-2i / n_rot)); into dst [t * ds + h * D] */
int ns_q35_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                        const float* w, float eps, int64_t n_rot, float theta, int64_t p0);
/* The cache [Hkv][cap][D] half: rows k [T, Hkv x D] (ks floats apart) and v (vs apart) at positions p0.. */
int ns_q35_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                    void* kc, void* vc);
/* Floats of scratch ns_q35_attn needs for T rows of Hq heads x D at positions p0.. (0: none) */
int64_t ns_q35_attn_scratch(int64_t T, int64_t Hq, int64_t D, int64_t p0);
/* Causal attention of rows q [T, Hq x D] (qs floats apart) at positions p0.. against the cache (D 256, Hq / Hkv 6 or
 * 1..8): out [T, Hq x D] */
int ns_q35_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                int64_t cap, int64_t p0, float* out, float* scratch);
/* The attention's output gate: out [T, H x D] *= sigmoid(gate), the gate at q_full[t * gs + h * 2D + D + d] */
int ns_q35_gate_mul(ns_gpu* g, float* out, const float* q_full, int64_t gs, int64_t T, int64_t H, int64_t D);
/* DeltaNet's per-head gates: eg [T, H, d] = exp(softplus(alpha + dt) * a) (each head's d copies), beta = sigmoid(beta);
 * alpha / beta [T, H] (as rows as apart) */
int ns_q35_gdn_gates(ns_gpu* g, const float* alpha, float* beta, const float* dt, const float* a, float* eg, int64_t T, int64_t H, int64_t d);
/* q / k heads to the value heads: dst [T, Hv, d] = src[t * ss + (h % Hk) * d ..] */
int ns_q35_expand(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t T, int64_t Hk, int64_t Hv, int64_t d);
/* DeltaNet's output: y [T, H, d] = rms_norm(o) * w * silu(z), per head; z rows zs floats apart */
int ns_q35_gdn_out(ns_gpu* g, const float* o, const float* z, int64_t zs, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps);
/* The index of each row's largest value (the lowest on a tie): x [rows, n] -> out [rows] */
int ns_q35_argmax(ns_gpu* g, const float* x, int64_t rows, int64_t n, int32_t* out);

#ifdef __cplusplus
}
#endif
#endif
