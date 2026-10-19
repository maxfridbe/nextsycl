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
/* As ns_q35_qk_norm_rope, each row at its own positions: pos [T][3] (t, h, w), pair k of the first n_rot turning by
 * one of them as Qwen3-VL's interleaved sections (s0, s1, s2) choose (llama.cpp's IMROPE); a text row has all three
 * equal and comes out as ns_q35_qk_norm_rope at that position */
int ns_q35_qk_norm_rope_pos(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                            const float* w, float eps, int64_t n_rot, float theta, const int32_t* pos, int64_t s0, int64_t s1, int64_t s2);
/* The vision tower (Qwen3-VL's ViT): y [rows, C] = layer norm of x (weight w, bias b) */
int ns_q35_vit_ln(ns_gpu* g, const float* x, const float* w, const float* b, float* y, int64_t rows, int64_t C, float eps);
/* y [rows, N] += b (null: none), then act: 0 none, 1 GELU (tanh form), 2 GELU (erf) */
int ns_q35_vit_bias_act(ns_gpu* g, float* y, const float* b, int64_t rows, int64_t N, int act);
/* The 2D rotary positions of qkv's rows [n][3 H D] (cs: [n][D/2] (cos, sin), NEOX halves): q turned in place, k turned
 * into kh [H][n][D] */
int ns_q35_vit_rope(ns_gpu* g, float* qkv, int64_t n, int64_t H, int64_t D, const float* cs, float* kh);
/* The heads' outputs ob [H][r][D] as rows out [r][H x D] */
int ns_q35_vit_heads(ns_gpu* g, const float* ob, int64_t H, int64_t r, int64_t D, float* out);
/* LogProbChain's attention of one query row q [Hq x D] over the cache's first n keys: each head's softmax weights
 * (1 / sqrt(D)), per key their mean over the heads (mode 0) or their largest (mode 1), added to out [n]; scratch [Hq][2] */
int ns_q35_attn_probs(ns_gpu* g, const float* q, const void* kc, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t n, int mode,
                      float* scratch, float* out);
/* SYCL graphs: recording what is queued on the GPU's queue from begin (*out: the graph) to end (then finalized; it ran
 * nothing), launched as one submission, freed */
int ns_q35_graph_begin(ns_gpu* g, void** out);
int ns_q35_graph_end(ns_gpu* g, void* h);
int ns_q35_graph_launch(ns_gpu* g, void* h);
void ns_q35_graph_free(void* h);
/* x [rows, n] = softmax(x * scale) a row */
int ns_q35_softmax_rows(ns_gpu* g, float* x, int64_t rows, int64_t n, float scale);
/* The cache [Hkv][cap][D] half: rows k [T, Hkv x D] (ks floats apart) and v (vs apart) at positions p0.. */
int ns_q35_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                    void* kc, void* vc);
/* Floats of scratch ns_q35_attn needs for T rows of Hq heads (Hkv key heads) x D at positions p0.. (0: none) */
int64_t ns_q35_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0);
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
