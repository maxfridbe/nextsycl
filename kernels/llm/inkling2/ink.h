/* ink.h: the inkling2 engine's own kernels (ink.cpp), bound by name by llm/inkling2/src/ffi.rs. Inkling-2-Mini
 * (llama.cpp's src/models/inkling2.cpp, inkling2-loop.cpp): attention with a relative-position bias and a per-row sink,
 * short convolutions with a state, attention over the depth (block residuals), loop controllers, a sigmoid-routed MoE
 * of NVFP4 experts with per-expert scales and the SiTU activation. Float32 activations, row-major; every launch on the
 * GPU's queue. ns_gpu is ns.h's. Types are ggml's codes: 0 F32, 8 Q8_0, 30 BF16, 40 NVFP4. */
#ifndef NS_INK2_H
#define NS_INK2_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- weights ---- */
/* whether ns_ink2_mv / ns_ink2_moe_mv / the expansions take the type (n_in: the row length) */
int ns_ink2_type_supported(int type, int64_t n_in);
/* n values of a typed tensor into float32 / half (rows are whole blocks) */
int ns_ink2_dequant(ns_gpu* g, int type, const void* src, int64_t n, float* dst);
int ns_ink2_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst);
/* decode-width products: y [T, n_out] = W [n_out, n_in] . x [T, n_in], T 1..16; a row's sums the same whatever T */
int ns_ink2_mv(ns_gpu* g, int type, const void* w, const float* x, float* y, int64_t n_in, int64_t n_out, int64_t T);
/* routed experts at decode width: pair p = t K + k uses expert ids[p] (its matrix at w + ids[p] * stride bytes); its
 * input x [T, n_in] row t (per_pair 0) or x [T K, n_in] row p (per_pair 1); y [T K, n_out] = scale[e] (W_e . x) (scale
 * nullable: 1) */
int ns_ink2_moe_mv(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const float* x, int per_pair, float* y,
                   int64_t T, int64_t K, int64_t n_in, int64_t n_out);

/* ---- the MoE ---- */
/* the router: logits [T, E] -> ids / weights [T, K]: p = sigmoid(logit), the K largest p + bias (bias nullable), weights
 * p / (sum of the K p + 1e-20) x scale */
int ns_ink2_route(ns_gpu* g, const float* logits, const float* bias, int32_t* ids, float* w, int64_t T, int64_t E, int64_t K, float scale);
/* SiTU: gu [rows, 2F] (gate | up), each row scaled by rs[row] first (rs nullable) -> act [rows, F] =
 * beta tanh(gate / beta) sigmoid(gate) x lb tanh(up / lb); act_h (nullable) the same in half */
int ns_ink2_situ(ns_gpu* g, const float* gu, const float* rs, float* act, uint16_t* act_h, int64_t rows, int64_t F, float beta, float lb);
/* y [T, C] = sum_k w[t K + k] e[t K + k] (e [T K, C]) */
int ns_ink2_combine(ns_gpu* g, const float* e, const float* w, float* y, int64_t T, int64_t K, int64_t C);
/* the prompt path's grouped rows: y [T, C] += sum over a token's entries j in [t_ptr[t], t_ptr[t + 1]) of w[j] rows[ent[j]] */
int ns_ink2_combine_rows(ns_gpu* g, const float* rows, const int32_t* t_ptr, const int32_t* ent, const float* w, float* y, int64_t T, int64_t C);
/* rows of x [., C] gathered into half: out [n, C] = x[idx[i]] */
int ns_ink2_gather_f16(ns_gpu* g, const float* x, const int32_t* idx, uint16_t* out, int64_t n, int64_t C);

/* ---- the depth ---- */
/* attention over the depth: states s_0 .. s_{ns-1} [T, C] (device pointers, at most 8), each RMS-normed (eps, no
 * weight); out [nq, T, C]: query q's softmax over the states of (normed s . query[q]) / sqrt(C), times the normed states */
int ns_ink2_mix(ns_gpu* g, const float* const* states, int64_t ns, const float* query, int64_t nq, float* out, int64_t T, int64_t C, float eps);
/* a loop controller's input: z [T, 2C] = [rms(cur) | rms(msg)] (eps, no weight) */
int ns_ink2_ctrl_z(ns_gpu* g, const float* cur, const float* msg, float* z, int64_t T, int64_t C, float eps);
/* and its output: out [T, C] = cur - tanh(f) cur + tanh(gn) msg (f, gn [T, C]: the forget and gain projections) */
int ns_ink2_ctrl_apply(ns_gpu* g, const float* cur, const float* msg, const float* f, const float* gn, float* out, int64_t T, int64_t C);

/* ---- the attention ---- */
/* the causal short convolution with its state: x [T, C] at positions after the state's (state [3, C]: the last three
 * inputs, updated); w [C, 4] (tap 3 the current row); dyn (nullable) [T, H, 4] input-dependent taps of head h's C / H
 * channels, tap l on the row l back; out [T, C] = (x + conv) x inv (out is not x); snap (nullable) [3 + T, C]: the
 * state then the rows, for a rollback */
int ns_ink2_conv(ns_gpu* g, const float* x, float* state, const float* w, const float* dyn, float* out, float* snap, int64_t T, int64_t C, int64_t H,
                 float inv);
/* the state after the first keep rows of the pass `snap` was written by: state = snap rows [keep, keep + 3) */
int ns_ink2_conv_keep(ns_gpu* g, const float* snap, float* state, int64_t keep, int64_t C);
/* rows x [T, H, D] in place: x scale[h, d] (nullable), then each (row, head) scaled to an L2 norm of at most thr; then
 * (log_alpha > 0) x tau(position) = 1 + alpha ln(max(1, (p0 + t + 1) / floor)) */
int ns_ink2_clip(ns_gpu* g, float* x, const float* scale, int64_t T, int64_t H, int64_t D, float thr, float log_alpha, float log_floor, int64_t p0);
/* RMS of each D-row (no weight): y = x / rms(x) (y may be x) */
int ns_ink2_rms_rows(ns_gpu* g, const float* x, float* y, int64_t rows, int64_t D, float eps);
/* rows k, v [T, Hkv, D] into the ring caches [Hkv, cap, D] half (slot = position % cap) */
int ns_ink2_kv_store(ns_gpu* g, const float* k, const float* v, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, void* kc, void* vc);
/* the attention, rows q [T, Hq, D] at positions p0.. (causal; win > 0: the keys less than win back): logit_j = scale q.k_j
 * + (d < E ? r[t, h] . relb[d] : 0) (d = the distance; r [T, Hq, R], relb [E, R]), with an extra logit sink[t, h] of no
 * value; out [T, Hq, D]. part: ns_ink2_attn_scratch floats (decode rows: the keys split at absolute positions, merged in
 * order, so a row's result is the same whatever rows share its pass) */
int64_t ns_ink2_attn_scratch(int64_t T, int64_t Hq, int64_t D, int64_t p0, int64_t win);
int ns_ink2_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, const float* r, const float* relb, int64_t E, int64_t R, const float* sink,
                 float* out, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, int64_t win, float scale, float* part);

/* ---- small ---- */
/* x [n] *= sigmoid(gate [n]) */
int ns_ink2_gate(ns_gpu* g, float* x, const float* gate, int64_t n);
/* y [n] = a + b (y may be a) */
int ns_ink2_add(ns_gpu* g, const float* a, const float* b, float* y, int64_t n);
/* logits [rows, ld] -> cap tanh(x / cap) over the first n of each row, packed into out [rows, n] (cap 0: copied) */
int ns_ink2_softcap(ns_gpu* g, const float* x, float* out, int64_t rows, int64_t n, int64_t ld, float cap);
/* each row's argmax (the lowest index wins a tie) */
int ns_ink2_argmax(ns_gpu* g, const float* x, int32_t* out, int64_t rows, int64_t n);
/* each row's draw (temperature, then top-p over its 256 largest), one uniform a row from (seed, pos0 + row) */
int ns_ink2_sample(ns_gpu* g, const float* x, int32_t* out, int64_t rows, int64_t n, float temperature, float top_p, uint64_t seed, int64_t pos0);

#ifdef __cplusplus
}
#endif
#endif
