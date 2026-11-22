/* silo.h: the inkling2 engine's silo - kernels tuned for Inkling-2-Mini's decode (dist/silo/libnextsycl-inkling2.so,
 * opened by the engine at load; without it the engine runs on its own kernels in the kind's library, ink.h). Its Q8_0 and
 * NVFP4 matrices are repacked once at load, losslessly: the quants (or the E2M1 codes) of every row apart from the scales
 * (Q8_0: half; NVFP4: its UE4M3 bytes), so a decode product reads 16-byte lines; the activations are quantized as
 * ggml's Q8_0 (as llama.cpp's own products quantize them) and each group of 32 is an int8 dot. A row's arithmetic is the
 * same whatever rows share its pass (a verify row equals its one-row pass bit for bit). */
#ifndef NS_INK2S_SILO_H
#define NS_INK2S_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- repacked matrices ---- */
/* whether the silo takes a type (8 Q8_0, 40 NVFP4) at a row length */
int ns_ink2s_supported(int type, int64_t n_in);
/* a repacked matrix [rows, n_in]: the quants (Q8_0: n_in int8 a row; NVFP4: n_in / 2 code bytes a row, a group of 32
 * values its 16 bytes as the GGUF block holds them) of all rows, then from ns_ink2s_scales_at the scales (Q8_0: n_in /
 * 32 halfs a row; NVFP4: n_in / 16 UE4M3 bytes a row). An expert stack is one matrix of experts x rows rows. */
size_t ns_ink2s_scales_at(int type, int64_t rows, int64_t n_in);
size_t ns_ink2s_packed_bytes(int type, int64_t rows, int64_t n_in);
int ns_ink2s_pack(ns_gpu* g, int type, const void* raw, int64_t rows, int64_t n_in, void* dst);
/* rows [row0, row0 + n) of a repacked matrix of `rows` rows into half: dst [n, n_in] (a prompt pass's products) */
int ns_ink2s_unpack_f16(ns_gpu* g, int type, const void* w, int64_t rows, int64_t row0, int64_t n, int64_t n_in, uint16_t* dst);

/* ---- the activations ---- */
/* rows x [T, n] quantized as ggml's Q8_0 (a block of 32: d = amax / 127, kept as half; q = round(x / d)): xq holds the
 * quants [T, n] int8, then (from ns_ink2s_xq_scales_at) the scales [T, n / 32] float */
size_t ns_ink2s_xq_bytes(int64_t T, int64_t n);
size_t ns_ink2s_xq_scales_at(int64_t T, int64_t n);
int ns_ink2s_xq(ns_gpu* g, const float* x, void* xq, int64_t T, int64_t n);
/* RMS norm (eps, weight w) straight to that layout: y (nullable) [T, n] the normed rows too */
int ns_ink2s_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t T, int64_t n, float eps);
/* the same for m (1..4) inputs in one launch: x[i], w[i], y[i] (nullable), xq[i] */
int ns_ink2s_rms_xq_many(ns_gpu* g, int m, const float* const* x, const float* const* w, float* const* y, void* const* xq, int64_t T, int64_t n, float eps);

/* the router of decode rows: logits [T, E] = x [T, C] . w [E, C] (float32) into the scratch logits, then (a work-item an
 * expert, at most 128) the choice as ink.h's ns_ink2_route:
 * sigmoid, the K largest p + bias, weights p / (sum + 1e-20) x scale into ids, wts [T, K] */
int ns_ink2s_router(ns_gpu* g, const float* x, const float* w, const float* bias, float* logits, int32_t* ids, float* wts, int64_t T, int64_t E,
                    int64_t C, int64_t K, float scale);

/* ---- the DSpark drafter (draft.cpp) ---- */
/* src [T, C] rounded to BF16 into feature k of feats [T, nf, C] */
int ns_ink2s_feat_put(ns_gpu* g, const float* src, float* feats, int64_t T, int64_t C, int64_t nf, int64_t k);
/* each row's features mixed by alpha [nf], RMS-normed (eps, weight w) into y (nullable) and its quantized rows xq (nullable) */
int ns_ink2s_feat_norm(ns_gpu* g, const float* feats, int64_t nf, const float* alpha, const float* w, float eps, float* y, void* xq, int64_t T, int64_t C);
/* silu(gate) up [T, F] quantized (ns_ink2s_xq's layout) */
int ns_ink2s_swiglu_xq(ns_gpu* g, const float* gate, const float* up, void* xq, int64_t T, int64_t F);
/* rows x [T, H x 128] in place: each head RMS-normed with w [128] (null: not), then NEOX-rotated at position p0 + t */
int ns_ink2s_qk_norm_rope(ns_gpu* g, float* x, int64_t T, int64_t H, const float* w, float eps, float theta, int64_t p0);
/* float32 to BF16 bits (round to nearest even) */
int ns_ink2s_to_bf16(ns_gpu* g, const float* x, uint16_t* y, int64_t n);
/* the Markov head's input row [state (rank) | w1[*tok] (BF16 [vocab, rank]) | hidden (C) | f_rnn (C)], BF16-rounded */
int ns_ink2s_rnn_inputs(ns_gpu* g, const float* state, const int32_t* tok, const uint16_t* w1, int64_t rank, const float* hidden, const float* f_rnn,
                        int64_t C, float* out);
/* its gates from the projection proj [3 rank] + bias: state [rank] updated in place, latent [rank] */
int ns_ink2s_rnn_gates(ns_gpu* g, const float* proj, const float* bias, float* state, float* latent, int64_t rank);
/* softcap(base + BF16(fold_b + bias)) over the first n (cap <= 0: none) into logits (nullable); its argmax into out[0],
 * the argmax's probability (softmax at temperature 1) into prob[0] (nullable) */
int ns_ink2s_bias_argmax(ns_gpu* g, const float* base, const float* bias, const float* fold_b, int64_t n, float cap, float* logits, int32_t* out,
                         float* prob);

/* ---- the products (repacked matrices) ---- */
/* y [T, rows] = W [rows, n_in] . x (xq: ns_ink2s_xq's layout of x [T, n_in]), T 1..16 */
int ns_ink2s_mv(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t rows, int64_t T);
/* several products in one launch (at most 8, one type): job j's matrix w[j] (rows[j] x n_in[j]) against xq[j] (T rows)
 * into y[j] [T, rows[j]] */
int ns_ink2s_mv_many(ns_gpu* g, int n, const int* ty, const void* const* w, const void* const* xq, float* const* y, const int64_t* rows,
                     const int64_t* n_in, int64_t T);

/* float32 / BF16 matrices (types 0, 30; n_in a multiple of 8) against float32 rows: y [T, rows] = W . x, T 1..16 (the
 * router, the loop controllers, the sink and convolution-tap projections) */
int ns_ink2s_fmv_supported(int type, int64_t n_in);
int ns_ink2s_fmv(ns_gpu* g, int type, const void* w, const float* x, float* y, int64_t n_in, int64_t rows, int64_t T);

/* ---- the routed experts at decode width (T 1..16, K experts a row; a stack of E experts) ---- */
/* gate | up of pair p = t K + k (its expert ids[p]: rows [2F, n_in] of the stack) against row t of xq, x the expert's
 * scale, through SiTU (beta, lb): act [T K, F] quantized straight into ns_ink2s_xq's layout (F a multiple of 32) */
int ns_ink2s_moe_gu(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* ids, const void* xq, void* act_q, int64_t T,
                    int64_t K, int64_t n_in, int64_t F, float beta, float lb);
/* a pass's distinct experts (T rows of K, P = T K pairs, at most 256): uni [1 + P + P + 16 P] int32 - their count, their
 * experts in order of first use, each one's pair count and pairs (pair order) */
int ns_ink2s_moe_union(ns_gpu* g, const int32_t* ids, int64_t P, int32_t* uni);
/* ns_ink2s_moe_gu with each distinct expert's lines read once for all its pairs (T at most 16): the same values */
int ns_ink2s_moe_gu_union(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* uni, const void* xq, void* act_q, int64_t T,
                          int64_t K, int64_t n_in, int64_t F, float beta, float lb);
/* ns_ink2s_moe_down the same way: each pair's weighted sums into part [P, n_out], then each row's in order into y */
int ns_ink2s_moe_down_union(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* uni, const float* wts, const void* act_q,
                            float* part, float* y, int64_t T, int64_t K, int64_t F, int64_t n_out);
/* the down products of the pairs (act_q [T K, F]; the stack's rows [n_out, F] an expert) weighted and summed: y [T,
 * n_out] = sum_k wts[t K + k] scale[e] (W_e . act) */
int ns_ink2s_moe_down(ns_gpu* g, int type, const void* w, int64_t E, const float* scale, const int32_t* ids, const float* wts, const void* act_q,
                      float* y, int64_t T, int64_t K, int64_t F, int64_t n_out);

/* ---- fused decode steps (dec.cpp; T 1..16) ---- */
/* the depth attention of states s_0 .. s_{ns-1} [T, C] (at most 8) by nq queries (at most 4), each mix then RMS-normed
 * by norm_w[q] (null: left unnormed) into out [nq, T, C], and quantized into xq[q] (nullable) */
int ns_ink2s_mix_norm(ns_gpu* g, const float* const* states, int64_t ns, const float* query, int64_t nq, const float* const* norm_w, float* out,
                      void* const* xq, int64_t T, int64_t C, float eps_mix, float eps_norm);
/* a short convolution with its residual: dst [T, C] = base (nullable; may be dst) + (x + conv(x)) inv, the state [3, C]
 * updated, snap (nullable) [3 + T, C] the state then the rows, out (nullable) the convolution's rows */
int ns_ink2s_conv_res(ns_gpu* g, const float* x, float* state, const float* w, float* snap, const float* base, float* dst, float* out, int64_t T, int64_t C,
                      float inv);
/* the attention's inputs after their projections, in one launch: k and v through their convolutions (the full layers'
 * input-dependent taps), k clipped and v RMS-normed, into the caches; q and r clipped after their scales (the full
 * layers' log scale after) in place */
typedef struct Ink2sQkv {
    int64_t T, p0, cap, Hq, Hkv, D, R;
    const float* kraw;   /* [T, Hkv D] */
    const float* vraw;
    const float* kdyn;   /* [T, Hkv, 4], nullable */
    const float* vdyn;
    float* kstate;       /* [3, Hkv D] */
    float* vstate;
    float* ksnap;        /* [3 + T, Hkv D], nullable */
    float* vsnap;
    const float* kw;     /* [Hkv D, 4] */
    const float* vw;
    float inv, inv_dyn;  /* 1 / sqrt(taps + 1), 1 / sqrt(2 taps + 1) */
    float* kout;         /* the rows as cached, float */
    float* vout;
    void* kc;            /* [Hkv, cap, D] half */
    void* vc;
    float* q;            /* [T, Hq, D], in place */
    float* r;            /* [T, Hq, R], in place */
    const float* qscale; /* [Hq, D] */
    const float* rscale; /* [Hq, R] */
    float thr, rthr, alpha, floor, eps;
    /* the BF16 projections made here from the normed inputs n0 / n1 / n2 [T, C] (nullable: kdyn / vdyn above, and no
     * sink): the sink [Hq, C] into sink [T, Hq] by the q work-groups; the k / v taps [Hkv x 4, C] by the k / v ones */
    int64_t C;
    const float* n0;
    const float* n1;
    const float* n2;
    const uint16_t* sinkw;
    float* sink;
    const uint16_t* kdw;
    const uint16_t* vdw;
} Ink2sQkv;
int ns_ink2s_post_qkv(ns_gpu* g, const Ink2sQkv* p);
/* the decode attention's splits (part: the base kernel's layout) merged in order with the sink, times sigmoid(gate),
 * into out [T, Hq x 128] and its quantized rows xq */
int ns_ink2s_merge_gate_xq(ns_gpu* g, const float* part, const float* sink, const float* gate, float* out, void* xq, int64_t T, int64_t Hq, int64_t ns);
/* (sink null: none; gate null: not gated) */
/* a loop controller: u [T, rank] = down [rank, 2C] . [rms(cur) | rms(msg)] (eps), then out [T, C] = cur - tanh(forget .
 * u) cur + tanh(gain . u) msg (gain, forget [C, rank]; rank at most 64) */
int ns_ink2s_ctrl_down(ns_gpu* g, const float* cur, const float* msg, const float* down, float* u, int64_t T, int64_t C, int64_t rank, float eps);
int ns_ink2s_ctrl_apply(ns_gpu* g, const float* cur, const float* msg, const float* u, const float* gain, const float* forget, float* out, int64_t T, int64_t C,
                        int64_t rank);
/* the same for n (1..4) controllers in one launch each (arrays of their buffers) */
int ns_ink2s_ctrl_down_many(ns_gpu* g, int n, const float* const* cur, const float* const* msg, const float* const* down, float* const* u, int64_t T,
                            int64_t C, int64_t rank, float eps);
int ns_ink2s_ctrl_apply_many(ns_gpu* g, int n, const float* const* cur, const float* const* msg, const float* const* u, const float* const* gain,
                             const float* const* forget, float* const* out, int64_t T, int64_t C, int64_t rank);
/* the decode attention split at absolute key positions (the base kernel's, ink.h ns_ink2_attn, without its merge):
 * part [T, Hq, ns, 130]; ns from ns_ink2s_attn_splits */
int64_t ns_ink2s_attn_splits(int64_t T, int64_t p0, int64_t win);
/* the same for a block (the drafter's): no bias, every row seeing from p0 + 1 - win (win <= 0: from 0) through the
 * whole block p0 .. p0 + T - 1 (not causal); merged by ns_ink2s_merge_gate_xq with sink and gate null */
int ns_ink2s_attn_part_block(ns_gpu* g, const float* q, const void* kc, const void* vc, float* part, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0,
                             int64_t win, float scale);
int ns_ink2s_attn_part(ns_gpu* g, const float* q, const void* kc, const void* vc, const float* r, const float* relb, int64_t E, int64_t R, float* part,
                       int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale);

/* ---- the prompt attention (pattn.cpp: flash attention on XMX) ---- */
/* rows q [T, Hq, 128] at positions p0.. (causal; win > 0: the keys less than win back) against the ring caches [Hkv,
 * cap, 128] half: logit = q.k / sqrt(128) + (distance < E ? r . relb[distance] : 0), the row's sink an extra logit of
 * no value; out [T, Hq, 128]. Three query heads a key head, r [T, Hq, R] with R at most 16 */
int ns_ink2s_attn_prompt_supported(int64_t Hq, int64_t Hkv, int64_t D, int64_t R);
int ns_ink2s_attn_prompt(ns_gpu* g, const float* q, const void* kc, const void* vc, const float* r, const float* relb, int64_t E, int64_t R, const float* sink,
                         float* out, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win);

#ifdef __cplusplus
}
#endif
#endif
