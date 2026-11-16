/* z3.h: the ZONOS3-Mini engine's own kernels (audio/zonos3), beside the shared diffusion kernels (nsd.h: the float32
 * products on oneMKL, the layer norm, GELU, conv1d) and one of the audio library's (Qwen3-TTS's per-channel scaled add).
 * Bound by its crate by name (audio/zonos3/src/ffi.rs). Every buffer is float32 unless named otherwise; every call is
 * queued on the GPU's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error().
 *
 * The arithmetic follows Zyphra's ZONOS3 reference (github.com/Zyphra/ZONOS3-Mini, MIT; the model Apache-2.0):
 * zonos3.reference.lm, servers/ggml/src/zonos3-graph.cpp, zac.cpp and spk-encoder.cpp, written anew here. */
#ifndef NS_AUDIO_ZONOS3_H
#define NS_AUDIO_ZONOS3_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* ---- the backbone */

/* The prompt rows' embeddings: out[r] = sum_c audio[c][ids[r, c]] + text[ids[r, ncb]] (ids int32 [rows, ncb + 1];
 * audio half [ncb, va, D], text half [vt, D]) */
int ns_audio_z3_embed(ns_gpu* g, const void* audio, int64_t ncb, int64_t va, const void* text, int64_t vt, const int32_t* ids, int64_t rows, int64_t D,
                      float* out);

/* RMS norm of rows [rows, C] (row stride `xs`, out contiguous [rows, C]): out = x / sqrt(mean(x^2) + eps) (* w, w float32 [C] or NULL) */
int ns_audio_z3_rms(ns_gpu* g, const float* x, int64_t xs, int64_t rows, int64_t C, const float* w, float eps, float* out);

/* Full AttnRes: a source's scores against every query, scores[r, j] = rms(src[r]) . q[j] (q float32 [Q, D]) */
int ns_audio_z3_attnres_score(ns_gpu* g, const float* src, int64_t rows, int64_t D, const float* q, int64_t Q, float eps, float* scores);

/* Full AttnRes: the mix of the first S sources for query `qi`: out[r] = sum_s softmax_s(scores[s, r, qi]) bank[s, r];
 * bank [S, cap, D], scores [S, cap, Q] (`cap` rows a source) */
int ns_audio_z3_attnres_mix(ns_gpu* g, const float* bank, const float* scores, int64_t S, int64_t cap, int64_t rows, int64_t D, int64_t Q, int64_t qi,
                            float* out);

/* x[r, c] += v[c] (x [rows, C]) */
int ns_audio_z3_add_vec(ns_gpu* g, float* x, int64_t rows, int64_t C, const float* v);

/* The gated DeltaNet's causal convolution and SiLU over the first C columns of the projection's rows (row stride P),
 * continuing `hist` [C, K - 1] (the oldest first; updated): out[t, c] = silu(sum_j w[c, j] seq[t + j, c]) */
int ns_audio_z3_gdn_conv(ns_gpu* g, const float* proj, int64_t rows, int64_t P, int64_t C, const float* w, int64_t K, float* hist, float* out);

/* The recurrence's inputs per row and head: q and k L2-normed (x / sqrt(sum x^2 + 1e-6)), q * S^-1/2, v as it is,
 * the decay exp(-exp(A_log) softplus(a + dt_bias)) and beta = sigmoid(b) from the projection's a | b columns (at
 * column ab0 of each P-wide row); conv [rows, 3 H S] -> q, k, v [rows, H, S], decay, beta [rows, H] */
int ns_audio_z3_gdn_prep(ns_gpu* g, const float* conv, const float* proj, int64_t rows, int64_t P, int64_t ab0, int64_t H, int64_t S, const float* a_log,
                         const float* dt_bias, float* q, float* k, float* v, float* decay, float* beta);

/* The gated delta rule over the rows in order, state [H, S (v), S (k)] float32 (updated): per row and head
 * state *= decay; d = (v - state k) beta; state += d k^T; y = state q. y [rows, H, S]. S = 128 */
int ns_audio_z3_gdn_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* decay, const float* beta, int64_t rows, int64_t H,
                         int64_t S, float* state, float* y);

/* The recurrence's readout: out[r, h] = rms(y[r, h]) * w * silu(z[r, h]) (eps), z at column z0 of each P-wide row */
int ns_audio_z3_gdn_out(ns_gpu* g, const float* y, const float* proj, int64_t P, int64_t z0, int64_t rows, int64_t H, int64_t S, const float* w, float eps,
                        float* out);

/* Per-head RMS norm in place (rows `xs` apart, H heads of D), times scale[h] when given (the query temperature) */
int ns_audio_z3_head_rms(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t H, int64_t D, const float* scale, float eps);

/* Rotary positions on adjacent pairs (2i, 2i + 1) in place: angle (pos0 + r) theta^(-2i / D); rows `xs` apart, H heads */
int ns_audio_z3_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t H, int64_t D, float theta, int64_t pos0);

/* Attention: query row r (at key position pos0 + r) of head h over keys [0, keys) of key head h / (Hq / Hk): q rows qs
 * apart, k and v rows kvs apart; causal: keys at or before the query's position; window > 0: also within `window` of
 * it. out [rows, Hq D]. D 16, 32 or 64 */
int ns_audio_z3_attention(ns_gpu* g, const float* q, int64_t qs, int64_t rows, const float* k, const float* v, int64_t kvs, int64_t keys, int64_t Hq,
                          int64_t Hk, int64_t D, float scale, int64_t pos0, int causal, int64_t window, float* out);

/* x[r, h, :] *= sigmoid(gate[r, h]) (x [rows, H, D], gate [rows, H]) */
int ns_audio_z3_head_gate(ns_gpu* g, float* x, const float* gate, int64_t rows, int64_t H, int64_t D);

/* x = cap tanh(x / cap) */
int ns_audio_z3_softcap(ns_gpu* g, float* x, int64_t n, float cap);

/* out = silu(a) * b (n values; out may be a) */
int ns_audio_z3_silu_mul(ns_gpu* g, const float* a, const float* b, float* out, int64_t n);

/* The router: the k experts with the smallest bias - logit (the lowest index on a tie), their weights the softmax of
 * their raw logits: logits [rows, E] -> idx int32 [rows, k], w [rows, k] */
int ns_audio_z3_topk(ns_gpu* g, const float* logits, const float* bias, int64_t rows, int64_t E, int64_t k, int32_t* idx, float* w);

/* The routed experts' up half from Q4_0 stacks [E, F, L] (gate and up): h[r, s] = silu(x[r] . gate[e]) * (x[r] . up[e]),
 * e = idx[r, s]; x [rows, L], h [rows, k, F] */
int ns_audio_z3_moe_up(ns_gpu* g, const float* x, int64_t rows, int64_t L, const void* gate, const void* up, const int32_t* idx, int64_t k, int64_t F,
                       float* h);

/* ...and their down half, weighted: out[r] = sum_s w[r, s] (h[r, s] . down[e]^T), down Q4_0 [E, L, F]; out [rows, L] */
int ns_audio_z3_moe_down(ns_gpu* g, const float* h, int64_t rows, int64_t k, int64_t F, const void* down, const int32_t* idx, const float* w, int64_t L,
                         float* out);

/* ---- the codec's decoder */

/* x[t] (+)= table[codes[t, col]] (+ bias): codes int32 [T, ncol], table [V, C]; acc 0: out = row + bias */
int ns_audio_z3_lookup(ns_gpu* g, const int32_t* codes, int64_t T, int64_t ncol, int64_t col, const float* table, int64_t C, const float* bias, int acc,
                       float* out);

/* A causal depthwise convolution: out[t, c] = bias[c] + sum_j w[c, j] x[t + j - (K - 1), c] (zeros before the start) */
int ns_audio_z3_dwconv_causal(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out);

/* A centred depthwise convolution (K odd): out[t, c] = bias[c] + sum_j w[c, j] x[t + j - K / 2, c] (zeros outside) */
int ns_audio_z3_dwconv_same(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out);

/* Patch upsampling: out[t h + j, c] = x[t, c h + j] (x [T, C h]) */
int ns_audio_z3_patch_up(ns_gpu* g, const float* x, int64_t T, int64_t C, int64_t h, float* out);

/* The iSTFT branch's spectrum: y [F, 2 bins] (log magnitude | phase) -> ri [F, 2 bins] (mag cos | mag sin, the
 * imaginary parts of bins 0 and bins - 1 zero), mag = exp(min(logmag, maxlog)) */
int ns_audio_z3_spec(ns_gpu* g, const float* y, int64_t F, int64_t bins, float maxlog, float* ri);

/* Overlap-add of windowed frames [F, n] (hop apart, centred: torch.istft(center=True, length=F hop)) divided by the
 * window's squared envelope: out [F hop] */
int ns_audio_z3_ola(ns_gpu* g, const float* frames, int64_t F, int64_t n, int64_t hop, const float* win, float* out);

/* The FIR mixer of the two branches: out[t] = sum_c sum_j w[c, j] x_c[t + j - (K - 1) / 2] (zeros outside), w [2, K] */
int ns_audio_z3_mixer(ns_gpu* g, const float* x0, const float* x1, int64_t L, const float* w, int64_t K, float* out);

/* Polyphase resampling (torchaudio's): out[n new + i] = sum_j kern[i, j] xp[n orig + j], xp = x zero padded `width`
 * ahead; kern [new, 2 width + orig]; out `total` samples */
int ns_audio_z3_resample(ns_gpu* g, const float* x, int64_t L, const float* kern, int64_t orig, int64_t nw, int64_t width, float* out, int64_t total);

/* ---- the speaker encoder (channels-last [F, T, C]) */

/* A grouped 3 x 3 convolution, zero padding 1: x [Fi, Ti, Ci] -> out [Fo, To, Co], w [Co, a, 3, 3] (a = Ci / groups
 * input channels a group, Co / groups outputs a group), strides sf, st; bias [Co] or NULL */
int ns_audio_z3_conv3x3(ns_gpu* g, const float* x, int64_t Fi, int64_t Ti, int64_t Ci, const float* w, int64_t Co, int64_t a, int64_t sf, int64_t st,
                        const float* bias, float* out);

/* x * scale[c] + shift[c] (x [n, C]), then max(0, .) when relu; out may be x; add (NULL or [n, C]) is added last */
int ns_audio_z3_affine(ns_gpu* g, const float* x, int64_t n, int64_t C, const float* scale, const float* shift, int relu, const float* add, float* out);

/* out[f, t, c] = x[f sf, t st, c] for x [F, T, C] (out [ceil(F / sf), ceil(T / st), C]) */
int ns_audio_z3_subsample(ns_gpu* g, const float* x, int64_t F, int64_t T, int64_t C, int64_t sf, int64_t st, float* out);

/* out [B, A, C] = x [A, B, C] with the first two axes swapped */
int ns_audio_z3_swap01(ns_gpu* g, const float* x, int64_t A, int64_t B, int64_t C, float* out);

/* out[t] = x[t / rep] (+ add[t]) for t < T (rows of C) */
int ns_audio_z3_repeat_rows(ns_gpu* g, const float* x, int64_t rep, int64_t T, int64_t C, const float* add, float* out);

/* softmax over the last axis of rows [n, C] in place */
int ns_audio_z3_softmax(ns_gpu* g, float* x, int64_t n, int64_t C);

/* out[t, j] = sum_e r[t, e] x[t, e, j] (x [T, E, J], r [T, E]) */
int ns_audio_z3_mix_experts(ns_gpu* g, const float* x, const float* r, int64_t T, int64_t E, int64_t J, float* out);

/* out[t] = sum_s w[s] x_s[t] over 4 stage features (x0..x3 [n]), w [4, n] */
int ns_audio_z3_fuse4(ns_gpu* g, const float* x0, const float* x1, const float* x2, const float* x3, const float* w, int64_t T, int64_t C, float* out);

/* Attentive statistics pooling: alpha = softmax over T of att [T, C]; out [2C] = (sum alpha x, sqrt(max(sum alpha x^2
 * - mean^2, floor))) */
int ns_audio_z3_astp(ns_gpu* g, const float* x, const float* att, int64_t T, int64_t C, float floor, float* out);

/* x = tanh(x) */
int ns_audio_z3_tanh(ns_gpu* g, float* x, int64_t n);

/* ---- the reference encoder (raw-reference cloning: the acoustic encoder and the SSL teacher) */

/* Snake over channels-first x [C, L]: x + inv_alpha[c] sin^2(alpha[c] x) */
int ns_audio_z3_snake_cf(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, const float* inv_alpha, float* out);

/* x += y (n values) */
int ns_audio_z3_add(ns_gpu* g, float* x, const float* y, int64_t n);

/* Patch downsampling: out[t, c h + j] = x[t h + j, c] (x [T, C] -> out [T / h, C h]) */
int ns_audio_z3_patch_down(ns_gpu* g, const float* x, int64_t T, int64_t C, int64_t h, float* out);

/* GLU: out[t, c] = x[t, c] sigmoid(x[t, C + c]) (x [T, 2C]) */
int ns_audio_z3_glu(ns_gpu* g, const float* x, int64_t T, int64_t C, float* out);

/* x = silu(x) */
int ns_audio_z3_silu(ns_gpu* g, float* x, int64_t n);

/* Wav2Vec2-BERT attention with relative positions (heads of 64): score = scale (q.k_j + q.dist[clip(j - i, -left, right)
 * + left]); keys with valid[j] == 0 masked (valid NULL: none); q, k, v rows qs apart, out [T, H 64]; bf != 0: the
 * reference's BF16 recipe (scores, probabilities and the output rounded to bfloat16) */
int ns_audio_z3_attn_rel(ns_gpu* g, const float* q, const float* k, const float* v, int64_t qs, int64_t T, int64_t H, int64_t D, const float* dist,
                         int64_t left, int64_t right, const float* valid, float scale, int bf, float* out);

/* x rounded to bfloat16 values (nearest, ties to even), in place */
int ns_audio_z3_bf16(ns_gpu* g, float* x, int64_t n);

/* LACI's ACI mask on one decode row: the n masked heads (spec: head, lo, hi as absolute key positions) recomputed over
 * keys 0..keys-1 with the text keys [ts, te) outside [lo, hi] blocked; out [Hq hd] overwritten at those heads */
int ns_audio_z3_attn_window(ns_gpu* g, const float* q, const float* k, const float* v, int64_t kvs, int64_t keys, int64_t Hq, int64_t Hk, int64_t hd,
                            float scale, const int* spec, int64_t n, int64_t ts, int64_t te, float* out);

/* rotate-half RoPE from host tables cos / sin [rows, D] on x [rows, H D] (rows xs apart) */
int ns_audio_z3_rope_tab(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t H, int64_t D, const float* cs, const float* sn);

/* ConvTranspose1d's scatter: cols [T, K, C] -> y [To, C] (+ bias; NULL: none), stride S, padding P */
int ns_audio_z3_col2im(ns_gpu* g, const float* cols, int64_t T, int64_t K, int64_t C, int64_t S, int64_t P, const float* bias, int64_t To, float* y);

/* full attention, heads of up to 128: q, k, v [T, H hd] rows qs apart -> out [T, H hd] */
int ns_audio_z3_attn_full(ns_gpu* g, const float* q, const float* k, const float* v, int64_t qs, int64_t T, int64_t H, int64_t hd, float scale, float* out);

/* the speech-edit LoRA on the routed experts: t [rows k, R] = down[e] (E, R, In) . x (per_slot: x [rows k, In], else x
 * [rows, In]); moe_up / moe_down with the up factors' terms (upg / upu [E, F, R], upd [E, L, R]) */
int ns_audio_z3_moe_lora_t(ns_gpu* g, const float* x, int64_t rows, int64_t In, const int32_t* idx, int64_t k, int64_t per_slot, const float* down,
                           int64_t R, float* t);
int ns_audio_z3_moe_up_lora(ns_gpu* g, const float* x, int64_t rows, int64_t L, const void* gate, const void* up, const int32_t* idx, int64_t k, int64_t F,
                            const float* tg, const float* tu, const float* upg, const float* upu, int64_t R, float* h);
int ns_audio_z3_moe_down_lora(ns_gpu* g, const float* h, int64_t rows, int64_t k, int64_t F, const void* down, const int32_t* idx, const float* w, int64_t L,
                              const float* td, const float* upd, int64_t R, float* out);

/* channels-first x [rows, cols] += bias[row] */
int ns_audio_z3_add_rows(ns_gpu* g, float* x, const float* bias, int64_t rows, int64_t cols);

#ifdef __cplusplus
}
#endif
#endif
