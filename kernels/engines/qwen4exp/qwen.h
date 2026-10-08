/* qwen.h: the qwen4exp engine's C ABI (qwen.cpp) - included by ns.h. All pointers device memory of the stage's GPU
 * unless named host. Weights are the GGUF's tensors as stored (GSQ-RCO keeps the small projections in BF16 and the
 * norms in F32, the forms the kernels read). */
#ifndef NS_QWEN_H
#define NS_QWEN_H
typedef struct ns_qw ns_qw;
typedef struct ns_qw_state ns_qw_state;

typedef struct ns_qw_layer {
    /* the hyper-connection reads: [0] attention half, [1] FFN half */
    const float* hc_norm[2];
    const uint16_t* hc_down[2];
    const uint16_t* hc_up[2];
    const uint16_t* hc_inject[2];
    /* Gated DeltaNet */
    int qkv_type, z_type, out_type;
    const void* qkv;
    const void* z_w;
    const void* out_w;
    const uint16_t* alpha;
    const uint16_t* beta;
    const float* conv;
    const float* ssm_a;
    const float* dt_bias;
    const float* ssm_norm;
    /* QSA */
    int q_type, k_type, v_type, o_type;
    const void* q_w;
    const void* k_w;
    const void* v_w;
    const void* o_w;
    const float* q_norm;
    const float* k_norm;
    const uint16_t* idx_q;
    const uint16_t* idx_k;
    const float* idx_q_norm;
    const float* idx_k_norm;
    /* MoE: the router, the shared expert, the routed experts' formats (their blobs are in the expert store) */
    const uint16_t* router;
    const uint16_t* sh_gate_inp;
    int sh_gate_type, sh_up_type, sh_down_type;
    const void* sh_gate;
    const void* sh_up;
    const void* sh_down;
    int gu_type, d_type;
    /* PLE (layer 1) */
    const uint16_t* ple_key;
    const uint16_t* ple_value;
    const float* ple_norm_key;
    const float* ple_norm_query;
    const float* ple_norm_conv;
    const uint16_t* ple_conv;
} ns_qw_layer;

typedef struct ns_qw_edges {
    /* the first stage: token_embd */
    int embd_type;
    const void* embd;
    size_t embd_row;
    /* the last stage: the final mixer and output */
    const float* out_hc_norm;
    const uint16_t* out_hc_down;
    const uint16_t* out_hc_up;
    int out_type;
    const void* out;
    int64_t vocab;
} ns_qw_edges;

typedef struct ns_qw_desc {
    int64_t lb, le, n_layer;
    /* routed experts a layer (512; the Coder's 256) */
    int64_t n_expert;
    /* the longest session this stage serves */
    int64_t max_cells;
    const ns_qw_layer* layers; /* le - lb of them */
    ns_qw_edges edges;
    /* the expert store: d_res [n_layer * 512] = slot or -1 (device), slot k at cache_base + slot_off[k] */
    const int32_t* d_res;
    const uint8_t* cache_base;
    const uint64_t* slot_off; /* host, n_slots */
    int64_t n_slots;
    const int32_t* h_res; /* host [n_layer * 512]: d_res's copy (the prompt path walks the experts on the host) */
    /* experts not in VRAM: [n_layer * 512] addresses in this GPU's pinned host memory (0 = none), device (the plan
     * kernel's) and host copies; null when every expert is in VRAM */
    const uint64_t* mirror;
    const uint64_t* h_mirror;
} ns_qw_desc;

int ns_qw_new(ns_gpu* g, const ns_qw_desc* d, ns_qw** out);
void ns_qw_free(ns_qw* w);
/* the stage's hand-off buffers (device): in (a later stage reads its residual there), out (an earlier one writes it);
 * hand_floats a token */
int ns_qw_buffers(ns_qw* w, float** hand_in, float** hand_out, size_t* hand_floats);

/* a session's state on this stage */
int ns_qw_state_new(ns_qw* w, int64_t max_cells, ns_qw_state** out);
void ns_qw_state_free(ns_qw_state* st);
int ns_qw_state_reset(ns_qw* w, ns_qw_state* st);
/* the session's graphs (windows of 1..max_t, the commit, the drafter's) recorded now rather than on first use */
int ns_qw_state_warm(ns_qw* w, ns_qw_state* st, int max_t);
/* checkpoints: the state of a session at position pos, as bytes (save / load host memory) */
int ns_qw_state_bytes(ns_qw_state* st, int64_t pos, uint64_t* bytes);
int ns_qw_state_save(ns_qw* w, ns_qw_state* st, int64_t pos, void* host);
int ns_qw_state_load(ns_qw* w, ns_qw_state* st, int64_t pos, const void* host);
int ns_qw_state_copy(ns_qw* w, ns_qw_state* dst, const ns_qw_state* src, int64_t pos);

/* T (1..8) tokens at pos0.. through the stage's layers. ple_rows (host, T x 2560): the tokens' PLE rows, for the
 * stage with layer 1. On the last stage, logits rows [logits_from, T) are copied to logits_host (vocab floats a row), and
 * (argmax_host not null) each row's argmax to argmax_host, picked on the GPU (Strata's greedy pick: the lowest index wins).
 * The state's GDN layers and indexer are advanced only by ns_qw_commit (a one-token window commits itself). */
int ns_qw_window(ns_qw* w, ns_qw_state* st, int T, const int32_t* tokens, int64_t pos0, const float* ple_rows,
                 int logits_from, float* logits_host, int32_t* argmax_host);
/* the last window's first n_keep tokens made permanent */
int ns_qw_commit(ns_qw* w, ns_qw_state* st, int n_keep);

/* the prompt path (Strata's prefill.cpp): chunks of up to `chunk` tokens; ns_qw_prefill_buffers sizes the stage's
 * buffers for it (once; a larger chunk reallocates) and gives the residual R (chunk x 4 x 2560 floats, device): a later
 * stage reads its chunk's residual there, an earlier one leaves it there */
int ns_qw_prefill_buffers(ns_qw* w, int64_t chunk, float** R);
/* T tokens at pos0.. (committed: the state advances); ple_rows (host, T x 2560) for the stage with layer 1 */
int ns_qw_prefill(ns_qw* w, ns_qw_state* st, int64_t T, const int32_t* tokens, int64_t pos0, const float* ple_rows);

/* a control vector (cvec.cpp; Strata's --control-vector-scaled): dir n_layer x 2560 (project: unit directions;
 * add: the offsets), s n_layer (project: the scale, add: 1; 0 = layer not steered), mode 0 project / 1 add. Set
 * before any session (its graphs hold where it applies); enable switches it per request (on after set) */
int ns_qw_cvec_set(ns_qw* w, const float* dir, const float* s, int n_layer, int mode);
int ns_qw_cvec_enable(ns_qw* w, int on);

/* the MTP draft layer (mtp.cpp, Strata's MtpDrafter): on the last stage, from Strata's runtime directory (tools/mtp_rt.py:
 * dense.txt / dense.bin / experts.bin / draft_vocab.bin); its own K/V lives in each session's state, so it is loaded
 * before any session. embd: the token embedding on this stage's GPU (the drafter embeds its tokens). window: the cells
 * it attends to (Strata's --mtp-window, 32768; 0 = all) */
int ns_qw_mtp_load(ns_qw* w, const char* rt_dir, int max_t, int64_t window, int embd_type, const void* embd, size_t embd_row);
/* the drafter's K/V for prompt cells [cell0, cell0 + n): the prompt path's final residual rows on this stage (its R from
 * ns_qw_prefill_buffers, rows [0, n)), next_tokens (host) the token after each cell */
int ns_qw_mtp_prefill(ns_qw* w, ns_qw_state* st, int64_t n, int64_t cell0, const int32_t* next_tokens);
/* one round after the window just run on this stage (its final residual rows): the catch-up over rows [0, a] (row t pairs
 * the residual at p + t with tokens[t], the token at p + t + 1), then drafts from row a while each is at least min_p
 * likely (at most max_drafts); drafts / probs host, n_out the drafts made */
int ns_qw_mtp_draft(ns_qw* w, ns_qw_state* st, const int32_t* tokens, int64_t p, int a, int max_drafts, float min_p,
                    int32_t* drafts, float* probs, int* n_out);
#endif
