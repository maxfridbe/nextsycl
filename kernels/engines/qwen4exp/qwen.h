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
    /* the longest session this stage serves */
    int64_t max_cells;
    const ns_qw_layer* layers; /* le - lb of them */
    ns_qw_edges edges;
    /* the expert store: d_res [n_layer * 512] = slot or -1 (device), slot k at cache_base + slot_off[k] */
    const int32_t* d_res;
    const uint8_t* cache_base;
    const uint64_t* slot_off; /* host, n_slots */
    int64_t n_slots;
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
/* checkpoints: the state of a session at position pos, as bytes (save / load host memory) */
int ns_qw_state_bytes(ns_qw_state* st, int64_t pos, uint64_t* bytes);
int ns_qw_state_save(ns_qw* w, ns_qw_state* st, int64_t pos, void* host);
int ns_qw_state_load(ns_qw* w, ns_qw_state* st, int64_t pos, const void* host);
int ns_qw_state_copy(ns_qw* w, ns_qw_state* dst, const ns_qw_state* src, int64_t pos);

/* T (1..8) tokens at pos0.. through the stage's layers. ple_rows (host, T x 2560): the tokens' PLE rows, for the
 * stage with layer 1. On the last stage, logits rows [logits_from, T) are copied to logits_host (vocab floats a row).
 * The state's GDN layers and indexer are advanced only by ns_qw_commit (a one-token window commits itself). */
int ns_qw_window(ns_qw* w, ns_qw_state* st, int T, const int32_t* tokens, int64_t pos0, const float* ple_rows,
                 int logits_from, float* logits_host);
/* the last window's first n_keep tokens made permanent */
int ns_qw_commit(ns_qw* w, ns_qw_state* st, int n_keep);
#endif
