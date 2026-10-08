// qwen_internal.hpp: what qwen.cpp (the window) and prefill.cpp (the prompt path) share - the stage and the session
// state. Not part of the ABI.
#pragma once
#include "ns.h"
#include "ns_internal.hpp"

#include <sycl/sycl.hpp>

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/ngram.hpp"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace qw {

constexpr float EPS = 1e-6f;
// the artifact's geometry, which the kernels are built for (the Rust side checks the file against it)
constexpr int64_t N = 2560, HC = 4, HC_LR = 320, K = 10, NE = 512, NFF = 640;   // NE: the most experts a layer (the buffers')
constexpr int64_t S = 128, HK = 16, HV = 48, C = 2 * HK * S + HV * S, ZV = HV * S, DCONV = 4;
constexpr int64_t NH = 24, NKV = 2, HD = 256, IQ = 4, ID = 128;
constexpr int64_t GDN_FLOATS = S * HV * S + C * (DCONV - 1);
constexpr int64_t HS = (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
constexpr int MAXT = strata::kernels::kVerifyMaxT;

inline bool is_qsa(int64_t l) { return l % 4 == 3; }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

// one QSA layer's state (Strata's QsaState, int8 K/V, fully resident, identity page table)
struct Qsa {
    int8_t* k_q = nullptr;
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;
    uint16_t* v_scale = nullptr;
    int32_t* page_table = nullptr;
    float* idx_tail = nullptr;
    float* idx_dead = nullptr;
    float* idx_pooled = nullptr;
    int32_t* idx_block_pos = nullptr;
};

using GraphExec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;

// NS_QW_EAGER=1: every launch as it runs, no graphs (the A/B)
inline bool graphs_on() {
    static const bool on = [] { const char* v = std::getenv("NS_QW_EAGER"); return !(v && v[0] == '1'); }();
    return on;
}

// `body`'s work on q: recorded into `slot` the first time (a graph a window size and session: the session's state
// pointers are in it), replayed after. Everything that varies between replays is read from device or pinned memory.
template <class F> void run_graph(sycl::queue& q, std::unique_ptr<GraphExec>& slot, F&& body, bool launch = true) {
    if (!graphs_on()) {
        if (launch) body();
        return;
    }
    if (!slot) {
        namespace X = sycl::ext::oneapi::experimental;
        X::command_graph<X::graph_state::modifiable> g(q.get_context(), q.get_device());
        g.begin_recording(q);
        try {
            body();
        } catch (...) {
            g.end_recording(q);
            throw;
        }
        g.end_recording(q);
        slot = std::make_unique<GraphExec>(g.finalize());
    }
    if (launch) q.ext_oneapi_graph(*slot);
}

}  // namespace qw

struct ns_qw_state {
    ns_gpu* g = nullptr;
    int64_t max_cells = 0;
    void* arena = nullptr;
    uint64_t bytes = 0;
    float* gdn = nullptr;      // the stage's GDN layers, GDN_FLOATS each: [recurrent state | conv history]
    std::vector<qw::Qsa> qsa;  // the stage's QSA layers
    qw::Qsa mtp;               // the draft layer's own K/V (the stage with the drafter; dense attention: no indexer)
    float* ple_hist = nullptr; // HS floats when the stage has the PLE layer
    // what save / load copy: [offset, bytes] of the fixed parts, and the per-cell parts (bytes a cell)
    struct Part { uint64_t off, fixed, per_cell; };
    std::vector<Part> parts;
    // its graphs: the window a size, the commit, the drafter's round a catch-up size and its steps
    std::unique_ptr<qw::GraphExec> win[qw::MAXT + 1], commit, mtp_round[qw::MAXT + 1], mtp_step[qw::MAXT + 1], mtp_pf[qw::MAXT + 1];
};

struct ns_qw {
    ns_gpu* g = nullptr;
    sycl::queue* q = nullptr;
    int64_t lb = 0, le = 0, n_layer = 0, max_cells = 0, vocab = 0;
    int64_t ne = 512;   // routed experts a layer in this model
    std::vector<ns_qw_layer> L;   // le - lb
    ns_qw_edges E{};
    int64_t nG = 0, nQ = 0;       // the stage's GDN / QSA layers
    std::vector<int64_t> gdn_idx, qsa_idx;   // per stage layer
    bool has_ple = false;
    strata::kernels::QsaShapes s{};
    int64_t cap = 0, max_blocks = 0, attn_floats = 0, plan_i32 = 0;
    // the expert store
    const int32_t* d_res = nullptr;
    const uint8_t* cache_base = nullptr;
    unsigned long long* slot_off = nullptr;
    // device arena
    void* arena = nullptr;
    int32_t *tok, *step, *pos, *commit, *one;
    float *ple, *emb, *R, *mixed, *bo, *inj, *inj2, *lo, *rs, *xn;
    uint8_t *xq, *nat_xq, *hit_scratch;
    float *qkv_L, *h_L, *gate_L, *beta_L, *z, *y, *y_dummy;
    float *qfull, *qcur, *kcur, *vcur, *idx_raw_L, *qidx, *scores;
    int32_t* sel;
    float *attn, *attn32, *attn_scratch, *tail_snap;
    float *logits, *w;
    int32_t* ids;
    float *shared, *parts;
    int32_t* plan;
    float *head_mixed, *sh_gate, *sh_up, *sh_g, *head_logits, *hist_snap;
    uint16_t* sh_bf16;
    uint8_t* ple_scratch;
    float* hand_in;
    float* hand_out;
    // pinned staging (this GPU's context)
    int32_t* h_in = nullptr;   // tok | step | pos | commit
    float* h_ple = nullptr;
    float* h_logits = nullptr;   // the last stage: the head's rows, read back after a window
    int32_t* out_ids = nullptr;  // the rows' argmax (device), its scratch
    uint8_t* arg_scratch = nullptr;
    uint32_t* h_plan_err = nullptr;
    // the last window, for its commit
    int last_t = 0;
    int64_t last_pos0 = 0;
    ns_qw_state* last_st = nullptr;
    // the prompt path (prefill.cpp): its buffers, made on first use; the residency table on the host
    struct Pf* pf = nullptr;
    // the MTP draft layer (mtp.cpp), on the last stage
    struct Mtp* mtp = nullptr;
    // a control vector (cvec.cpp), on every stage
    struct Cv* cv = nullptr;
    std::vector<int32_t> h_res;
    std::vector<unsigned long long> h_slot_off;
    const unsigned long long* mirror = nullptr;   // device: the host mirror's addresses (resident_plan_set_mirror)
    std::vector<unsigned long long> h_mirror;
};

namespace qw {
void prefill_free(ns_qw* w);   // prefill.cpp
// the prompt path's residual rows (its final ones after a chunk on the last stage) and the chunk they hold; null / 0 before
// the first prompt
float* prefill_rows(ns_qw* w);
int64_t prefill_cap(ns_qw* w);
void mtp_free(ns_qw* w);       // mtp.cpp
void mtp_warm(ns_qw* w, ns_qw_state* st);   // mtp.cpp: the drafter's graphs recorded
// cvec.cpp: layer l's FFN write is followed by the vector; the vector on T residual stacks (with the pending write first)
bool cvec_covers(const ns_qw* w, int64_t l);
void cvec_apply(ns_qw* w, float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld,
                const float* inj, int64_t inj_ld, bool write);
void cvec_free(ns_qw* w);
}

