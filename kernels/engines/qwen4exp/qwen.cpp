// qwen.cpp: Qwen3.8-Flash-Next's (qwen4exp) layers on the Strata SYCL kernels - the C ABI the Rust engine
// (engines/qwen4exp) drives. A port of Strata's verify window (sycl/src/core/verify.cpp at intel-arc-0.1.40,
// record_window / capture_commit): T tokens (1..8) at consecutive positions through a stage's layers, every
// per-token computation the kernels' own, in the order Strata runs them, so a token's logits are bitwise what Strata's
// window gives. Differences from Strata, all of them host-side: one group (no CPU experts to overlap: every routed
// expert is in VRAM), no graph capture yet (each window is launched as it runs), the inputs copied from this GPU's
// pinned memory, the PLE rows hashed and read by the Rust side.
//
// A stage is layers [lb, le) on one GPU; a stage that does not start at layer 0 takes its residual from the previous
// stage's hand-off (R, the pending write bo and inject, per token), one that does not end at the last layer writes
// that hand-off and has no head. The session state (GDN recurrences and conv histories, the QSA layers' int8 K/V and
// indexer, the PLE history) is per stage: ns_qw_state.
#include "ns.h"
#include "ns_internal.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstring>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr float EPS = 1e-6f;
// the artifact's geometry, which the kernels are built for (the Rust side checks the file against it)
constexpr int64_t N = 2560, HC = 4, HC_LR = 320, K = 10, NE = 512, NFF = 640;
constexpr int64_t S = 128, HK = 16, HV = 48, C = 2 * HK * S + HV * S, ZV = HV * S, DCONV = 4;
constexpr int64_t NH = 24, NKV = 2, HD = 256, IQ = 4, ID = 128;
constexpr int64_t GDN_FLOATS = S * HV * S + C * (DCONV - 1);
constexpr int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
constexpr int MAXT = kVerifyMaxT;

bool is_qsa(int64_t l) { return l % 4 == 3; }

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

}  // namespace

struct ns_qw_state {
    ns_gpu* g = nullptr;
    int64_t max_cells = 0;
    void* arena = nullptr;
    uint64_t bytes = 0;
    float* gdn = nullptr;      // the stage's GDN layers, GDN_FLOATS each: [recurrent state | conv history]
    std::vector<Qsa> qsa;      // the stage's QSA layers
    float* ple_hist = nullptr; // HS floats when the stage has the PLE layer
    // what save / load copy: [offset, bytes] of the fixed parts, and the per-cell parts (bytes a cell)
    struct Part { uint64_t off, fixed, per_cell; };
    std::vector<Part> parts;
};

struct ns_qw {
    ns_gpu* g = nullptr;
    sycl::queue* q = nullptr;
    int64_t lb = 0, le = 0, n_layer = 0, max_cells = 0, vocab = 0;
    std::vector<ns_qw_layer> L;   // le - lb
    ns_qw_edges E{};
    int64_t nG = 0, nQ = 0;       // the stage's GDN / QSA layers
    std::vector<int64_t> gdn_idx, qsa_idx;   // per stage layer
    bool has_ple = false;
    QsaShapes s{};
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
    uint32_t* h_plan_err = nullptr;
    // the last window, for its commit
    int last_t = 0;
    int64_t last_pos0 = 0;
    ns_qw_state* last_st = nullptr;
};

namespace {

int64_t handoff_floats() { return HC * N + N + HC; }

const int32_t* pos_k(const ns_qw* w) { return w->pos + MAXT * NH; }
const int32_t* pos_i(const ns_qw* w) { return w->pos + MAXT * (NH + NKV); }

void carve(ns_qw* w, Bump& b) {
    const uint64_t T = MAXT;
    const int max_in = (int) std::max<int64_t>(std::max<int64_t>(N, ZV), NH * HD);
    w->tok = b.take<int32_t>(T); w->step = b.take<int32_t>(T * kStepCount);
    w->pos = b.take<int32_t>(T * (NH + NKV + IQ)); w->commit = b.take<int32_t>(2 + T); w->one = b.take<int32_t>(4);
    w->ple = b.take<float>(T * N); w->emb = b.take<float>(T * N); w->R = b.take<float>(T * HC * N);
    w->mixed = b.take<float>(T * N); w->bo = b.take<float>(T * N);
    w->inj = b.take<float>(T * HC); w->inj2 = b.take<float>(T * HC);
    w->lo = b.take<float>(T * HC_LR); w->rs = b.take<float>(T * HC); w->xn = b.take<float>(T * HC * N);
    w->xq = b.take<uint8_t>(native_q8_1_bytes(max_in, (int) T));
    w->qkv_L = b.take<float>(w->nG * T * C); w->h_L = b.take<float>(w->nG * T * C);
    w->gate_L = b.take<float>(w->nG * T * HV); w->beta_L = b.take<float>(w->nG * T * HV);
    w->z = b.take<float>(T * ZV); w->y = b.take<float>(T * ZV); w->y_dummy = b.take<float>(T * ZV);
    w->qfull = b.take<float>(T * NH * 2 * HD); w->qcur = b.take<float>(T * NH * HD);
    w->kcur = b.take<float>(T * NKV * HD); w->vcur = b.take<float>(T * NKV * HD);
    w->idx_raw_L = b.take<float>(w->nQ * T * ID); w->qidx = b.take<float>(T * IQ * ID);
    w->scores = b.take<float>(T * (uint64_t) w->max_blocks); w->sel = b.take<int32_t>(T * (uint64_t) w->cap);
    w->attn = b.take<float>(T * NH * HD); w->attn32 = b.take<float>(T * NH * HD);
    w->attn_scratch = b.take<float>(T * (uint64_t) w->attn_floats);
    w->tail_snap = b.take<float>(w->nQ * (uint64_t) ((w->s.idx_block - 1) * ID));
    w->logits = b.take<float>(T * NE); w->w = b.take<float>(T * K); w->ids = b.take<int32_t>(T * K);
    w->shared = b.take<float>(T * N); w->parts = b.take<float>(T * K * N);
    w->plan = b.take<int32_t>((uint64_t) w->plan_i32 + 16);
    w->nat_xq = b.take<uint8_t>(T * (N / 32) * 36);
    w->hit_scratch = b.take<uint8_t>(native_expert_scratch_bytes((int64_t) (T * K), NFF));
    w->head_mixed = b.take<float>(T * N);
    w->sh_bf16 = b.take<uint16_t>(T * N); w->sh_gate = b.take<float>(T * NFF);
    w->sh_up = b.take<float>(T * NFF); w->sh_g = b.take<float>(T + 4);
    w->head_logits = b.take<float>(w->vocab > 0 ? T * (uint64_t) w->vocab : 1);
    w->hist_snap = b.take<float>(T * HS);
    w->ple_scratch = b.take<uint8_t>(ple_block_scratch_bytes() + NG_HC_DIM * sizeof(float));
    w->hand_in = b.take<float>(T * handoff_floats());
    w->hand_out = b.take<float>(T * handoff_floats());
}

// the switches Strata's --native preset turns on (generate.cpp): every kernel below is the pinned native variant
void native_preset() {
    gr_set_native_mmvf(true);
    ple_set_native_bf16(true);
    ple_set_native_postops(true);
    shared_expert_set_native_bf16(true);
    native_moe_combine_set_enabled(true);
    native_gdn_set_enabled(true);
    native_router_set_enabled(true);
    native_qsa_set_enabled(true);
    native_qsa_indexer_set_enabled(true);
    native_rope_set_enabled(true);
}

float* Rt(ns_qw* w, int t) { return w->R + (size_t) t * HC * N; }

}  // namespace

extern "C" {

int ns_qw_new(ns_gpu* g, const ns_qw_desc* d, ns_qw** out) {
    NS_TRY
    native_preset();
    auto* w = new ns_qw();
    w->g = g;
    w->q = &g->q;
    w->lb = d->lb; w->le = d->le; w->n_layer = d->n_layer; w->max_cells = d->max_cells;
    w->L.assign(d->layers, d->layers + (d->le - d->lb));
    w->E = d->edges;
    w->vocab = d->le == d->n_layer ? d->edges.vocab : 0;
    w->has_ple = d->lb <= 1 && 1 < d->le;
    for (int64_t l = d->lb; l < d->le; ++l) {
        w->gdn_idx.push_back(is_qsa(l) ? -1 : w->nG);
        w->qsa_idx.push_back(is_qsa(l) ? w->nQ : -1);
        (is_qsa(l) ? w->nQ : w->nG) += 1;
    }
    w->s = qsa_real_shapes();
    w->cap = qsa_selection_width(kTopkMaxCells, w->s);
    w->max_blocks = d->max_cells / w->s.idx_block + 2;
    w->attn_floats = (int64_t) qsa_decode_attn_scratch_floats(w->cap, w->s);
    {   // the plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
        const int64_t cap = (int64_t) MAXT * K;
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        w->plan_i32 = ptr_off + 4 * cap + (cap + 1) + 1;
    }
    w->d_res = d->d_res;
    w->cache_base = d->cache_base;
    Bump count;
    carve(w, count);
    w->arena = sycl::malloc_device(count.used + 64, g->dev, g->ctx);
    if (w->arena == nullptr) { delete w; return ns_fail("qwen: the window's device buffers (" + std::to_string(count.used >> 20) + " MiB) do not fit"); }
    g->q.memset(w->arena, 0, count.used).wait();
    Bump real;
    real.base = (uint8_t*) w->arena;
    carve(w, real);
    const int32_t one = 1;
    g->q.memcpy(w->one, &one, sizeof one).wait();
    if (d->n_slots > 0) {
        w->slot_off = sycl::malloc_device<unsigned long long>((size_t) d->n_slots, g->dev, g->ctx);
        g->q.memcpy(w->slot_off, d->slot_off, (size_t) d->n_slots * 8).wait();
    }
    const size_t in_ints = (size_t) MAXT * (1 + kStepCount + NH + NKV + IQ) + 2 + MAXT;
    w->h_in = (int32_t*) sycl::malloc_host(in_ints * 4, g->ctx);
    w->h_ple = (float*) sycl::malloc_host((size_t) MAXT * N * 4, g->ctx);
    w->h_plan_err = (uint32_t*) sycl::malloc_host(64, g->ctx);
    if (!w->h_in || !w->h_ple || !w->h_plan_err) { ns_qw_free(w); return ns_fail("qwen: pinned staging"); }
    std::memset(w->h_plan_err, 0, 64);
    *out = w;
    return 0;
    NS_CATCH
}

void ns_qw_free(ns_qw* w) {
    if (w == nullptr) return;
    try {
        w->q->wait();
        if (w->arena) sycl::free(w->arena, w->g->ctx);
        if (w->slot_off) sycl::free(w->slot_off, w->g->ctx);
        if (w->h_in) sycl::free(w->h_in, w->g->ctx);
        if (w->h_ple) sycl::free(w->h_ple, w->g->ctx);
        if (w->h_plan_err) sycl::free(w->h_plan_err, w->g->ctx);
    } catch (...) {}
    delete w;
}

int ns_qw_buffers(ns_qw* w, float** hand_in, float** hand_out, size_t* hand_floats) {
    *hand_in = w->hand_in;
    *hand_out = w->hand_out;
    *hand_floats = (size_t) handoff_floats();
    return 0;
}

// ---- session state

int ns_qw_state_new(ns_qw* w, int64_t max_cells, ns_qw_state** out) {
    NS_TRY
    if (max_cells > w->max_cells || max_cells < 8) return ns_fail("qwen: a session's context past the stage's");
    auto* st = new ns_qw_state();
    st->g = w->g;
    st->max_cells = max_cells;
    const QsaShapes& s = w->s;
    const int64_t pages = (max_cells + s.page_size - 1) / s.page_size;
    const uint64_t rows = (uint64_t) pages * s.n_head_kv * s.page_size;
    const uint64_t scale_row = (uint64_t) (s.head_dim / KV_Q8_GROUP);
    const uint64_t pooled_rows = (uint64_t) (max_cells / s.idx_block + 2);
    auto lay = [&](Bump& b) {
        st->parts.clear();
        auto part = [&](uint64_t fixed, uint64_t per_cell) {
            // a part's bytes: `fixed` always, plus `per_cell` for each cell up to the session's position
            st->parts.push_back({b.used, fixed, per_cell});
        };
        part((uint64_t) w->nG * GDN_FLOATS * 4, 0);
        st->gdn = b.take<float>((uint64_t) w->nG * GDN_FLOATS);
        st->qsa.assign((size_t) w->nQ, Qsa{});
        for (auto& q : st->qsa) {
            // one cell is n_head_kv rows of its page: (page * n_head_kv + h) * page_size + cell % page_size, so a prefix
            // of cells is a prefix of pages - whole pages copied
            part(0, (uint64_t) s.n_head_kv * s.head_dim);
            q.k_q = b.take<int8_t>(rows * s.head_dim);
            part(0, (uint64_t) s.n_head_kv * s.head_dim);
            q.v_q = b.take<int8_t>(rows * s.head_dim);
            part(0, (uint64_t) s.n_head_kv * scale_row * 2);
            q.k_scale = b.take<uint16_t>(rows * scale_row);
            part(0, (uint64_t) s.n_head_kv * scale_row * 2);
            q.v_scale = b.take<uint16_t>(rows * scale_row);
            q.page_table = b.take<int32_t>((uint64_t) pages);
            part((uint64_t) (s.idx_block - 1) * ID * 4, 0);
            q.idx_tail = b.take<float>((uint64_t) (s.idx_block - 1) * ID);
            part((uint64_t) ID * 4, 0);
            q.idx_dead = b.take<float>((uint64_t) ID);
            part(2 * ID * 4, ID);   // a pooled row a block: 1/idx_block of a row a cell, rounded up below
            q.idx_pooled = b.take<float>(pooled_rows * ID);
            part(4, 0);
            q.idx_block_pos = b.take<int32_t>(1);
        }
        if (w->has_ple) {
            part((uint64_t) HS * 4, 0);
            st->ple_hist = b.take<float>((uint64_t) HS);
        }
    };
    Bump count;
    lay(count);
    st->bytes = count.used;
    st->arena = sycl::malloc_device(count.used + 64, w->g->dev, w->g->ctx);
    if (st->arena == nullptr) { delete st; return ns_fail("qwen: a session's state (" + std::to_string(count.used >> 20) + " MiB) does not fit"); }
    Bump real;
    real.base = (uint8_t*) st->arena;
    lay(real);
    std::vector<int32_t> tab((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
    for (auto& q : st->qsa) w->g->q.memcpy(q.page_table, tab.data(), tab.size() * 4).wait();
    *out = st;
    return ns_qw_state_reset(w, st);
    NS_CATCH
}

void ns_qw_state_free(ns_qw_state* st) {
    if (st == nullptr) return;
    try {
        st->g->q.wait();
        if (st->arena) sycl::free(st->arena, st->g->ctx);
    } catch (...) {}
    delete st;
}

int ns_qw_state_reset(ns_qw* w, ns_qw_state* st) {
    NS_TRY
    // everything but the page tables: zero (Strata's session_init / qsa_state_zero)
    sycl::queue& q = w->g->q;
    q.memset(st->gdn, 0, (size_t) w->nG * GDN_FLOATS * 4);
    const QsaShapes& s = w->s;
    const int64_t pages = (st->max_cells + s.page_size - 1) / s.page_size;
    const uint64_t rows = (uint64_t) pages * s.n_head_kv * s.page_size;
    for (auto& x : st->qsa) {
        q.memset(x.k_q, 0, rows * s.head_dim);
        q.memset(x.v_q, 0, rows * s.head_dim);
        q.memset(x.k_scale, 0, rows * (s.head_dim / KV_Q8_GROUP) * 2);
        q.memset(x.v_scale, 0, rows * (s.head_dim / KV_Q8_GROUP) * 2);
        q.memset(x.idx_tail, 0, (size_t) (s.idx_block - 1) * ID * 4);
        q.memset(x.idx_dead, 0, (size_t) ID * 4);
        q.memset(x.idx_pooled, 0, (size_t) (st->max_cells / s.idx_block + 2) * ID * 4);
        q.memset(x.idx_block_pos, 0, 4);
    }
    if (st->ple_hist) q.memset(st->ple_hist, 0, (size_t) HS * 4);
    q.wait();
    return 0;
    NS_CATCH
}

// the bytes of each part a session at position `pos` holds
static uint64_t part_bytes(const ns_qw_state* st, const ns_qw_state::Part& p, int64_t pos) {
    if (p.per_cell == 0) return p.fixed;
    // the K/V parts: whole pages; the pooled indexer rows: a row a completed block, plus the two spare
    const int64_t cells = std::min<int64_t>(st->max_cells, (pos + 3) / 4 * 4);
    if (p.fixed > 0) return p.fixed + (uint64_t) (cells / 4) * p.per_cell * 4;
    return (uint64_t) cells * p.per_cell;
}

int ns_qw_state_bytes(ns_qw_state* st, int64_t pos, uint64_t* bytes) {
    uint64_t n = 0;
    for (const auto& p : st->parts) n += part_bytes(st, p, pos);
    *bytes = n;
    return 0;
}

int ns_qw_state_save(ns_qw* w, ns_qw_state* st, int64_t pos, void* host) {
    NS_TRY
    uint8_t* h = (uint8_t*) host;
    for (const auto& p : st->parts) {
        const uint64_t b = part_bytes(st, p, pos);
        w->g->q.memcpy(h, (uint8_t*) st->arena + p.off, b);
        h += b;
    }
    w->g->q.wait();
    return 0;
    NS_CATCH
}

int ns_qw_state_load(ns_qw* w, ns_qw_state* st, int64_t pos, const void* host) {
    NS_TRY
    if (pos > st->max_cells) return ns_fail("qwen: a checkpoint past this session's context");
    const uint8_t* h = (const uint8_t*) host;
    for (const auto& p : st->parts) {
        const uint64_t b = part_bytes(st, p, pos);
        w->g->q.memcpy((uint8_t*) st->arena + p.off, h, b);
        h += b;
    }
    w->g->q.wait();
    return 0;
    NS_CATCH
}

int ns_qw_state_copy(ns_qw* w, ns_qw_state* dst, const ns_qw_state* src, int64_t pos) {
    NS_TRY
    if (dst->max_cells < pos || dst->parts.size() != src->parts.size()) return ns_fail("qwen: copying a session into a smaller one");
    for (size_t i = 0; i < src->parts.size(); ++i)
        w->g->q.memcpy((uint8_t*) dst->arena + dst->parts[i].off, (const uint8_t*) src->arena + src->parts[i].off,
                       part_bytes(src, src->parts[i], pos));
    w->g->q.wait();
    return 0;
    NS_CATCH
}

// ---- the window

int ns_qw_window(ns_qw* w, ns_qw_state* st, int T, const int32_t* tokens, int64_t pos0, const float* ple_rows,
                 int logits_from, float* logits_host) {
    NS_TRY
    if (T < 1 || T > MAXT) return ns_fail("qwen: a window holds 1.." + std::to_string(MAXT) + " tokens");
    if (pos0 + T > st->max_cells) return ns_fail("qwen: the window runs past the session's context");
    sycl::queue* cs = w->q;
    const QsaShapes& s = w->s;
    const GrShapes gs{N, HC, HC_LR};
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t MT = MAXT;
    const bool self_commit = T == 1;   // Strata's one_token_self_commit (on except on HIP)
    native_expert_set_swiglu_limit(0.0f);   // the shared grouped-expert kernels' nextsycl switches: as upstream
    native_expert_set_lanes(0);
    native_expert_set_phase(0);

    // ---- the inputs (Strata's stage_inputs), through this GPU's pinned memory
    {
        int32_t* h = w->h_in;
        int32_t* h_tok = h;
        int32_t* h_step = h_tok + MT;
        int32_t* h_pos = h_step + MT * kStepCount;
        int32_t* pk = h_pos + MT * NH;
        int32_t* pi = pk + MT * NKV;
        for (int t = 0; t < T; ++t) {
            h_tok[t] = tokens[t];
            qsa_step_fill(h_step + t * kStepCount, pos0 + t, s);
            const int32_t p = (int32_t) (pos0 + t);
            for (int64_t i = 0; i < NH; ++i) h_pos[t * NH + i] = p;
            for (int64_t i = 0; i < NKV; ++i) pk[t * NKV + i] = p;
            for (int64_t i = 0; i < IQ; ++i) pi[t * IQ + i] = p;
        }
        cs->memcpy(w->tok, h_tok, (size_t) T * 4);
        cs->memcpy(w->step, h_step, (size_t) T * kStepCount * 4);
        cs->memcpy(w->pos, h_pos, (size_t) MT * (NH + NKV + IQ) * 4);
        if (w->has_ple) {
            if (ple_rows == nullptr) return ns_fail("qwen: the PLE layer's stage needs the window's PLE rows");
            std::memcpy(w->h_ple, ple_rows, (size_t) T * N * 4);
            cs->memcpy(w->ple, w->h_ple, (size_t) T * N * 4);
        }
    }

    // ---- the embeddings broadcast to the streams, or the previous stage's residual, pending write and inject
    if (w->lb > 0) {
        const float* hin = w->hand_in;
        cs->memcpy(w->R, hin, (size_t) T * HC * N * 4);
        cs->memcpy(w->bo, hin + (size_t) T * HC * N, (size_t) T * N * 4);
        cs->memcpy(w->inj2, hin + (size_t) T * (HC + 1) * N, (size_t) T * HC * 4);
    } else {
        iq_embed_rows(w->E.embd_type, w->E.embd, w->E.embd_row, w->tok, T, N, w->emb, cs);
        broadcast_streams(w->emb, w->R, N, (int) HC, T, cs);
    }

    const int n = T, tb = 0, te = T;
    float* xm = w->mixed;
    for (int64_t l = w->lb; l < w->le; ++l) {
        const ns_qw_layer& v = w->L[(size_t) (l - w->lb)];
        bool pending = l > 0;
        if (l == 1 && w->has_ple) {
            // the PLE block (Strata: verify.cpp pre(1), STRATA_PLE_BATCH's default form)
            float* normalized = (float*) (w->ple_scratch + ple_block_scratch_bytes());
            PleWeights pw{};
            pw.key_bf16 = v.ple_key;
            pw.value_bf16 = v.ple_value;
            pw.norm_key = v.ple_norm_key;
            pw.norm_query = v.ple_norm_query;
            pw.norm_conv = v.ple_norm_conv;
            pw.conv1d_f16 = v.ple_conv;
            const bool batch_kv = n > 1 && ple_native_bf16_enabled() && ple_native_postops_enabled();
            if (batch_kv) {
                bf16_gemv_fp32_mmvf_multi(w->ple, N, v.ple_key, w->xn, HC * N, N, HC * N, n, cs);
                bf16_gemv_fp32_mmvf_multi(w->ple, N, v.ple_value, w->z, N, N, N, n, cs);
            }
            if (n > 1) gr_write_multi(Rt(w, 0), w->bo, w->inj2, gs, Rt(w, 0), n, cs);
            for (int t = 0; t < T; ++t) {
                if (n == 1) gr_write(Rt(w, t), w->bo + t * N, w->inj2 + t * HC, gs, Rt(w, t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(w, t);
                if (batch_kv)
                    ple_block_projected(w->xn + (size_t) t * HC * N, w->z + (size_t) t * N, Rt(w, t), st->ple_hist, pw, po,
                                        w->ple_scratch, cs);
                else
                    ple_block(w->ple + t * N, Rt(w, t), st->ple_hist, pw, po, w->ple_scratch, cs);
                ple_history_advance(st->ple_hist, normalized, cs);
                copy_from_mapped(w->hist_snap + (size_t) t * HS, st->ple_hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(w, t); a.R_out = Rt(w, t); a.apply = apply;
                a.bo_prev = w->bo + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = v.hc_norm[half]; a.w_down = v.hc_down[half];
                a.w_up = v.hc_up[half]; a.w_inject = v.hc_inject[half];
                a.eps = EPS; a.lo = w->lo + t * HC_LR; a.rs = w->rs + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = w->mixed + t * N;
            }
            fused_gr_read_multi(fa, n, w->xn + (size_t) tb * HC * N, cs);
        };
        gr_read_group(0, pending, w->inj2, w->inj);
        const int64_t li = l - w->lb;
        if (!is_qsa(l)) {
            // ======================= GDN =======================
            const int64_t gi = w->gdn_idx[(size_t) li];
            float* state = st->gdn + (size_t) gi * GDN_FLOATS;
            float* conv = state + S * HV * S;
            float* qkv = w->qkv_L + (size_t) gi * MT * C;
            float* hb = w->h_L + (size_t) gi * MT * C;
            float* gate = w->gate_L + (size_t) gi * MT * HV;
            float* beta = w->beta_L + (size_t) gi * MT * HV;
            native_quantize_q8_1(xm, w->xq, (int) N, n, cs);
            native_mmvq(v.qkv_type, v.qkv, w->xq, qkv, (int) N, (int) C, n, cs);
            gdn_conv_l2_multi(conv, qkv, v.conv, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb, self_commit);
            gdn_ab_multi(xm, v.alpha, v.beta, v.dt_bias, v.ssm_a, gate, beta, (int) N, (int) HV, n, cs);
            native_mmvq(v.z_type, v.z_w, w->xq, w->z, (int) N, (int) ZV, n, cs);
            gdn_step_norm_multi(state, hb, (int) C, gate, beta, w->z, v.ssm_norm, EPS, w->y, (int) HK, (int) HV, te,
                                self_commit ? w->one : nullptr, cs, tb, nullptr);
            native_quantize_q8_1(w->y, w->xq, (int) ZV, n, cs);
            native_mmvq(v.out_type, v.out_w, w->xq, w->bo, (int) ZV, (int) N, n, cs);
        } else {
            // ======================= QSA =======================
            const int64_t qi = w->qsa_idx[(size_t) li];
            Qsa& sq = st->qsa[(size_t) qi];
            float* idx_raw = w->idx_raw_L + (size_t) qi * MT * ID;
            const bool fuse_nr = native_norm_rope_usable((int) HD, (int) s.n_rot);
            auto norm_rope = [&](float* data, const float* norm, int rows, int cols, const int32_t* p) {
                if (fuse_nr && native_norm_rope_usable(cols, (int) s.n_rot)) {
                    native_qsa_rms_norm_rope(data, cols, norm, data, rows, cols, (int) s.n_rot, EPS, rope_scaling(), p, cs);
                    return;
                }
                native_qsa_rms_norm_weighted(data, norm, data, cols, rows, EPS, cs);
                native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
            };
            native_quantize_q8_1(xm, w->xq, (int) N, n, cs);
            bf16_gemv_fp32_mmvf_multi(w->mixed, N, v.idx_k, idx_raw, ID, N, ID, n, cs);
            native_mmvq(v.k_type, v.k_w, w->xq, w->kcur, (int) N, (int) (NKV * HD), n, cs);
            native_mmvq(v.v_type, v.v_w, w->xq, w->vcur, (int) N, (int) (NKV * HD), n, cs);
            norm_rope(w->kcur, v.k_norm, (int) (n * NKV), (int) HD, pos_k(w));
            copy_from_mapped(w->tail_snap + (size_t) qi * TS, sq.idx_tail, TS, cs);
            const KvHostPools none{};
            kv_append_q8_steps(sq.k_q, sq.v_q, sq.k_scale, sq.v_scale, sq.page_table, w->step, kStepCount, w->kcur,
                               w->vcur, (int) (NKV * HD), n, s, cs, &none);
            const QsaIndexerBuffers ib{sq.idx_tail, sq.idx_dead, sq.idx_pooled, sq.idx_block_pos};
            native_qsa_indexer_append_steps(idx_raw, w->step + kStepPos, kStepCount, n, 0, v.idx_k_norm, EPS, ib, s,
                                            st->max_cells, rope_scaling(), cs);
            native_mmvq(v.q_type, v.q_w, w->xq, w->qfull, (int) N, (int) (NH * 2 * HD), n, cs);
            if (fuse_nr) {   // the q/gate split reads q straight out of the q|gate rows (stride 2 * HD)
                native_qsa_rms_norm_rope(w->qfull, (int) (2 * HD), v.q_norm, w->qcur, (int) (n * NH), (int) HD,
                                         (int) s.n_rot, EPS, rope_scaling(), w->pos, cs);
            } else {
                for (int64_t r = 0; r < n * NH; ++r)
                    cs->memcpy(w->qcur + r * HD, w->qfull + r * 2 * HD, (size_t) HD * 4);
                norm_rope(w->qcur, v.q_norm, (int) (n * NH), (int) HD, w->pos);
            }
            bf16_gemv_fp32_mmvf_multi(w->mixed, N, v.idx_q, w->qidx, IQ * ID, N, IQ * ID, n, cs);
            norm_rope(w->qidx, v.idx_q_norm, (int) (n * IQ), (int) ID, pos_i(w));
            qsa_block_scores(sq.idx_pooled, sq.idx_dead, w->qidx, w->step, n, w->max_blocks, s, w->scores, cs);
            qsa_block_topk(w->scores, w->step, n, w->max_blocks, w->cap, s, w->sel, cs);
            QsaAttnPools pools;
            pools.page_table = sq.page_table;
            pools.k_q = sq.k_q; pools.v_q = sq.v_q; pools.k_scale = sq.k_scale; pools.v_scale = sq.v_scale;
            qsa_decode_attn_batch(w->qcur, pools, w->sel, w->step, w->cap, s, w->attn_scratch, w->attn, n, cs);
            native_qsa_gate_apply(w->attn, w->qfull, w->attn32, (int) (n * NH), (int) HD, cs);
            native_quantize_q8_1(w->attn32, w->xq, (int) (NH * HD), n, cs);
            native_mmvq(v.o_type, v.o_w, w->xq, w->bo, (int) (NH * HD), (int) N, n, cs);
        }
        gr_read_group(1, true, w->inj, w->inj2);
        // ---- the router: softmax over the experts, top 10, renormalized
        bf16_gemv_fp32_mmvf_multi(w->mixed, N, v.router, w->logits, NE, N, NE, n, cs);
        native_router_top10_multi_ne(w->logits, w->ids, w->w, n, (int) NE, cs);
        resident_plan(w->ids, n * (int) K, (int) K, w->d_res + l * NE, (int) NE, w->cache_base, w->slot_off, 0, w->plan,
                      (long long) MT * K, nullptr, 0, cs, w->h_plan_err);
        // ---- the shared expert, scaled by sigmoid(gate_inp_shexp . x)
        NativeSharedWeights nsw;
        nsw.gate_type = v.sh_gate_type; nsw.gate_data = v.sh_gate;
        nsw.up_type = v.sh_up_type; nsw.up_data = v.sh_up;
        nsw.down_type = v.sh_down_type; nsw.down_data = v.sh_down;
        nsw.q8_1 = w->xq;
        shared_expert_multi(n, xm, w->sh_bf16, nsw, v.sh_gate_inp, w->sh_gate, w->sh_up, w->sh_g, w->shared, N, NFF, cs,
                            nullptr, 0);
        quantize_q8_1_rows(xm, n, N, w->nat_xq, cs);
        // ---- the routed experts, every one in VRAM: the plan the device made
        {
            const int64_t cap = (int64_t) n * K, capx = (int64_t) MT * K;
            int32_t* pl = w->plan;
            const int32_t* p_counts = pl;
            const int32_t* p_start = pl + 4;
            const int32_t* p_dst = p_start + capx + 1;
            const int32_t* p_tok = p_dst + capx;
            const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
            const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
            const NativeExpertLayout EL = native_expert_layout(v.gu_type, v.d_type, N, NFF);
            native_expert_grouped(EL, p_ptr, p_start, p_counts, p_dst, p_tok, cap, cap, w->nat_xq, w->hit_scratch,
                                  w->parts, cs, 0);
        }
        if (n > 1) native_moe_combine_multi(w->parts, w->w, w->shared, w->bo, N, K, n, cs);
        else native_moe_combine(w->parts, w->w, w->shared, w->bo, N, K, cs);
        if (l == w->n_layer - 1) gr_write_multi(Rt(w, 0), w->bo, w->inj2, gs, Rt(w, 0), n, cs);   // (Strata: every n)
    }
    if (w->le < w->n_layer) {   // an earlier stage: hand the residual on, no head
        float* hout = w->hand_out;
        cs->memcpy(hout, w->R, (size_t) T * HC * N * 4);
        cs->memcpy(hout + (size_t) T * HC * N, w->bo, (size_t) T * N * 4);
        cs->memcpy(hout + (size_t) T * (HC + 1) * N, w->inj2, (size_t) T * HC * 4);
    } else {
        // ---- the head: the final mixer (no pending write, no inject), then output for T columns
        FusedGrArgs fa[kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            fa[t].R = Rt(w, t); fa[t].R_out = Rt(w, t); fa[t].apply = false;
            fa[t].w_norm = w->E.out_hc_norm; fa[t].w_down = w->E.out_hc_down;
            fa[t].w_up = w->E.out_hc_up; fa[t].eps = EPS;
            fa[t].lo = w->lo + t * HC_LR; fa[t].rs = w->rs + t * HC; fa[t].mixed = w->head_mixed + t * N;
        }
        fused_gr_read_multi(fa, T, w->xn, cs);
        native_quantize_q8_1(w->head_mixed, w->xq, (int) N, T, cs);
        native_mmvq(w->E.out_type, w->E.out, w->xq, w->head_logits, (int) N, (int) w->vocab, T, cs);
        if (logits_host != nullptr && logits_from < T)
            cs->memcpy(logits_host, w->head_logits + (size_t) logits_from * w->vocab,
                       (size_t) (T - logits_from) * w->vocab * 4);
    }
    cs->wait_and_throw();
    if (*(volatile uint32_t*) w->h_plan_err != 0) {
        *(volatile uint32_t*) w->h_plan_err = 0;
        return ns_fail("qwen: the expert plan met an expert that is not in VRAM");
    }
    w->last_t = T;
    w->last_pos0 = pos0;
    w->last_st = st;
    return 0;
    NS_CATCH
}

int ns_qw_commit(ns_qw* w, ns_qw_state* st, int n_keep) {
    NS_TRY
    if (st != w->last_st || n_keep < 1 || n_keep > w->last_t) return ns_fail("qwen: a commit that is not the last window's");
    w->last_st = nullptr;
    if (w->last_t == 1) return 0;   // a one-token window committed itself
    sycl::queue* cs = w->q;
    const QsaShapes& s = w->s;
    const int64_t TS = (s.idx_block - 1) * ID, MT = MAXT;
    int32_t* hc = w->h_in;   // reused: the window's inputs are on the device
    hc[0] = n_keep;
    hc[1] = n_keep - 1;
    for (int t = 0; t < MT; ++t) hc[2 + t] = t < n_keep ? (int32_t) (w->last_pos0 + t) : -1;
    cs->memcpy(w->commit, hc, (size_t) (2 + MT) * 4);
    for (int64_t l = w->lb; l < w->le; ++l) {
        const ns_qw_layer& v = w->L[(size_t) (l - w->lb)];
        const int64_t li = l - w->lb;
        if (!is_qsa(l)) {
            const int64_t gi = w->gdn_idx[(size_t) li];
            float* state = st->gdn + (size_t) gi * GDN_FLOATS;
            float* conv = state + S * HV * S;
            gdn_conv_commit(conv, w->qkv_L + (size_t) gi * MT * C, (int) C, w->commit, cs);
            gdn_step_norm_multi(state, w->h_L + (size_t) gi * MT * C, (int) C, w->gate_L + (size_t) gi * MT * HV,
                                w->beta_L + (size_t) gi * MT * HV, w->z, v.ssm_norm, EPS, w->y_dummy, (int) HK, (int) HV,
                                (int) MT, w->commit, cs, (int) MT);
        } else {
            const int64_t qi = w->qsa_idx[(size_t) li];
            Qsa& sq = st->qsa[(size_t) qi];
            copy_from_mapped(sq.idx_tail, w->tail_snap + (size_t) qi * TS, TS, cs);
            const QsaIndexerBuffers ib{sq.idx_tail, sq.idx_dead, sq.idx_pooled, sq.idx_block_pos};
            native_qsa_indexer_append_steps(w->idx_raw_L + (size_t) qi * MT * ID, w->commit + 2, 1, (int) MT, 0,
                                            v.idx_k_norm, EPS, ib, s, st->max_cells, rope_scaling(), cs);
        }
    }
    if (w->has_ple) copy_indexed(st->ple_hist, w->hist_snap, HS, w->commit + 1, HS, cs);
    cs->wait_and_throw();
    return 0;
    NS_CATCH
}

}  // extern "C"
