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
#include <vector>

namespace qw {

constexpr float EPS = 1e-6f;
// the artifact's geometry, which the kernels are built for (the Rust side checks the file against it)
constexpr int64_t N = 2560, HC = 4, HC_LR = 320, K = 10, NE = 512, NFF = 640;
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

}  // namespace qw

struct ns_qw_state {
    ns_gpu* g = nullptr;
    int64_t max_cells = 0;
    void* arena = nullptr;
    uint64_t bytes = 0;
    float* gdn = nullptr;      // the stage's GDN layers, GDN_FLOATS each: [recurrent state | conv history]
    std::vector<qw::Qsa> qsa;  // the stage's QSA layers
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
    uint32_t* h_plan_err = nullptr;
    // the last window, for its commit
    int last_t = 0;
    int64_t last_pos0 = 0;
    ns_qw_state* last_st = nullptr;
    // the prompt path (prefill.cpp): its buffers, made on first use; the residency table on the host
    struct Pf* pf = nullptr;
    std::vector<int32_t> h_res;
    std::vector<unsigned long long> h_slot_off;
};

namespace qw {
void prefill_free(ns_qw* w);   // prefill.cpp
}

