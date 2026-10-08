// prefill.cpp: Qwen3.8-Flash-Next's prompt path on the Strata SYCL kernels - a port of the port's prefill.cpp
// (sycl/src/prefill/prefill.cpp at intel-arc-0.1.40, Prefill::run_impl's default path): a chunk of T tokens through a
// stage's layers at once, committed (the GDN recurrences, the conv histories, the K/V and the indexer advance).
//
// What it keeps of Strata's, per layer and half: the hyper-connection read as BF16 GEMMs over the chunk (gr_norm_rs,
// down, silu, up, inject, gr_mix_r), the projections through Gemm::native (each weight dequantized to FP16, then
// oneMKL), the GDN's gates / conv / recurrence kernels, QSA's appends, the indexer's batch append, the block scores
// as GEMM tiles (the SYCL port's), the prompt attention, and the MoE: the router and the shared expert as GEMMs, the
// (token, k) pairs grouped by expert on the host, each routed expert dequantized to FP16 and run as two GEMMs, one
// combine; the write fused with the next half's norm. What it does not have: the expert stream (every expert is in
// VRAM here), MMQ (not in the SYCL port), the layer-split helpers, the HIP / CUDA variants.
#include "qwen_internal.hpp"

#include <dpct/dpct.hpp>
#include <oneapi/mkl/blas.hpp>

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace strata::kernels {   // the SYCL port's (qsa_select.dp.cpp; declared by Strata in its prefill.cpp)
void qsa_block_scores_reduce(const float* S, int64_t ld, int64_t b0, int64_t nb, const float* dead, const float* q_idx,
                             const int32_t* steps, int64_t nq, int64_t max_blocks, float* scores, void* stream);
}

using namespace qw;
namespace SK = strata::kernels;
namespace P = strata::prefill;

namespace {
constexpr int64_t D = HC * N;            // the residual stack's width (10240)
constexpr int64_t LR = HC_LR;
constexpr int64_t GEMM_SCRATCH = 32ll << 20;   // FP16 elements of the largest dequantized dense weight (Strata's)
constexpr size_t GEMM_WS = 32u << 20;
constexpr int64_t SEL_TILE = 8192;            // blocks per GEMM tile of the QSA block scores (Strata's kSelTile)
constexpr int64_t SEL_BATCH = 256, ATTN_BATCH = 32;
constexpr int DQ = 2;
}  // namespace

struct Pf {
    int64_t T = 0;   // the chunk the buffers hold
    void* arena = nullptr;
    // always live
    float *emb, *R, *grs, *lo, *gated, *inj, *mixed, *bo;
    uint16_t *xn16, *lo16, *mixed_bf, *mixed_h;
    int32_t *steps_dev, *tok_dev;
    // one region, the attention half's and the MoE half's scratch (never live at once)
    uint8_t* region = nullptr;
    uint64_t region_bytes = 0;
    float *qkv, *z, *ab, *gate, *beta, *hbuf, *y;
    uint16_t* y_h;
    float *Kc, *Vc, *Qf, *q, *idx_raw, *q_idx, *attn, *sel_scores, *sel_S, *attn_scratch;
    uint16_t* attn_h;
    int32_t* sel_ids;
    float *logits, *wts, *GU, *Dm, *sgate, *sup, *shared, *sg;
    int32_t *ids, *slot_dev, *src_dev;
    uint16_t *Xs, *Hh, *sh_h;
    uint16_t *dq_gu[DQ], *dq_d[DQ];
    float *ple_emb, *ple_norm;
    uint16_t* gemm_scratch;
    void* gemm_ws;
    int64_t cap = 0, max_blocks = 0;
    P::Gemm gemm;
    // pinned host (this GPU's context)
    int32_t *h_tok = nullptr, *h_steps = nullptr, *h_ids = nullptr, *h_slot = nullptr, *h_src = nullptr;
    float* h_ple = nullptr;
    std::vector<int32_t> cnt, off;
};

namespace {

uint64_t gdn_set(Bump& a, Pf* p, size_t T) {
    p->qkv = a.take<float>(T * C); p->z = a.take<float>(T * ZV); p->ab = a.take<float>(T * 2 * HV);
    p->gate = a.take<float>(T * HV); p->beta = a.take<float>(T * HV); p->hbuf = a.take<float>(T * C);
    p->y = a.take<float>(T * ZV); p->y_h = a.take<uint16_t>(T * ZV);
    return a.used;
}
uint64_t qsa_set(Bump& a, Pf* p, size_t T, const SK::QsaShapes& s) {
    p->Kc = a.take<float>(T * 512); p->Vc = a.take<float>(T * 512); p->Qf = a.take<float>(T * 12288);
    p->q = a.take<float>(T * ZV); p->idx_raw = a.take<float>(T * 128); p->q_idx = a.take<float>(T * 512);
    p->attn = a.take<float>(T * ZV); p->attn_h = a.take<uint16_t>(T * ZV);
    p->sel_ids = a.take<int32_t>(T * (size_t) p->cap);
    p->sel_scores = a.take<float>((size_t) SEL_BATCH * (size_t) p->max_blocks);
    p->sel_S = a.take<float>((size_t) SEL_BATCH * 4 * (size_t) SEL_TILE);
    p->attn_scratch = a.take<float>((size_t) ATTN_BATCH * SK::qsa_decode_attn_scratch_floats(p->cap, s));
    return a.used;
}
uint64_t moe_set(Bump& a, Pf* p, size_t T) {
    p->logits = a.take<float>(T * NE); p->wts = a.take<float>(T * K); p->ids = a.take<int32_t>(T * K);
    p->slot_dev = a.take<int32_t>(T * K); p->src_dev = a.take<int32_t>(T * K);
    p->Xs = a.take<uint16_t>(T * K * N);
    p->GU = a.take<float>(T * K * 1280);
    p->Hh = a.take<uint16_t>(T * K * 640);
    p->Dm = a.take<float>(T * K * N);
    p->sgate = a.take<float>(T * 640); p->sup = a.take<float>(T * 640); p->sh_h = a.take<uint16_t>(T * 640);
    p->shared = a.take<float>(T * N); p->sg = a.take<float>(T);
    return a.used;
}

// everything of a chunk of T, counted (base null) or carved
void carve(Pf* p, Bump& o, size_t T, const SK::QsaShapes& s) {
    p->emb = o.take<float>(T * N); p->R = o.take<float>(T * D); p->grs = o.take<float>(T * HC);
    p->xn16 = o.take<uint16_t>(T * D); p->lo = o.take<float>(T * LR); p->lo16 = o.take<uint16_t>(T * LR);
    p->gated = o.take<float>(T * D); p->inj = o.take<float>(T * HC);
    p->mixed = o.take<float>(T * N); p->mixed_bf = o.take<uint16_t>(T * N); p->mixed_h = o.take<uint16_t>(T * N);
    p->bo = o.take<float>(T * N);
    p->steps_dev = o.take<int32_t>(T * SK::kStepCount); p->tok_dev = o.take<int32_t>(T);
    Bump c1, c2, c3;
    const uint64_t region = std::max({gdn_set(c1, p, T), qsa_set(c2, p, T, s), moe_set(c3, p, T)});
    p->region = o.take<uint8_t>(region);
    p->region_bytes = region;
    Bump a{p->region}, b{p->region}, c{p->region};
    gdn_set(a, p, T);
    qsa_set(b, p, T, s);
    moe_set(c, p, T);
    for (int i = 0; i < DQ; ++i) { p->dq_gu[i] = o.take<uint16_t>(1280 * N); p->dq_d[i] = o.take<uint16_t>(N * 640); }
    p->ple_emb = o.take<float>(T * N);
    p->ple_norm = o.take<float>((size_t) SK::NG_HC_DIM);
    p->gemm_scratch = o.take<uint16_t>((size_t) GEMM_SCRATCH);
    p->gemm_ws = o.take<uint8_t>(GEMM_WS);
}

void free_pf(ns_qw* w) {
    Pf* p = w->pf;
    if (p == nullptr) return;
    auto& ctx = w->g->ctx;
    if (p->arena) sycl::free(p->arena, ctx);
    for (void* h : {(void*) p->h_tok, (void*) p->h_steps, (void*) p->h_ids, (void*) p->h_slot, (void*) p->h_src, (void*) p->h_ple})
        if (h) sycl::free(h, ctx);
    delete p;
    w->pf = nullptr;
}

}  // namespace

namespace qw {
void prefill_free(ns_qw* w) {
    try { free_pf(w); } catch (...) {}
}
}  // namespace qw

extern "C" {

int ns_qw_prefill_buffers(ns_qw* w, int64_t chunk, float** R) {
    NS_TRY
    if (chunk < 1) return ns_fail("qwen: a prompt chunk of no tokens");
    if (w->pf == nullptr || w->pf->T < chunk) {
        w->q->wait();
        free_pf(w);
        auto* p = new Pf();
        p->T = chunk;
        p->cap = w->cap;
        p->max_blocks = w->max_blocks;
        Bump count;
        carve(p, count, (size_t) chunk, w->s);
        p->arena = sycl::malloc_device(count.used + 256, w->g->dev, w->g->ctx);
        if (p->arena == nullptr) {
            delete p;
            return ns_fail("qwen: the prompt path's buffers (" + std::to_string(count.used >> 20) + " MiB for " +
                           std::to_string(chunk) + " tokens) do not fit");
        }
        Bump real{(uint8_t*) p->arena};
        carve(p, real, (size_t) chunk, w->s);
        auto& ctx = w->g->ctx;
        p->h_tok = (int32_t*) sycl::malloc_host((size_t) chunk * 4, ctx);
        p->h_steps = (int32_t*) sycl::malloc_host((size_t) chunk * SK::kStepCount * 4, ctx);
        p->h_ids = (int32_t*) sycl::malloc_host((size_t) chunk * K * 4, ctx);
        p->h_slot = (int32_t*) sycl::malloc_host((size_t) chunk * K * 4, ctx);
        p->h_src = (int32_t*) sycl::malloc_host((size_t) chunk * K * 4, ctx);
        p->h_ple = (float*) sycl::malloc_host((size_t) chunk * N * 4, ctx);
        w->pf = p;
        if (!p->h_tok || !p->h_steps || !p->h_ids || !p->h_slot || !p->h_src || !p->h_ple) {
            free_pf(w);
            return ns_fail("qwen: the prompt path's pinned staging");
        }
        std::string err;
        if (!p->gemm.init_external(w->q, p->gemm_scratch, GEMM_SCRATCH, p->gemm_ws, GEMM_WS, err)) {
            free_pf(w);
            return ns_fail("qwen: " + err);
        }
        p->cnt.assign((size_t) NE, 0);
        p->off.assign((size_t) NE + 1, 0);
    }
    *R = w->pf->R;
    return 0;
    NS_CATCH
}

int ns_qw_prefill(ns_qw* w, ns_qw_state* st, int64_t T, const int32_t* tokens, int64_t pos0, const float* ple_rows) {
    NS_TRY
    Pf* p = w->pf;
    if (p == nullptr || T < 1 || T > p->T) return ns_fail("qwen: a prompt chunk past the prompt path's buffers");
    if (pos0 + T > st->max_cells) return ns_fail("qwen: the prompt runs past the session's context");
    if (w->last_st != nullptr) return ns_fail("qwen: a prompt chunk while a window is not committed");
    sycl::queue* cs = w->q;
    const SK::QsaShapes& s = w->s;
    P::Gemm& gm = p->gemm;
    const SK::KvHostPools none{};

    // ---- the embeddings broadcast to the streams (a later stage: its R is the previous stage's, copied in)
    if (w->lb == 0) {
        std::memcpy(p->h_tok, tokens, (size_t) T * 4);
        cs->memcpy(p->tok_dev, p->h_tok, (size_t) T * 4);
        SK::iq_embed_rows(w->E.embd_type, w->E.embd, w->E.embd_row, p->tok_dev, T, N, p->emb, cs);
        P::gr_broadcast(p->emb, p->R, T, cs);
    }
    // ---- the QSA step records of every position
    for (int64_t t = 0; t < T; ++t) SK::qsa_step_fill(p->h_steps + t * SK::kStepCount, pos0 + t, s);
    cs->memcpy(p->steps_dev, p->h_steps, (size_t) T * SK::kStepCount * 4);
    if (w->has_ple) {
        if (ple_rows == nullptr) return ns_fail("qwen: the PLE layer's stage needs the chunk's PLE rows");
        std::memcpy(p->h_ple, ple_rows, (size_t) T * N * 4);
        cs->memcpy(p->ple_emb, p->h_ple, (size_t) T * N * 4);
    }

    bool normed = false;   // the previous half's write already normed R for this half (grs, xn16)
    for (int64_t l = w->lb; l < w->le; ++l) {
        const ns_qw_layer& v = w->L[(size_t) (l - w->lb)];
        // ---- the PLE block at layer 1: the whole chunk (its projections as GEMMs, the postops batched), or token by
        // token when the scratch region holds too few tokens a batch (Strata: ple_batch)
        if (l == 1 && w->has_ple) {
            SK::PleWeights pw{};
            pw.key_bf16 = v.ple_key;
            pw.value_bf16 = v.ple_value;
            pw.norm_key = v.ple_norm_key;
            pw.norm_query = v.ple_norm_query;
            pw.norm_conv = v.ple_norm_conv;
            pw.conv1d_f16 = v.ple_conv;
            constexpr int64_t HD = SK::NG_HC_DIM;
            const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * 2 + 4096;
            const bool batch = SK::ple_native_postops_enabled() && p->region_bytes / per_token >= 64;
            if (batch) {
                const int64_t SB = std::min<int64_t>(T, (int64_t) (p->region_bytes / per_token));
                for (int64_t s0 = 0; s0 < T; s0 += SB) {
                    const int64_t nb = std::min(SB, T - s0);
                    uint8_t* q = p->region;
                    auto carve_f = [&](size_t n) { float* r = (float*) q; q += (n * 4 + 255) & ~(size_t) 255; return r; };
                    float* key = carve_f((size_t) nb * HD);
                    float* qn = carve_f((size_t) nb * HD);
                    float* gated = carve_f((size_t) nb * HD);
                    float* val = carve_f((size_t) nb * N);
                    float* gate = carve_f((size_t) nb * 4);
                    uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                    P::to_bf16(p->ple_emb + s0 * N, e16, nb * N, cs, nullptr);
                    gm.bf16(e16, v.ple_key, key, nb, HD, N);
                    gm.bf16(e16, v.ple_value, val, nb, N, N);
                    SK::native_ple_postops_batch(key, p->R + s0 * D, val, st->ple_hist, pw, qn, gated, gate, (int) nb, cs);
                }
            } else {
                // (the block's own scratch: the window's, idle between windows)
                float* normalized = p->ple_norm;
                for (int64_t t = 0; t < T; ++t) {
                    SK::PleOut po;
                    po.normalized = normalized;
                    po.result = p->R + t * D;
                    SK::ple_block(p->ple_emb + t * N, p->R + t * D, st->ple_hist, pw, po, w->ple_scratch, cs);
                    SK::ple_history_advance(st->ple_hist, normalized, cs);
                }
            }
        }
        for (int half = 0; half < 2; ++half) {
            // ---- the hyper-connection read of this half
            const float* wn = v.hc_norm[half];
            if (!normed) P::gr_norm_rs(p->R, wn, EPS, p->grs, p->xn16, T, cs, nullptr, D);
            normed = false;
            gm.bf16(p->xn16, v.hc_down[half], p->lo, T, LR, D);
            P::gr_silu(p->lo, p->lo16, T, cs, nullptr);
            gm.bf16(p->lo16, v.hc_up[half], p->gated, T, D, LR);
            gm.bf16(p->xn16, v.hc_inject[half], p->inj, T, HC, D);
            P::gr_mix_r(p->R, p->grs, wn, p->gated, p->mixed, p->mixed_bf, T, cs, p->mixed_h, nullptr);
            const int64_t li = l - w->lb;
            if (half == 0 && !is_qsa(l)) {
                // ======================= GDN =======================
                const int64_t gi = w->gdn_idx[(size_t) li];
                float* state = st->gdn + (size_t) gi * GDN_FLOATS;
                float* conv = state + S * HV * S;
                gm.native(p->mixed_h, v.qkv_type, v.qkv, p->qkv, T, C, N);
                gm.native(p->mixed_h, v.z_type, v.z_w, p->z, T, ZV, N);
                gm.bf16(p->mixed_bf, v.alpha, p->ab, T, HV, N, 2 * HV);
                gm.bf16(p->mixed_bf, v.beta, p->ab + HV, T, HV, N, 2 * HV);
                P::gdn_gates(p->ab, v.dt_bias, v.ssm_a, p->gate, p->beta, T, cs);
                P::gdn_conv(conv, p->qkv, v.conv, p->hbuf, T, EPS, cs);
                P::gdn_recurrence(state, p->hbuf, p->gate, p->beta, p->z, v.ssm_norm, EPS, p->y, p->y_h, T, cs, 0);
                gm.native(p->y_h, v.out_type, v.out_w, p->bo, T, N, ZV);
            } else if (half == 0) {
                // ======================= QSA =======================
                const int64_t qi = w->qsa_idx[(size_t) li];
                Qsa& sq = st->qsa[(size_t) qi];
                gm.native(p->mixed_h, v.k_type, v.k_w, p->Kc, T, NKV * HD, N);
                gm.native(p->mixed_h, v.v_type, v.v_w, p->Vc, T, NKV * HD, N);
                gm.native(p->mixed_h, v.q_type, v.q_w, p->Qf, T, NH * 2 * HD, N);
                gm.bf16(p->mixed_bf, v.idx_k, p->idx_raw, T, ID, N);
                gm.bf16(p->mixed_bf, v.idx_q, p->q_idx, T, IQ * ID, N);
                P::rms_rows(p->Kc, v.k_norm, T * NKV, HD, HD, EPS, cs);
                P::rope(p->Kc, T, NKV, HD, NKV * HD, pos0, SK::rope_scaling(), cs);
                P::kv_append(p->Kc, p->Vc, T, pos0, sq.page_table, s.page_size, nullptr, nullptr, sq.k_q, sq.v_q, sq.k_scale,
                             sq.v_scale, cs, &none, nullptr);
                P::split_q(p->Qf, p->q, T, cs);
                P::rms_rows(p->q, v.q_norm, T * NH, HD, HD, EPS, cs);
                P::rope(p->q, T, NH, HD, NH * HD, pos0, SK::rope_scaling(), cs);
                P::rms_rows(p->q_idx, v.idx_q_norm, T * IQ, ID, ID, EPS, cs);
                P::rope(p->q_idx, T, IQ, ID, IQ * ID, pos0, SK::rope_scaling(), cs);
                const SK::QsaIndexerBuffers ib{sq.idx_tail, sq.idx_dead, sq.idx_pooled, sq.idx_block_pos};
                SK::native_qsa_indexer_append_batch(p->idx_raw, T, pos0, 0, v.idx_k_norm, EPS, ib, s, st->max_cells,
                                                   SK::rope_scaling(), cs);
                // the selection, SEL_BATCH queries at a time: the block scores as GEMM tiles (pooled keys x the batch's
                // indexer queries) and a relu-sum, then the top-k
                for (int64_t t0 = 0; t0 < T; t0 += SEL_BATCH) {
                    const int64_t nb = std::min(SEL_BATCH, T - t0);
                    const int32_t* steps0 = p->steps_dev + t0 * SK::kStepCount;
                    const int64_t active = (int64_t) p->h_steps[(size_t) ((t0 + nb - 1) * SK::kStepCount + SK::kStepNBid)] + 1;
                    for (int64_t b0 = 0; b0 < active; b0 += SEL_TILE) {
                        const int64_t nbk = std::min<int64_t>(SEL_TILE, active - b0);
                        oneapi::mkl::blas::column_major::gemm(*cs, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                                                              nbk, nb * 4, 128, 1.0f, sq.idx_pooled + b0 * 128, 128,
                                                              p->q_idx + t0 * 512, 128, 0.0f, p->sel_S, SEL_TILE);
                        SK::qsa_block_scores_reduce(p->sel_S, SEL_TILE, b0, nbk, sq.idx_dead, p->q_idx + t0 * 512, steps0, nb,
                                                   p->max_blocks, p->sel_scores, cs);
                    }
                    SK::qsa_block_topk(p->sel_scores, steps0, nb, p->max_blocks, p->cap, s, p->sel_ids + t0 * p->cap, cs, active);
                }
                SK::QsaAttnPools pools;
                pools.page_table = sq.page_table;
                pools.k_q = sq.k_q; pools.v_q = sq.v_q; pools.k_scale = sq.k_scale; pools.v_scale = sq.v_scale;
                if (!SK::qsa_prompt_attn_batch(p->q, pools, p->sel_ids, p->steps_dev, p->cap, s, p->attn, T, cs))
                    for (int64_t t0 = 0; t0 < T; t0 += ATTN_BATCH) {
                        const int64_t nb = std::min(ATTN_BATCH, T - t0);
                        SK::qsa_decode_attn_batch(p->q + t0 * ZV, pools, p->sel_ids + t0 * p->cap, p->steps_dev + t0 * SK::kStepCount,
                                                 p->cap, s, p->attn_scratch, p->attn + t0 * ZV, nb, cs);
                    }
                P::gate_attn(p->attn, p->Qf, p->attn_h, T, cs, 0);
                gm.native(p->attn_h, v.o_type, v.o_w, p->bo, T, N, NH * HD);
            } else {
                // ======================= MoE =======================
                gm.bf16(p->mixed_bf, v.router, p->logits, T, NE, N);
                P::route(p->logits, p->ids, p->wts, T, NE, cs);
                gm.native(p->mixed_h, v.sh_gate_type, v.sh_gate, p->sgate, T, NFF, N);
                gm.native(p->mixed_h, v.sh_up_type, v.sh_up, p->sup, T, NFF, N);
                P::swiglu_pair(p->sgate, p->sup, p->sh_h, T, cs);
                gm.native(p->sh_h, v.sh_down_type, v.sh_down, p->shared, T, N, NFF);
                gm.bf16(p->mixed_bf, v.sh_gate_inp, p->sg, T, 1, N);
                // the (token, k) pairs grouped by expert on the host
                cs->memcpy(p->h_ids, p->ids, (size_t) T * K * 4);
                cs->wait_and_throw();
                std::fill(p->cnt.begin(), p->cnt.end(), 0);
                for (int64_t i = 0; i < T * K; ++i) {
                    const int32_t e = p->h_ids[(size_t) i];
                    if (e < 0 || e >= NE) return ns_fail("qwen: a routed expert id out of range");
                    ++p->cnt[(size_t) e];
                }
                int32_t r = 0;
                for (int32_t e = 0; e < NE; ++e) { p->off[(size_t) e] = r; r += p->cnt[(size_t) e]; }
                p->off[(size_t) NE] = r;
                std::vector<int32_t> fill(p->off.begin(), p->off.end() - 1);
                for (int64_t i = 0; i < T * K; ++i) {
                    const int32_t e = p->h_ids[(size_t) i];
                    const int32_t at = fill[(size_t) e]++;
                    p->h_slot[(size_t) i] = at;
                    p->h_src[(size_t) at] = (int32_t) (i / K);
                }
                cs->memcpy(p->slot_dev, p->h_slot, (size_t) T * K * 4);
                cs->memcpy(p->src_dev, p->h_src, (size_t) T * K * 4);
                P::gather_rows16(p->mixed_h, p->src_dev, p->Xs, T * K, N, cs);
                // the routed experts in id order, from their VRAM blobs: dequantized to FP16, gate | up, SwiGLU, down
                const SK::NativeExpertLayout EL = SK::native_expert_layout(v.gu_type, v.d_type, N, NFF);
                size_t j = 0;
                for (int32_t e = 0; e < NE; ++e) {
                    const int32_t ne = p->cnt[(size_t) e];
                    if (ne == 0) continue;
                    const int32_t slot = w->h_res[(size_t) l * NE + e];
                    if (slot < 0) return ns_fail("qwen: an expert not in this stage's VRAM");
                    const uint8_t* blob = w->cache_base + w->h_slot_off[(size_t) slot];
                    const int q = (int) (j++ % DQ);
                    SK::iq_dequant_gu_f16(v.gu_type, blob, blob + EL.up_off, NFF, N, p->dq_gu[q], cs);
                    SK::iq_dequant_f16(v.d_type, blob + EL.down_off, N * NFF, p->dq_d[q], cs);
                    const int64_t o0 = p->off[(size_t) e];
                    gm.f16(p->Xs + o0 * N, p->dq_gu[q], p->GU + o0 * 1280, ne, 1280, N);
                    P::swiglu_interleaved(p->GU + o0 * 1280, p->Hh + o0 * 640, ne, cs);
                    gm.f16(p->Hh + o0 * 640, p->dq_d[q], p->Dm + o0 * N, ne, N, 640);
                }
                P::moe_combine(p->Dm, p->slot_dev, p->wts, p->shared, p->sg, p->bo, T, cs);
            }
            // ---- the write of this half, fused with the next half's norm when nothing else touches R in between (not
            // the stage's last half, not before the PLE block of layer 1)
            const int64_t nl = half == 0 ? l : l + 1;
            const bool fuse = nl < w->le && !(half == 1 && nl == 1 && w->has_ple);
            if (fuse) {
                const float* wnn = half == 0 ? v.hc_norm[1] : w->L[(size_t) (nl - w->lb)].hc_norm[0];
                P::gr_write_norm_rs(p->R, p->bo, p->inj, HC, wnn, EPS, p->grs, p->xn16, T, cs, nullptr, D);
                normed = true;
            } else {
                P::gr_write(p->R, p->bo, p->inj, HC, T, cs);
            }
        }
    }
    cs->wait_and_throw();
    return 0;
    NS_CATCH
}

}  // extern "C"
