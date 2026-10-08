// mtp.cpp: Qwen3.8-Flash-Next's MTP draft layer on the Strata SYCL kernels - a port of the port's MtpDrafter
// (sycl/src/core/mtp.cpp at intel-arc-0.1.40: load, bind, record_front, record_rest, prefill, draft), argmax drafts.
//
// Cell i of the layer pairs the main model's final residual at position i with the token at position i + 1; its
// output predicts the token at i + 2 and its own residual feeds the next draft step. Its projections are Q8_0, its 512
// experts Strata's S2 blobs (all in VRAM), its attention dense over the last `window` cells of its own int8 K/V (in the
// session's state), its head the main head's rows for a token subset (draft_vocab.bin). Drafts only: the verify window
// decides every token. Differences from Strata, host-side: eager launches (no graphs), inputs copied in rather than read
// from mapped memory, outputs read back after each launch.
#include "qwen_internal.hpp"

#include <dpct/dpct.hpp>

#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace qw;
namespace SK = strata::kernels;

struct Mtp {
    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    std::vector<Tensor> tensors;
    uint8_t* dense = nullptr;
    uint8_t* experts = nullptr;
    uint8_t* dhead = nullptr;    // the main head's rows for draft_vocab.bin's tokens
    int32_t* dvocab = nullptr;   // subset index -> token id
    int64_t n_dvocab = 0;
    int dhead_type = -1;
    int embd_type = -1;
    const void* embd = nullptr;
    size_t embd_row = 0;
    int max_t = 0;
    int64_t window = 0, cap = 0, attn_floats = 0;
    void* arena = nullptr;
    // device
    int32_t *tok, *step, *pos, *row, *ident;
    float *Rin, *R, *emb, *en, *e2, *hn, *h2, *mixed, *inj, *inj2, *lo, *rs, *bo, *xn;
    uint8_t* xq;
    float *qfull, *qcur, *kcur, *vcur, *attn, *attn32, *attn_scratch;
    float *logits, *w, *shared, *parts, *y, *sample;
    int32_t *ids, *hit_slot, *hit_dst, *hit_count, *out_ids;
    unsigned long long* grp_ptr;
    int32_t *grp_start, *grp_counts;
    uint8_t* hit_xq;
    float* hit_xs;
    uint8_t* hit_scratch;
    float* sh_scratch;
    uint16_t* x_bf16;
    float* probs;
    uint8_t *arg_scratch, *top_scratch;
    float* dummy_inj;
    float* head_logits;
    int32_t* out_dev;   // the drafts (device), read back after each launch
    float* prob_dev;
    // pinned staging
    int32_t* h_in = nullptr;   // tok | step | pos | row
    int32_t* h_out = nullptr;
    float* h_prob = nullptr;

    const Tensor* find(const char* name, const char* kind) const {
        for (const auto& t : tensors) if (t.name == name && t.kind == kind) return &t;
        return nullptr;
    }
    const float* f32(const char* n) const { const Tensor* t = find(n, "f32"); return t ? (const float*) (dense + t->off) : nullptr; }
    const uint16_t* bf16(const char* n) const { const Tensor* t = find(n, "bf16"); return t ? (const uint16_t*) (dense + t->off) : nullptr; }
    const void* q8(const char* n) const { const Tensor* t = find(n, "q8_0"); return t ? (const void*) (dense + t->off) : nullptr; }
};

namespace {

constexpr int Q8_0 = 8;
constexpr int64_t HCN = HC * N;

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::fseek(f, 0, SEEK_END);
    const long long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? (size_t) n : 0);
    const bool ok = n >= 0 && (n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n);
    std::fclose(f);
    return ok;
}

void carve(Mtp* m, Bump& b) {
    const uint64_t T = (uint64_t) m->max_t, R2 = 2 * T;
    m->tok = b.take<int32_t>(T); m->step = b.take<int32_t>(R2 * 4); m->pos = b.take<int32_t>(R2 * NH); m->row = b.take<int32_t>(4);
    m->ident = b.take<int32_t>(T * (uint64_t) m->cap);
    m->Rin = b.take<float>(T * HCN); m->R = b.take<float>(T * HCN);
    m->emb = b.take<float>(T * N); m->en = b.take<float>(T * N); m->e2 = b.take<float>(T * N);
    m->hn = b.take<float>(T * HCN); m->h2 = b.take<float>(T * HCN);
    m->mixed = b.take<float>(T * N); m->inj = b.take<float>(T * HC); m->inj2 = b.take<float>(T * HC);
    m->lo = b.take<float>(T * HC_LR); m->rs = b.take<float>(T * HC); m->bo = b.take<float>(T * N);
    m->xn = b.take<float>(T * HCN);
    m->xq = b.take<uint8_t>(SK::native_q8_1_bytes((int) (NH * HD), 8));
    m->qfull = b.take<float>(T * NH * 2 * HD); m->qcur = b.take<float>(T * NH * HD);
    m->kcur = b.take<float>(T * NKV * HD); m->vcur = b.take<float>(T * NKV * HD);
    m->attn = b.take<float>(T * NH * HD); m->attn32 = b.take<float>(T * NH * HD);
    m->attn_scratch = b.take<float>((uint64_t) m->attn_floats);
    m->logits = b.take<float>(T * NE); m->w = b.take<float>(T * K); m->ids = b.take<int32_t>(T * K);
    m->shared = b.take<float>(T * N); m->parts = b.take<float>(T * K * N); m->y = b.take<float>(T * N);
    m->sample = b.take<float>(T * N);
    m->hit_slot = b.take<int32_t>(T * K); m->hit_dst = b.take<int32_t>(T * K); m->hit_count = b.take<int32_t>(4);
    m->grp_ptr = b.take<unsigned long long>(T * K); m->grp_start = b.take<int32_t>(T * K + 1);
    m->grp_counts = b.take<int32_t>(4);
    m->hit_xq = b.take<uint8_t>(T * (N / 32) * 34); m->hit_xs = b.take<float>(T * (N / 32));
    m->hit_scratch = b.take<uint8_t>(SK::moe_hit_grouped_scratch_bytes((int64_t) (T * K), N, NFF));
    m->sh_scratch = (float*) b.take<uint8_t>(SK::shared_expert_scratch_bytes(NFF));
    m->x_bf16 = b.take<uint16_t>(N);
    m->out_ids = b.take<int32_t>(T + 4);
    m->probs = b.take<float>(T + 4);
    m->arg_scratch = b.take<uint8_t>(SK::argmax_rows_scratch_bytes((int) T));
    m->top_scratch = b.take<uint8_t>(SK::row_top_prob_scratch_bytes((int) T));
    m->dummy_inj = b.take<float>(HC);
    m->head_logits = b.take<float>(T * (uint64_t) m->n_dvocab);
    m->out_dev = b.take<int32_t>(16);
    m->prob_dev = b.take<float>(16);
}

// the layer's front for T rows at step rows [row0, +T): the embedding, the fc projections, the attention
// hyper-connection read (R, inj, mixed) and the K/V appended (Strata: record_front)
void front(ns_qw* w, ns_qw_state* st, Mtp* m, int T, int row0) {
    sycl::queue* cs = w->q;
    const SK::QsaShapes& s = w->s;
    const int32_t* step = m->step + row0 * 4;
    const int32_t* pos = m->pos + row0 * NH;
    // ---- the two input branches
    SK::iq_embed_rows(m->embd_type, m->embd, m->embd_row, m->tok, T, N, m->emb, cs);
    SK::native_qsa_rms_norm_weighted(m->emb, m->f32("pre_fc_norm_embedding.weight"), m->en, (int) N, T, EPS, cs);
    SK::native_quantize_q8_1(m->en, m->xq, (int) N, T, cs);
    SK::native_mmvq(Q8_0, m->q8("fc_embedding.weight"), m->xq, m->e2, (int) N, (int) N, T, cs);
    SK::native_qsa_rms_norm_weighted(m->Rin, m->f32("pre_fc_norm_hidden.weight"), m->hn, (int) HCN, T, EPS, cs);
    for (int c0 = 0; c0 < T * HC; c0 += 8) {
        const int nc = (int) std::min<int64_t>(8, T * HC - c0);
        SK::native_quantize_q8_1(m->hn + (size_t) c0 * N, m->xq, (int) N, nc, cs);
        SK::native_mmvq(Q8_0, m->q8("fc_hidden.weight"), m->xq, m->h2 + (size_t) c0 * N, (int) N, (int) N, nc, cs);
    }
    SK::add_streams_broadcast(m->h2, m->e2, m->R, N, (int) HC, T, cs);
    // ---- the attention hyper-connection
    {
        SK::FusedGrArgs fa[SK::kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            fa[t].R = m->R + (size_t) t * HCN; fa[t].R_out = m->R + (size_t) t * HCN; fa[t].apply = false;
            fa[t].w_norm = m->f32("attn_hyper_connection.hc_norm.weight");
            fa[t].w_down = m->bf16("attn_hyper_connection.input_mix_weight_down.weight");
            fa[t].w_up = m->bf16("attn_hyper_connection.input_mix_weight_up.weight");
            fa[t].w_inject = m->bf16("attn_hyper_connection.block_inject_weight.weight");
            fa[t].eps = EPS; fa[t].lo = m->lo + t * HC_LR; fa[t].rs = m->rs + t * HC;
            fa[t].inject_out = m->inj + t * HC; fa[t].mixed = m->mixed + t * N;
        }
        SK::fused_gr_read_multi(fa, T, m->xn, cs);
    }
    // ---- attention: K/V into the layer's own cache
    SK::native_quantize_q8_1(m->mixed, m->xq, (int) N, T, cs);
    SK::native_mmvq(Q8_0, m->q8("self_attn.k_proj.weight"), m->xq, m->kcur, (int) N, (int) (NKV * HD), T, cs);
    SK::native_mmvq(Q8_0, m->q8("self_attn.v_proj.weight"), m->xq, m->vcur, (int) N, (int) (NKV * HD), T, cs);
    const bool fuse_nr = SK::native_rope_enabled() && SK::native_norm_rope_usable((int) HD, (int) s.n_rot);
    if (T == 1 && fuse_nr) {
        SK::native_qsa_rms_norm_rope(m->kcur, (int) HD, m->f32("self_attn.k_norm.weight"), m->kcur, (int) NKV, (int) HD,
                                     (int) s.n_rot, EPS, SK::rope_scaling(), pos, cs);
    } else {
        SK::native_qsa_rms_norm_weighted(m->kcur, m->f32("self_attn.k_norm.weight"), m->kcur, (int) HD, (int) (T * NKV), EPS, cs);
        for (int t = 0; t < T; ++t) {
            float* kc = m->kcur + t * NKV * HD;
            SK::native_rope_apply(kc, kc, (int) NKV, (int) HD, (int) s.n_rot, SK::rope_scaling(), pos + t * NH, cs);
        }
    }
    const SK::KvHostPools none{};
    SK::kv_append_q8_steps(st->mtp.k_q, st->mtp.v_q, st->mtp.k_scale, st->mtp.v_scale, st->mtp.page_table, step, 4,
                           m->kcur, m->vcur, (int) (NKV * HD), T, s, cs, &none);
}

// the rest for row 0 at step row `step_row`: the attention from the q projection, the MLP, the final mixer, the head,
// the draft and its probability (Strata: record_rest, argmax drafts)
void rest(ns_qw* w, ns_qw_state* st, Mtp* m, int step_row) {
    sycl::queue* cs = w->q;
    const SK::QsaShapes& s = w->s;
    const SK::GrShapes gs{N, HC, HC_LR};
    const int T = 1;
    const int32_t* step = m->step + step_row * 4;
    const int32_t* pos = m->pos + step_row * NH;
    const bool fuse_nr = SK::native_rope_enabled() && SK::native_norm_rope_usable((int) HD, (int) s.n_rot);
    // ---- dense attention over the window's cells
    SK::native_mmvq(Q8_0, m->q8("self_attn.q_proj.weight"), m->xq, m->qfull, (int) N, (int) (NH * 2 * HD), T, cs);
    if (fuse_nr) {
        SK::native_qsa_rms_norm_rope(m->qfull, (int) (2 * HD), m->f32("self_attn.q_norm.weight"), m->qcur, (int) (T * NH),
                                     (int) HD, (int) s.n_rot, EPS, SK::rope_scaling(), pos, cs);
    } else {
        for (int64_t r = 0; r < T * NH; ++r) cs->memcpy(m->qcur + r * HD, m->qfull + r * 2 * HD, (size_t) HD * 4);
        SK::native_qsa_rms_norm_weighted(m->qcur, m->f32("self_attn.q_norm.weight"), m->qcur, (int) HD, (int) (T * NH), EPS, cs);
        SK::native_rope_apply(m->qcur, m->qcur, (int) (T * NH), (int) HD, (int) s.n_rot, SK::rope_scaling(), pos, cs);
    }
    SK::QsaAttnPools pools;
    pools.page_table = st->mtp.page_table;
    pools.k_q = st->mtp.k_q; pools.v_q = st->mtp.v_q; pools.k_scale = st->mtp.k_scale; pools.v_scale = st->mtp.v_scale;
    if (m->window > 0) SK::window_ids(const_cast<int32_t*>(step), T, (int) m->window, m->ident, m->cap, cs);
    SK::qsa_decode_attn_batch(m->qcur, pools, m->ident, step, m->cap, s, m->attn_scratch, m->attn, T, cs);
    SK::native_qsa_gate_apply(m->attn, m->qfull, m->attn32, (int) (T * NH), (int) HD, cs);
    SK::native_quantize_q8_1(m->attn32, m->xq, (int) (NH * HD), T, cs);
    SK::native_mmvq(Q8_0, m->q8("self_attn.o_proj.weight"), m->xq, m->bo, (int) (NH * HD), (int) N, T, cs);
    // ---- the MLP hyper-connection (the attention write folded in)
    {
        SK::FusedGrArgs fa[SK::kFusedGrMaxT];
        fa[0].R = m->R; fa[0].R_out = m->R; fa[0].apply = true;
        fa[0].bo_prev = m->bo; fa[0].inj_prev = m->inj;
        fa[0].w_norm = m->f32("mlp_hyper_connection.hc_norm.weight");
        fa[0].w_down = m->bf16("mlp_hyper_connection.input_mix_weight_down.weight");
        fa[0].w_up = m->bf16("mlp_hyper_connection.input_mix_weight_up.weight");
        fa[0].w_inject = m->bf16("mlp_hyper_connection.block_inject_weight.weight");
        fa[0].eps = EPS; fa[0].lo = m->lo; fa[0].rs = m->rs;
        fa[0].inject_out = m->inj2; fa[0].mixed = m->mixed;
        SK::fused_gr_read_multi(fa, T, m->xn, cs);
    }
    // ---- MoE: the router, the 512 resident experts (S2 blobs), the shared expert, the combine, the write
    SK::NativeSharedWeights nsw;
    nsw.gate_type = Q8_0; nsw.gate_data = m->q8("mlp.shared_expert.gate_proj.weight");
    nsw.up_type = Q8_0; nsw.up_data = m->q8("mlp.shared_expert.up_proj.weight");
    nsw.down_type = Q8_0; nsw.down_data = m->q8("mlp.shared_expert.down_proj.weight");
    nsw.q8_1 = m->xq;
    const SK::SForm none{};
    if (!SK::shared_expert_native_bf16_enabled()) SK::f32_to_bf16_bulk(m->mixed, m->x_bf16, N, cs);
    SK::bf16_gemv_fp32_mmvf(m->mixed, m->bf16("mlp.gate.weight"), m->logits, (int) N, (int) NE, cs);
    if (SK::native_router_enabled()) SK::native_router_top10(m->logits, m->ids, m->w, cs);
    SK::moe_group_resident(m->ids, (int) (T * K), (int) K, m->experts, (int64_t) SK::cpu::BLOB, m->grp_ptr, m->grp_start,
                           m->grp_counts, m->hit_dst, m->hit_slot, cs);
    SK::quantize_q8_0_scaled(m->mixed, m->hit_xq, m->hit_xs, (int64_t) T * N, cs);
    SK::moe_grouped_s2(m->grp_ptr, m->grp_start, m->grp_counts, m->hit_dst, m->hit_slot, (int64_t) T * K, (int64_t) T * K,
                       m->hit_xq, m->hit_xs, m->hit_scratch, m->parts, cs);
    SK::shared_expert(nullptr, nullptr, m->x_bf16, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                      nullptr, nullptr, nullptr, m->bf16("mlp.shared_expert_gate.weight"), m->sh_scratch, m->shared, N, NFF, 32,
                      cs, m->mixed, &nsw);
    if (SK::native_moe_combine_enabled()) SK::native_moe_combine(m->parts, m->w, m->shared, m->y, N, K, cs);
    else SK::moe_combine(m->parts, m->w, m->shared, m->y, N, K, cs);
    SK::gr_write_multi(m->R, m->y, m->inj2, gs, m->R, T, cs);
    // ---- the final mixer and the draft head
    {
        SK::FusedGrArgs fa[SK::kFusedGrMaxT];
        fa[0].R = m->R; fa[0].R_out = m->R; fa[0].apply = false;
        fa[0].w_norm = m->f32("hyper_connection_mixer.hc_norm.weight");
        fa[0].w_down = m->bf16("hyper_connection_mixer.input_mix_weight_down.weight");
        fa[0].w_up = m->bf16("hyper_connection_mixer.input_mix_weight_up.weight");
        fa[0].eps = EPS; fa[0].lo = m->lo; fa[0].rs = m->rs; fa[0].mixed = m->sample;
        SK::fused_gr_read_multi(fa, T, m->xn, cs);
    }
    SK::native_quantize_q8_1(m->sample, m->xq, (int) N, T, cs);
    SK::native_mmvq(m->dhead_type, m->dhead, m->xq, m->head_logits, (int) N, (int) m->n_dvocab, T, cs);
    if (SK::argmax_rows_wanted()) {
        SK::argmax_rows(m->head_logits, T, (int) m->n_dvocab, m->arg_scratch, m->out_ids, cs);
    } else {
        SK::SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        SK::sample_tokens(m->head_logits, T, (int) m->n_dvocab, nullptr, 0, sp, m->out_ids, cs);
    }
    if (SK::multi_block_head_ops()) SK::row_top_prob_split(m->head_logits, T, (int) m->n_dvocab, m->out_ids, m->probs, m->top_scratch, cs);
    else SK::row_top_prob(m->head_logits, T, (int) m->n_dvocab, m->out_ids, m->probs, cs);
    SK::map_ids(m->out_ids, m->dvocab, T, cs);
}

// the round on the device: its inputs from pinned memory, the window's final residual rows, the catch-up over T cells,
// the rest for row a (staged in row_), the first draft
void round_body(ns_qw* w, ns_qw_state* st, Mtp* m, int T) {
    sycl::queue* cs = w->q;
    const int max_t = m->max_t;
    const int ra = 2 * max_t - 1;
    int32_t* h_tok = m->h_in;
    int32_t* h_step = h_tok + max_t;
    int32_t* h_pos = h_step + 2 * max_t * 4;
    int32_t* h_row = h_pos + 2 * max_t * NH;
        SK::copy_i32_from_mapped(m->tok, h_tok, T, cs);
        SK::copy_i32_from_mapped(m->step, h_step, (int64_t) 2 * max_t * 4, cs);
        SK::copy_i32_from_mapped(m->pos, h_pos, (int64_t) 2 * max_t * NH, cs);
        SK::copy_i32_from_mapped(m->row, h_row, 2, cs);
        cs->memcpy(m->Rin, w->R, (size_t) T * HCN * 4);
        front(w, st, m, T, 0);
        if (T > 1) {
            SK::copy_row_to_first(m->row, m->R, HCN, m->inj, HC, m->mixed, N, cs);
            SK::native_quantize_q8_1(m->mixed, m->xq, (int) N, 1, cs);
        }
        rest(w, st, m, ra);
        SK::mtp_select(m->R, HCN, m->out_ids, m->row + 1, m->Rin, m->tok, m->out_dev, 0, cs, m->probs, m->prob_dev);
}

// chain step j on the device: one row from the previous step's residual and token
void step_body(ns_qw* w, ns_qw_state* st, Mtp* m, int j) {
    sycl::queue* cs = w->q;
    const int row = m->max_t + j - 1;
            front(w, st, m, 1, row);
            rest(w, st, m, row);
            SK::mtp_select(m->R, HCN, m->out_ids, m->row + 1, m->Rin, m->tok, m->out_dev, j, cs, m->probs, m->prob_dev);
}

void put_step(int32_t* step, int32_t* pos, int row, int64_t cell) {
    step[row * 4 + 0] = (int32_t) cell;
    step[row * 4 + 1] = (int32_t) (cell + 1);
    step[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
    step[row * 4 + 3] = (int32_t) (cell + 1);
    for (int64_t h = 0; h < NH; ++h) pos[row * NH + h] = (int32_t) cell;
}

void free_mtp(ns_qw* w) {
    Mtp* m = w->mtp;
    if (m == nullptr) return;
    auto& ctx = w->g->ctx;
    for (void* p : {(void*) m->dense, (void*) m->experts, (void*) m->dhead, (void*) m->dvocab, m->arena})
        if (p) sycl::free(p, ctx);
    for (void* p : {(void*) m->h_in, (void*) m->h_out, (void*) m->h_prob})
        if (p) sycl::free(p, ctx);
    delete m;
    w->mtp = nullptr;
}

}  // namespace

namespace qw {
void mtp_free(ns_qw* w) {
    try { free_mtp(w); } catch (...) {}
}
void mtp_warm(ns_qw* w, ns_qw_state* st) {
    Mtp* m = w->mtp;
    if (m == nullptr || st->mtp.k_q == nullptr) return;
    for (int T = 1; T <= m->max_t; ++T) {
        run_graph(*w->q, st->mtp_round[T], [&] { round_body(w, st, m, T); }, false);
        run_graph(*w->q, st->mtp_pf[T], [&] { front(w, st, m, T, 0); }, false);
    }
    for (int j = 1; j < m->max_t; ++j) run_graph(*w->q, st->mtp_step[j], [&] { step_body(w, st, m, j); }, false);
}
}  // namespace qw

extern "C" {

int ns_qw_mtp_load(ns_qw* w, const char* rt_dir, int max_t, int64_t window, int embd_type, const void* embd, size_t embd_row) {
    NS_TRY
    if (w->le != w->n_layer || w->E.out == nullptr) return ns_fail("qwen mtp: the draft layer goes on the stage with the head");
    if (max_t < 2 || max_t > MAXT) return ns_fail("qwen mtp: a window of 2.." + std::to_string(MAXT) + " tokens");
    if (w->mtp != nullptr) return ns_fail("qwen mtp: loaded already");
    auto* m = new Mtp();
    w->mtp = m;
    auto fail = [&](const std::string& why) { free_mtp(w); return ns_fail("qwen mtp: " + why); };
    const std::string dir(rt_dir);
    m->max_t = max_t;
    m->embd_type = embd_type;
    m->embd = embd;
    m->embd_row = embd_row;
    sycl::queue& q = w->g->q;
    auto& ctx = w->g->ctx;
    // ---- the index and the dense weights
    {
        std::ifstream idx(dir + "/dense.txt");
        if (!idx) return fail("cannot open " + dir + "/dense.txt (Strata's tools/mtp_rt.py writes it)");
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Mtp::Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) return fail("malformed dense.txt line: " + line);
            m->tensors.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(dir + "/dense.bin", blob)) return fail("cannot read dense.bin");
        m->dense = (uint8_t*) sycl::malloc_device(blob.size(), w->g->dev, ctx);
        if (m->dense == nullptr) return fail("the dense weights do not fit");
        q.memcpy(m->dense, blob.data(), blob.size()).wait();
    }
    const char* q8s[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                         "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                         "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : q8s) if (!m->q8(n)) return fail(std::string(n) + " is missing (q8_0)");
    const char* f32s[] = {"pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "attn_hyper_connection.hc_norm.weight",
                          "mlp_hyper_connection.hc_norm.weight", "hyper_connection_mixer.hc_norm.weight",
                          "self_attn.k_norm.weight", "self_attn.q_norm.weight"};
    for (const char* n : f32s) if (!m->f32(n)) return fail(std::string(n) + " is missing (f32)");
    const char* bf16s[] = {"attn_hyper_connection.input_mix_weight_down.weight", "attn_hyper_connection.input_mix_weight_up.weight",
                           "attn_hyper_connection.block_inject_weight.weight", "mlp_hyper_connection.input_mix_weight_down.weight",
                           "mlp_hyper_connection.input_mix_weight_up.weight", "mlp_hyper_connection.block_inject_weight.weight",
                           "hyper_connection_mixer.input_mix_weight_down.weight", "hyper_connection_mixer.input_mix_weight_up.weight",
                           "mlp.gate.weight", "mlp.shared_expert_gate.weight"};
    for (const char* n : bf16s) if (!m->bf16(n)) return fail(std::string(n) + " is missing (bf16)");
    // ---- the 512 routed experts, an S2 blob each
    {
        const uint64_t bytes = (uint64_t) NE * SK::cpu::BLOB;
        std::FILE* f = std::fopen((dir + "/experts.bin").c_str(), "rb");
        if (f == nullptr) return fail("cannot open experts.bin");
        m->experts = (uint8_t*) sycl::malloc_device(bytes, w->g->dev, ctx);
        if (m->experts == nullptr) { std::fclose(f); return fail("the 512 experts do not fit"); }
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (std::fread(chunk.data(), 1, (size_t) n, f) != (size_t) n) { std::fclose(f); return fail("experts.bin is truncated"); }
            q.memcpy(m->experts + off, chunk.data(), n).wait();
            off += n;
        }
        std::fclose(f);
    }
    // ---- the draft head: the main head's rows for draft_vocab.bin's tokens
    {
        std::vector<uint8_t> raw;
        if (!read_file(dir + "/draft_vocab.bin", raw) || raw.size() < 4 || raw.size() % 4 != 0) return fail("cannot read draft_vocab.bin");
        m->n_dvocab = (int64_t) (raw.size() / 4);
        const size_t hrow = (size_t) SK::iq_row_bytes(w->E.out_type, N);
        m->dvocab = (int32_t*) sycl::malloc_device(raw.size(), w->g->dev, ctx);
        m->dhead = (uint8_t*) sycl::malloc_device((size_t) m->n_dvocab * hrow, w->g->dev, ctx);
        if (m->dvocab == nullptr || m->dhead == nullptr) return fail("the draft head does not fit");
        q.memcpy(m->dvocab, raw.data(), raw.size()).wait();
        SK::gather_rows((const uint8_t*) w->E.out, (int64_t) hrow, m->dvocab, m->n_dvocab, m->dhead, w->q);
        q.wait();
        m->dhead_type = w->E.out_type;
    }
    // ---- buffers
    m->window = window > 0 && window < w->max_cells ? window : 0;
    m->cap = (((m->window > 0 ? m->window : w->max_cells) + 63) / 64) * 64;
    m->attn_floats = (int64_t) SK::qsa_decode_attn_scratch_floats(m->cap, w->s);
    Bump count;
    carve(m, count);
    m->arena = sycl::malloc_device(count.used + 256, w->g->dev, ctx);
    if (m->arena == nullptr) return fail("its buffers (" + std::to_string(count.used >> 20) + " MiB) do not fit");
    q.memset(m->arena, 0, count.used).wait();
    Bump real{(uint8_t*) m->arena};
    carve(m, real);
    {
        std::vector<int32_t> id((size_t) max_t * (size_t) m->cap);
        for (int t = 0; t < max_t; ++t)
            for (int64_t i = 0; i < m->cap; ++i) id[(size_t) t * (size_t) m->cap + (size_t) i] = (int32_t) i;
        q.memcpy(m->ident, id.data(), id.size() * 4).wait();
    }
    const size_t in_ints = (size_t) max_t * (1 + 2 * 4 + 2 * NH) + 4;
    m->h_in = (int32_t*) sycl::malloc_host(in_ints * 4, ctx);
    m->h_out = (int32_t*) sycl::malloc_host(64, ctx);
    m->h_prob = (float*) sycl::malloc_host(64, ctx);
    if (!m->h_in || !m->h_out || !m->h_prob) return fail("pinned staging");
    std::fprintf(stderr, "qwen mtp: draft layer loaded (%lld draft-head tokens, window %lld cells)\n", (long long) m->n_dvocab,
                 (long long) m->window);
    return 0;
    NS_CATCH
}

int ns_qw_mtp_prefill(ns_qw* w, ns_qw_state* st, int64_t n, int64_t cell0, const int32_t* next_tokens) {
    NS_TRY
    Mtp* m = w->mtp;
    if (m == nullptr || st->mtp.k_q == nullptr) return ns_fail("qwen mtp: no draft layer (or a session made before it)");
    if (n > prefill_cap(w)) return ns_fail("qwen mtp: the prompt rows are not the prompt path's");
    if (cell0 + n > st->max_cells) return ns_fail("qwen mtp: past the session's context");
    sycl::queue* cs = w->q;
    // the prompt path's final residual rows: its R (this stage's last layer written)
    const float* Rp = prefill_rows(w);
    int32_t* h_tok = m->h_in;
    int32_t* h_step = h_tok + m->max_t;
    int32_t* h_pos = h_step + 2 * m->max_t * 4;
    for (int64_t c = 0; c < n; c += m->max_t) {
        const int T = (int) std::min<int64_t>(m->max_t, n - c);
        for (int t = 0; t < T; ++t) {
            h_tok[t] = next_tokens[c + t];
            put_step(h_step, h_pos, t, cell0 + c + t);
        }
        cs->memcpy(m->tok, h_tok, (size_t) T * 4);
        cs->memcpy(m->step, h_step, (size_t) T * 16);
        cs->memcpy(m->pos, h_pos, (size_t) T * NH * 4);
        cs->memcpy(m->Rin, Rp + (size_t) c * HCN, (size_t) T * HCN * 4);
        run_graph(*cs, st->mtp_pf[T], [&] { front(w, st, m, T, 0); });
        cs->wait_and_throw();   // (the staging is reused by the next group)
    }
    return 0;
    NS_CATCH
}

int ns_qw_mtp_draft(ns_qw* w, ns_qw_state* st, const int32_t* tokens, int64_t p, int a, int max_drafts, float min_p,
                    int32_t* drafts, float* probs, int* n_out) {
    NS_TRY
    Mtp* m = w->mtp;
    *n_out = 0;
    if (m == nullptr || st->mtp.k_q == nullptr) return ns_fail("qwen mtp: no draft layer (or a session made before it)");
    if (a < 0 || a >= m->max_t) return ns_fail("qwen mtp: draft arguments out of range");
    const int T = a + 1;   // the catch-up covers the accepted rows (#783 PR-i: the rejected ones are never read)
    const int max_steps = std::min(m->max_t - 1, max_drafts);
    if (max_steps < 1) return 0;
    if (p + a + max_steps > st->max_cells) return 0;   // no room ahead: no drafts
    sycl::queue* cs = w->q;
    const int ra = 2 * m->max_t - 1;
    int32_t* h_tok = m->h_in;
    int32_t* h_step = h_tok + m->max_t;
    int32_t* h_pos = h_step + 2 * m->max_t * 4;
    int32_t* h_row = h_pos + 2 * m->max_t * NH;
    for (int t = 0; t < T; ++t) {
        h_tok[t] = tokens[t];
        put_step(h_step, h_pos, t, p + t);
    }
    put_step(h_step, h_pos, ra, p + a);   // draft 0's cell
    for (int j = 1; j < max_steps; ++j) put_step(h_step, h_pos, m->max_t + j - 1, p + a + j);
    h_row[0] = a;
    h_row[1] = 0;
    // ---- the round: its inputs from pinned memory, the window's final residual rows (this stage ran it: its R), the
    // catch-up's front for T cells, the rest for row a (on row 0), the draft
    const int max_t = m->max_t;
    run_graph(*cs, st->mtp_round[T], [&] { round_body(w, st, m, T); });
    cs->memcpy(m->h_out, m->out_dev, 4);
    cs->memcpy(m->h_prob, m->prob_dev, 4);
    cs->wait_and_throw();
    drafts[0] = m->h_out[0];
    probs[0] = m->h_prob[0];
    int n = 1;
    // ---- the chain, while the last draft is likely enough to be verified
    for (int j = 1; j < max_steps && probs[j - 1] >= min_p; ++j) {
        const int row = m->max_t + j - 1;
        run_graph(*cs, st->mtp_step[j], [&] { step_body(w, st, m, j); });
        cs->memcpy(m->h_out + j, m->out_dev + j, 4);
        cs->memcpy(m->h_prob + j, m->prob_dev + j, 4);
        cs->wait_and_throw();
        drafts[j] = m->h_out[j];
        probs[j] = m->h_prob[j];
        ++n;
    }
    *n_out = n;
    return 0;
    NS_CATCH
}

}  // extern "C"
