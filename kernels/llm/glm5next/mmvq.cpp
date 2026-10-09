// mmvq.cpp: decode-width matrix-vector products straight from the stored blocks (1..8 tokens), through the
// imported kernels: activations quantized once to Q8_1, then each weight type's dot product (ggml's arithmetic).
#include "ns.h"
#include "glm.h"
#include "ns_internal.hpp"

#include <cstdlib>

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

extern "C" {

size_t ns_q8_1_bytes(int64_t n_in, int64_t ncols) { return (size_t) (n_in / 32) * 36 * (size_t) ncols; }

int ns_mmvq_supported(int type) {
    return strata::kernels::iq_supported(type) || strata::kernels::native_mmvq_supported(type);
}

// x [ncols, n_in] float32 -> q8_1 blocks (column j's at j * n_in / 32)
int ns_quantize_q8_1(ns_gpu* g, const float* x, void* q8_1, int64_t n_in, int64_t ncols) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_quantize_q8_1: n_in a multiple of 32");
    strata::kernels::quantize_q8_1_rows(x, ncols, n_in, q8_1, &g->q);
    return 0;
    NS_CATCH
}

// y [ncols, n_out] = W [n_out, n_in] (ggml type, stored blocks) . x (q8_1, ncols columns); ncols 1..8
int ns_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (ncols < 1 || ncols > 8) return ns_fail("ns_mmvq: 1..8 columns");
    // the tuned dense kernels first (Q8_0, Q3_K-Q6_K, ...), the IQ family for the rest (IQ2_XXS, Q2_K, ...);
    // NS_MMVQ_IQ_FIRST=1 swaps the order (for comparisons)
    static const bool iq_first = getenv("NS_MMVQ_IQ_FIRST") && getenv("NS_MMVQ_IQ_FIRST")[0] == '1';
    const bool native = strata::kernels::native_mmvq_supported(type), iq = strata::kernels::iq_supported(type);
    if (native && !(iq_first && iq)) {
        strata::kernels::native_mmvq(type, w, x_q8_1, y, (int) n_in, (int) n_out, (int) ncols, &g->q);
    } else if (iq) {
        strata::kernels::iq_mmvq(type, w, x_q8_1, y, (int) n_in, (int) n_out, (int) ncols, &g->q);
    } else {
        return ns_fail("ns_mmvq: ggml type " + std::to_string(type) + " has no decode kernel");
    }
    return 0;
    NS_CATCH
}

// ---- a layer's routed experts in two launches (the imported grouped kernels): gate+up for every entry, then
// SwiGLU (clamped at `limit` when > 0) quantized, then down. grp_ptr [groups] the experts' [gate | up | down]
// blobs, grp_start [groups + 1] their entries' ranges, n_groups (one int), ent_tok [entries] each entry's token
// row of x_q8_1, ent_dst [entries] its output row of out [., n_embd]; all device memory.
int ns_moe_grouped_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    return strata::kernels::native_expert_supported(gu_type, d_type, n_embd, n_ff) ? 1 : 0;
}

size_t ns_moe_scratch_bytes(int64_t entries, int64_t n_ff) { return strata::kernels::native_expert_scratch_bytes(entries, n_ff); }

int ns_moe_grouped(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t groups, int64_t entries, const void* x_q8_1,
                   void* scratch, float* out, float limit, int lanes, int phase) {
    NS_TRY
    if (!strata::kernels::native_expert_supported(gu_type, d_type, n_embd, n_ff))
        return ns_fail("ns_moe_grouped: types " + std::to_string(gu_type) + "/" + std::to_string(d_type) + " not supported");
    strata::kernels::native_expert_set_swiglu_limit(limit);
    strata::kernels::native_expert_set_lanes(lanes);
    strata::kernels::native_expert_set_phase(phase);
    const auto L = strata::kernels::native_expert_layout(gu_type, d_type, n_embd, n_ff);
    strata::kernels::native_expert_grouped(L, (const unsigned long long*) grp_ptr, grp_start, n_groups, ent_dst, ent_tok, groups, entries, x_q8_1,
                                           scratch, out, &g->q);
    return 0;
    NS_CATCH
}

// n values of a stored expert matrix (n a multiple of 256) expanded to fp16, for the prompt path's half GEMMs
int ns_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst) {
    NS_TRY
    if (!strata::kernels::iq_supported(type) || n % 256 != 0) return ns_fail("ns_dequant_f16: ggml type " + std::to_string(type));
    strata::kernels::iq_dequant_f16(type, src, n, dst, &g->q);
    return 0;
    NS_CATCH
}

}  // extern "C"
