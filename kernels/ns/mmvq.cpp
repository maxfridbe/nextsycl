// mmvq.cpp: decode-width matrix-vector products straight from the stored blocks (1..8 tokens), through the
// imported kernels: activations quantized once to Q8_1, then each weight type's dot product (ggml's arithmetic).
#include "ns.h"
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

}  // extern "C"
