// fused.cpp: an expert's gate | up for a prompt chunk's tokens, its 2-bit weights decoded straight into the matrix
// engine (joint_matrix on the XMX units) - no fp16 copy of the expert in memory. C[M, N] (f32) = X[M, K] (f16) .
// W[N, K]^T, W in IQ2_XXS (gate's rows then up's: a slot's first two matrices). A sub-group's lane n holds column n
// of B (16 consecutive k of weight row n) - the layout probed on Xe2 - so each lane decodes its own 16 weights.
// Measured alone against expanding to fp16 + oneMKL (scratchpad xmx/fused.cpp): 2x faster at 64 tokens, even at
// 128-256: the engine uses it for the experts with few tokens.
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ns.h"
#include "ns_internal.hpp"
#include "ggml-common.h"

namespace jm = sycl::ext::oneapi::experimental::matrix;

namespace {
constexpr int TM = 8, TN = 16, TK = 16, SG = 16, RM = 4, NSG = 8;

// 8 values of group il (0..3) of 32-group ib of a block - the dequantizer's arithmetic (iq_kernels' dq_iq2_xxs)
inline void dq8(const block_iq2_xxs* b, int ib, int il, float* v) {
    const uint16_t* q2 = b->qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) b->d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint32_t s7 = (aux32 >> 7 * il) & 127;
    const uint32_t signs = s7 | ((sycl::popcount(s7) & 1u) << 7);
    const uint64_t g = iq2xxs_grid[aux8[il]];
    for (int j = 0; j < 8; ++j) v[j] = d * (float) ((g >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
}
}  // namespace

extern "C" {

int ns_moe_fused_gu(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K) {
    NS_TRY
    if (M % (RM * TM) != 0 || N % (TN * NSG) != 0 || K % 256 != 0) return ns_fail("ns_moe_fused_gu: shapes");
    const sycl::half* A = (const sycl::half*) x;
    const block_iq2_xxs* W = (const block_iq2_xxs*) w;
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / (RM * TM), N / (TN * NSG) * NSG * SG), sycl::range<2>(1, NSG * SG)),
                      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = sg.get_local_id()[0];
        const int64_t m0 = it.get_group(0) * RM * TM;
        const int64_t n0 = (it.get_group(1) * NSG + it.get_local_id(1) / SG) * TN;
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, TN> acc[RM];
        for (int i = 0; i < RM; ++i) jm::joint_matrix_fill(sg, acc[i], 0.0f);
        auto gA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(A);
        const block_iq2_xxs* wr = W + (n0 + lane) * (K / 256);
        for (int64_t k = 0; k < K; k += TK) {
            const block_iq2_xxs* b = wr + k / 256;
            const int ib = (k % 256) / 32, il = (k % 32) / 8;
            float v[16];
            dq8(b, ib, il, v);
            dq8(b, ib, il + 1, v + 8);
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, TK, TN, jm::layout::ext_intel_packed> bm;
            jm::joint_matrix_fill(sg, bm, sycl::half(0));
            int e = 0;
            jm::joint_matrix_apply(sg, bm, [&](sycl::half& h) { h = sycl::half(v[e++]); });
            for (int i = 0; i < RM; ++i) {
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, TM, TK, jm::layout::row_major> a;
                jm::joint_matrix_load(sg, a, gA + (m0 + i * TM) * K + k, K);
                jm::joint_matrix_mad(sg, acc[i], a, bm, acc[i]);
            }
        }
        for (int i = 0; i < RM; ++i)
            jm::joint_matrix_store(sg, acc[i],
                                   sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(out + (m0 + i * TM) * N + n0),
                                   N, jm::layout::row_major);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
