// fused.cpp: an expert's gate | up for a prompt chunk's tokens, its 2-bit weights decoded straight into the matrix
// engine (joint_matrix on the XMX units) - no fp16 copy of the expert in memory. C[M, N] (f32) = X[M, K] (f16) .
// W[N, K]^T, W in IQ2_XXS (gate's rows then up's: a slot's first two matrices). A sub-group's lane n holds column n
// of B (16 consecutive k of weight row n) - the layout probed on Xe2 - so each lane decodes its own 16 weights.
// Measured alone (4096 x 4096, B70) against expanding to fp16 + oneMKL: 98 us against 305 at 32 tokens, 110 / 199 at
// 64, 128 / 242 at 128, 199 / 267 at 256.
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ns.h"
#include "ns_internal.hpp"
#include "ggml-common.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace jm = sycl::ext::oneapi::experimental::matrix;

namespace {
constexpr int TM = 8, TN = 16, TK = 16, SG = 16, NSG = 8;

// RM row tiles (8 tokens each) a sub-group; 32 k a step: the 32-group's 8 bytes read once (its 4 grid indices, the
// scale and signs - the dequantizer's arithmetic, iq_kernels' dq_iq2_xxs), two B tiles from them
template <int RM>
void fused_gu(sycl::queue& q, const sycl::half* A, const block_iq2_xxs* W, float* out, int64_t M, int64_t N, int64_t K) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<(RM > 4 ? 256 : 128)>};
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / (RM * TM), N / (TN * NSG) * NSG * SG), sycl::range<2>(1, NSG * SG)), props,
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = sg.get_local_id()[0];
        const int64_t m0 = it.get_group(0) * RM * TM;
        const int64_t n0 = (it.get_group(1) * NSG + it.get_local_id(1) / SG) * TN;
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, TN> acc[RM];
        for (int i = 0; i < RM; ++i) jm::joint_matrix_fill(sg, acc[i], 0.0f);
        auto gA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(A);
        const block_iq2_xxs* wr = W + (n0 + lane) * (K / 256);
        for (int64_t k = 0; k < K; k += 32) {
            const block_iq2_xxs* b = wr + k / 256;
            const sycl::uint2 q8 = *reinterpret_cast<const sycl::uint2*>(b->qs + 4 * ((k % 256) / 32));
            const uint32_t aux32 = q8.y();
            const float d = (float) b->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                sycl::half v[16];
#pragma unroll
                for (int g2 = 0; g2 < 2; ++g2) {
                    const int il = hh * 2 + g2;
                    const uint32_t s7 = (aux32 >> 7 * il) & 127;
                    const uint32_t signs = s7 | ((sycl::popcount(s7) & 1u) << 7);
                    const uint64_t g = iq2xxs_grid[(q8.x() >> 8 * il) & 0xff];
#pragma unroll
                    for (int j = 0; j < 8; ++j) v[g2 * 8 + j] = sycl::half(d * (float) ((g >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f));
                }
                // lane n holds column n of B (16 consecutive k of weight row n): the layout probed on Xe2
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, TK, TN, jm::layout::ext_intel_packed> bm;
                jm::joint_matrix_fill(sg, bm, sycl::half(0));
                int e = 0;
                jm::joint_matrix_apply(sg, bm, [&](sycl::half& h) { h = v[e++]; });
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, TM, TK, jm::layout::row_major> a;
                    jm::joint_matrix_load(sg, a, gA + (m0 + i * TM) * K + k + hh * 16, K);
                    jm::joint_matrix_mad(sg, acc[i], a, bm, acc[i]);
                }
            }
        }
        for (int i = 0; i < RM; ++i)
            jm::joint_matrix_store(sg, acc[i],
                                   sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(out + (m0 + i * TM) * N + n0),
                                   N, jm::layout::row_major);
    });
}
}  // namespace

extern "C" {

// M a multiple of 32 (of 64 above 64: those take 8 row tiles a sub-group, in the large register file)
int ns_moe_fused_gu(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K) {
    NS_TRY
    const bool wide = M > 64;
    if (M % (wide ? 64 : 32) != 0 || N % (TN * NSG) != 0 || K % 256 != 0) return ns_fail("ns_moe_fused_gu: shapes");
    const sycl::half* A = (const sycl::half*) x;
    const block_iq2_xxs* W = (const block_iq2_xxs*) w;
    if (wide) fused_gu<8>(g->q, A, W, out, M, N, K);
    else fused_gu<4>(g->q, A, W, out, M, N, K);
    return 0;
    NS_CATCH
}

}  // extern "C"
