// fused.cpp: an expert's gate | up or down for a prompt chunk's tokens, its 2-bit weights decoded straight into the
// matrix engine (joint_matrix on the XMX units) - no fp16 copy of the expert in memory. C[M, N] (f32) = X[M, K] (f16)
// . W[N, K]^T, W in IQ2_XXS (gate | up: gate's rows then up's, a slot's first two matrices) or Q2_K (down). A sub-group's lane n holds column n
// of B (16 consecutive k of weight row n) - the layout probed on Xe2 - so each lane decodes its own 16 weights.
// Measured alone, the weights cold (gate | up 4096 x 4096): B70 117 us vs 202 for expanding to fp16 + oneMKL at 64
// tokens, 135 / 214 at 128; down (4096 x 2048), 8 row tiles: B70 39 us at 64 tokens, 40 at 128, 55 at 256, B65 47 /
// 66 / 119 (oneMKL alone on the expanded weights: 18 / 25 / 90, 31 / 62 / 61).
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ns.h"
#include "ns_internal.hpp"
#include "ggml-common.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace jm = sycl::ext::oneapi::experimental::matrix;

namespace {
constexpr int TM = 8, TN = 16, TK = 16, SG = 16, NSG = 8;

// The 32 k from k of a weight row (k a multiple of 32), as two B columns of 16: the dequantizers' arithmetic
struct Iq2xxs {  // gate | up: a 32-group's 8 bytes - its 4 grid indices, the scale and the signs (iq_kernels' dq_iq2_xxs)
    static constexpr int BYTES = sizeof(block_iq2_xxs);
    static void decode(const void* row, int64_t k, sycl::half (&v)[2][16]) {
        const block_iq2_xxs* b = (const block_iq2_xxs*) row + k / 256;
        const sycl::uint2 q8 = *reinterpret_cast<const sycl::uint2*>(b->qs + 4 * ((k % 256) / 32));
        const uint32_t aux32 = q8.y();
        const float d = (float) b->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t s7 = (aux32 >> 7 * il) & 127;
            const uint32_t signs = s7 | ((sycl::popcount(s7) & 1u) << 7);
            const uint64_t g = iq2xxs_grid[(q8.x() >> 8 * il) & 0xff];
#pragma unroll
            for (int j = 0; j < 8; ++j) v[il / 2][(il % 2) * 8 + j] = sycl::half(d * (float) ((g >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f));
        }
    }
};
struct Q2k {  // down: 16 k share a scale byte and a 2-bit shift of 16 consecutive quant bytes (ggml's dequantize_row_q2_K)
    static constexpr int BYTES = sizeof(block_q2_K);
    static void decode(const void* row, int64_t k, sycl::half (&v)[2][16]) {
        const block_q2_K* b = (const block_q2_K*) row + k / 256;
        const int n = (k % 256) / 128, j = (k % 128) / 32;
        const float d = (float) b->dm[0], dm = (float) b->dm[1];
        const sycl::uint4* qp = reinterpret_cast<const sycl::uint4*>(b->qs + 32 * n);
        const uint16_t sc2 = *reinterpret_cast<const uint16_t*>(b->scales + 8 * n + 2 * j);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const sycl::uint4 qq = qp[hf];
            const uint32_t sc = (sc2 >> (8 * hf)) & 0xff;
            const float dl = d * (sc & 0xF), ml = dm * (sc >> 4);
#pragma unroll
            for (int l = 0; l < 16; ++l) v[hf][l] = sycl::half(dl * (float) ((qq[l / 4] >> (8 * (l % 4) + 2 * j)) & 3) - ml);
        }
    }
};

// RM row tiles (8 tokens each) a sub-group, 32 k a step: lane n decodes its weight row's 32 k into two B tiles
template <int RM, class Dec>
void fused(sycl::queue& q, const sycl::half* A, const uint8_t* W, float* out, int64_t M, int64_t N, int64_t K) {
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
        const uint8_t* wr = W + (n0 + lane) * (K / 256) * Dec::BYTES;
        for (int64_t k = 0; k < K; k += 32) {
            sycl::half v[2][16];
            Dec::decode(wr, k, v);
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                // lane n holds column n of B (16 consecutive k of weight row n): the layout probed on Xe2
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, TK, TN, jm::layout::ext_intel_packed> bm;
                jm::joint_matrix_fill(sg, bm, sycl::half(0));
                int e = 0;
                jm::joint_matrix_apply(sg, bm, [&](sycl::half& h) { h = v[hh][e++]; });
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

// M a multiple of 32 (of 64 above 64: those take 8 row tiles a sub-group, in the large register file)
template <class Dec>
int run(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K, const char* name) {
    const bool wide = M > 64;
    if (M % (wide ? 64 : 32) != 0 || N % (TN * NSG) != 0 || K % 256 != 0) return ns_fail(name);
    const sycl::half* A = (const sycl::half*) x;
    if (wide) fused<8, Dec>(g->q, A, (const uint8_t*) w, out, M, N, K);
    else fused<4, Dec>(g->q, A, (const uint8_t*) w, out, M, N, K);
    return 0;
}
}  // namespace

extern "C" {

int ns_moe_fused_gu(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K) {
    NS_TRY
    return run<Iq2xxs>(g, x, w, out, M, N, K, "ns_moe_fused_gu: shapes");
    NS_CATCH
}

int ns_moe_fused_down(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K) {
    NS_TRY
    return run<Q2k>(g, x, w, out, M, N, K, "ns_moe_fused_down: shapes");
    NS_CATCH
}

}  // extern "C"
