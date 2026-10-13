// idx.cpp: the deepseek4 engine's lightning indexer at decode widths (ds4.h: ns_ds4_idx_score_dec, the top-k's
// gather and map). The shared path (a float32 / fp16 GEMM writing every head's score, then a pass summing them, then
// one work-group's radix select a row) grows with the context three ways; at 1M a layer's indexer is 262K blocks:
//
//   score: the keys straight from their fp16 cache as XMX A tiles (8 blocks x 16 dims), the row's 64 heads' queries
//   packed once a work-group in local memory as B (VNNI), 8 blocks x 64 heads a sub-group; relu, the head weights and
//   the sum over the heads in registers - one float a block written, -inf past the blocks the row sees;
//   top-k in two levels: the shared radix select within chunks of C blocks (many work-groups), then over the chunks'
//   candidates; gather and map keep the positions ascending.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds4.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>
#include <cstdint>

namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
using half = sycl::half;
constexpr int DI = 128, HI = 64, SG = 16, M = 8, NSG = 8, WG = NSG * SG;

template <typename T>
inline auto lp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto gp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}
}  // namespace

extern "C" {

int ns_ds4_idx_score_dec(ns_gpu* g, const uint16_t* q16, const float* w, const uint16_t* cache, float* score, int64_t T, int64_t H, int64_t D,
                         int64_t n, int64_t ld, int64_t pos0) {
    NS_TRY
    if (H != HI || D != DI) return ns_fail("ns_ds4_idx_score_dec: 64 heads of 128");
    const half* qh = reinterpret_cast<const half*>(q16);
    const half* kc = reinterpret_cast<const half*>(cache);
    const int64_t per_wg = NSG * M;   // blocks a work-group
    const int64_t wgs = (ld + per_wg - 1) / per_wg;
    g->q.submit([&](sycl::handler& h) {
        // Q^T packed: [D/2][HI][2] halves (a word = dims 2k, 2k + 1 of a head)
        sycl::local_accessor<half, 1> qp(sycl::range<1>(DI * HI), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * wgs) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int64_t gi = it.get_group(0), t = gi / wgs, c0 = (gi % wgs) * per_wg + sgi * M;
            half* qs = qp.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int i = tid; i < DI * HI; i += WG) {
                const int hh = i / DI, k = i % DI;
                qs[(k / 2) * (HI * 2) + hh * 2 + (k & 1)] = qh[(t * HI + hh) * DI + k];
            }
            sycl::group_barrier(it.get_group());
            const int64_t nv = sycl::min<int64_t>(n, (pos0 + t + 1) / 4);
            float* out = score + t * ld;
            if (c0 >= ld) return;
            if (c0 >= nv) {   // nothing this sub-group's blocks see
                if (lane < M && c0 + lane < ld) out[c0 + lane] = -INFINITY;
                return;
            }
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> acc[HI / 16];
#pragma unroll
            for (int nt = 0; nt < HI / 16; ++nt) jm::joint_matrix_fill(sg, acc[nt], 0.0f);
            // rows past the cache's blocks read block n - 1 (in bounds; masked below)
            const int64_t cr = sycl::min<int64_t>(c0, n - M);
            const half* kb = kc + (cr < 0 ? 0 : cr) * DI;
#pragma unroll
            for (int k = 0; k < DI / 16; ++k) {
                jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> a;
                jm::joint_matrix_load(sg, a, gp(kb + k * 16), (size_t) DI);
#pragma unroll
                for (int nt = 0; nt < HI / 16; ++nt) {
                    jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> b;
                    jm::joint_matrix_load(sg, b, lp(qs + (k * 16 / 2) * (HI * 2) + nt * 16 * 2), (size_t) (HI * 2));
                    jm::joint_matrix_mad(sg, acc[nt], a, b, acc[nt]);
                }
            }
            // element e of a lane is (block e, head nt * 16 + lane): relu, x the head's weight, summed over the heads
            float s[M] = {};
#pragma unroll
            for (int nt = 0; nt < HI / 16; ++nt) {
                const float wh = w[t * HI + nt * 16 + lane];
                int e = 0;
                jm::joint_matrix_apply(sg, acc[nt], [&](float& x) { s[e] += sycl::fmax(x, 0.0f) * wh; ++e; });
            }
            const int64_t shift = c0 - (cr < 0 ? 0 : cr);   // rows of the tile before this sub-group's first block
#pragma unroll
            for (int e = 0; e < M; ++e) {
                const float v = sycl::reduce_over_group(sg, s[e], sycl::plus<float>());
                const int64_t j = c0 + e - shift;
                if (lane == 0 && e >= shift && j < ld) out[j] = j < nv ? v : -INFINITY;
            }
            if (lane == 0)
                for (int64_t e = M - shift; e < M; ++e)
                    if (c0 + e < ld) out[c0 + e] = -INFINITY;   // past the cache (when c0 > n - M)
        });
    });
    return 0;
    NS_CATCH
}

int ns_ds4_topk_gather(ns_gpu* g, const float* score, const int32_t* sel1, float* cval, int32_t* cpos, int64_t T, int64_t nch, int64_t C, int64_t K) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, nch * K), [=](sycl::id<2> id) {
        const int64_t t = id[0], i = id[1], c = i / K;
        const int64_t p = c * C + sel1[(t * nch + c) * K + i % K];
        cval[t * nch * K + i] = score[t * nch * C + p];
        cpos[t * nch * K + i] = (int32_t) p;
    });
    return 0;
    NS_CATCH
}

int ns_ds4_topk_map(ns_gpu* g, const int32_t* sel2, const int32_t* cpos, int32_t* sel, int64_t T, int64_t ncand, int64_t K) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, K), [=](sycl::id<2> id) {
        const int64_t t = id[0], k = id[1];
        sel[t * K + k] = cpos[t * ncand + sel2[t * K + k]];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
