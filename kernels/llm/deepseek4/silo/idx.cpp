// idx.cpp: the deepseek4 engine's lightning indexer at decode widths (ds4.h: ns_ds4_idx_score_dec, the top-k's
// gather and map). The shared path (a float32 / fp16 GEMM writing every head's score, then a pass summing them, then
// one work-group's radix select a row) grows with the context three ways; at 1M a layer's indexer is 262K blocks:
//
//   score: the keys straight from their fp16 cache as XMX A tiles (8 blocks x 16 dims), the row's 64 heads' queries
//   packed once a work-group in local memory as B (VNNI), 8 blocks x 64 heads a sub-group; relu, the head weights and
//   the sum over the heads in registers - one float a block written, -inf past the blocks the row sees;
//   prompt rows (ns_ds4_idx_score_pre): the same, R rows a work-group (their queries in local memory) and 256 blocks;
//   a sub-group's key tiles held in registers over its R rows; work-groups ordered chunk-major so the rows running
//   at once share the chunk's keys in L2 (the GEMM writing 64 heads' scores, read back to be summed, moved 64x the
//   output through memory - a 256K prompt's indexer was half its time);
//   top-k in two levels: the shared radix select within chunks of C blocks (many work-groups), then over the chunks'
//   candidates; gather and map keep the positions ascending.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds4.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>

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

// R rows a work-group, TB tiles of M blocks a sub-group, KT of them sharing each query tile loaded
template <int R, int TB, int KT, bool GQ>
void idx_score_pre(ns_gpu* g, const half* qh, const float* w, const half* kc, float* score, int64_t T, int64_t n, int64_t ld, int64_t pos0) {
    constexpr int64_t per_wg = NSG * TB * M;
    const int64_t chunks = (ld + per_wg - 1) / per_wg, rgs = (T + R - 1) / R;
    g->q.submit([&](sycl::handler& h) {
        // Q^T packed a row: [D/2][HI][2] halves
        sycl::local_accessor<half, 1> qp(sycl::range<1>(GQ ? 1 : R * DI * HI), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (chunks * rgs) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int64_t gi = it.get_group(0), t0 = (gi % rgs) * R, ch = gi / rgs;
            const int nr = (int) sycl::min<int64_t>(R, T - t0);
            // the blocks the group's last row sees
            const int64_t nv_max = sycl::min<int64_t>(n, (pos0 + t0 + nr) / 4);
            const int64_t wg0 = ch * per_wg;
            half* qs = qp.get_multi_ptr<sycl::access::decorated::no>().get();
            if (!GQ && wg0 < nv_max) {
                for (int i = tid; i < nr * DI * HI; i += WG) {
                    const int r = i / (DI * HI), j = i % (DI * HI), hh = j / DI, k = j % DI;
                    qs[r * DI * HI + (k / 2) * (HI * 2) + hh * 2 + (k & 1)] = qh[((t0 + r) * HI + hh) * DI + k];
                }
            }
            if (!GQ) sycl::group_barrier(it.get_group());
            // KT tiles of M blocks at a time, each query tile from local memory used by all KT
            for (int tb = 0; tb < TB; tb += KT) {
                const int64_t b0 = wg0 + (sgi * TB + tb) * M;
                if (b0 >= ld) break;
                if (b0 >= nv_max) {   // past what every row of the group sees
                    for (int r = 0; r < nr; ++r)
                        for (int j = 0; j < KT; ++j)
                            if (lane < M && b0 + j * M + lane < ld) score[(t0 + r) * ld + b0 + j * M + lane] = -INFINITY;
                    continue;
                }
                int64_t cr[KT];
#pragma unroll
                for (int j = 0; j < KT; ++j) cr[j] = sycl::min<int64_t>(b0 + j * M, n - M);
                for (int r = 0; r < nr; ++r) {
                    const int64_t t = t0 + r, nv = sycl::min<int64_t>(n, (pos0 + t + 1) / 4);
                    float* out = score + t * ld;
                    const half* qr = GQ ? qh + t * DI * HI : qs + r * DI * HI;
                    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> acc[KT][HI / 16];
#pragma unroll
                    for (int j = 0; j < KT; ++j)
#pragma unroll
                        for (int nt = 0; nt < HI / 16; ++nt) jm::joint_matrix_fill(sg, acc[j][nt], 0.0f);
#pragma unroll
                    for (int k = 0; k < DI / 16; ++k) {
                        jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> a[KT];
#pragma unroll
                        for (int j = 0; j < KT; ++j) jm::joint_matrix_load(sg, a[j], gp(kc + cr[j] * DI + k * 16), (size_t) DI);
#pragma unroll
                        for (int nt = 0; nt < HI / 16; ++nt) {
                            jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> b;
                            if constexpr (GQ) jm::joint_matrix_load(sg, b, gp(qr + (k * 16 / 2) * (HI * 2) + nt * 16 * 2), (size_t) (HI * 2));
                            else jm::joint_matrix_load(sg, b, lp(qr + (k * 16 / 2) * (HI * 2) + nt * 16 * 2), (size_t) (HI * 2));
#pragma unroll
                            for (int j = 0; j < KT; ++j) jm::joint_matrix_mad(sg, acc[j][nt], a[j], b, acc[j][nt]);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < KT; ++j) {
                        const int64_t c0 = b0 + j * M;
                        if (c0 >= ld) break;
                        const int64_t shift = c0 - cr[j];
                        float s[M] = {};
#pragma unroll
                        for (int nt = 0; nt < HI / 16; ++nt) {
                            const float wh = w[t * HI + nt * 16 + lane];
                            int e = 0;
                            jm::joint_matrix_apply(sg, acc[j][nt], [&](float& x) { s[e] += sycl::fmax(x, 0.0f) * wh; ++e; });
                        }
#pragma unroll
                        for (int e = 0; e < M; ++e) {
                            const float v = sycl::reduce_over_group(sg, s[e], sycl::plus<float>());
                            const int64_t jj = cr[j] + e;
                            if (lane == 0 && e >= shift && jj < ld) out[jj] = jj < nv ? v : -INFINITY;
                        }
                        if (lane == 0)
                            for (int64_t e = M - shift; e < M; ++e)
                                if (c0 + e < ld) out[c0 + e] = -INFINITY;
                    }
                }
            }
        });
    });
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

int ns_ds4_idx_pack(ns_gpu* g, const float* q, uint16_t* packed, int64_t T) {
    NS_TRY
    half* o = reinterpret_cast<half*>(packed);
    g->q.parallel_for(sycl::range<1>((size_t) (T * HI * DI)), [=](sycl::id<1> id) {
        const int64_t i = id[0], t = i / (HI * DI), j = i % (HI * DI), hh = j / DI, k = j % DI;
        o[t * HI * DI + (k / 2) * (HI * 2) + hh * 2 + (k & 1)] = (half) q[i];
    });
    return 0;
    NS_CATCH
}

int ns_ds4_idx_score_pre(ns_gpu* g, const uint16_t* q16, const float* w, const uint16_t* cache, float* score, int64_t T, int64_t H, int64_t D,
                         int64_t n, int64_t ld, int64_t pos0, int64_t packed) {
    NS_TRY
    if (H != HI || D != DI) return ns_fail("ns_ds4_idx_score_pre: 64 heads of 128");
    if (n < M) return ns_fail("ns_ds4_idx_score_pre: fewer blocks than a tile");
    const half* qh = reinterpret_cast<const half*>(q16);
    const half* kc = reinterpret_cast<const half*>(cache);
    static const int rows = [] { const char* v = std::getenv("NS_DS4_IDX_ROWS"); return v ? std::atoi(v) : 1; }();
    static const int kt = [] { const char* v = std::getenv("NS_DS4_IDX_KT"); return v ? std::atoi(v) : 2; }();
    if (packed) {   // the queries packed in global memory: no local memory, so rows cost no occupancy
        if (rows >= 4) idx_score_pre<4, 4, 2, true>(g, qh, w, kc, score, T, n, ld, pos0);
        else if (rows == 2) idx_score_pre<2, 4, 2, true>(g, qh, w, kc, score, T, n, ld, pos0);
        else idx_score_pre<1, 4, 2, true>(g, qh, w, kc, score, T, n, ld, pos0);
    } else if (kt >= 2) {
        if (rows >= 2) idx_score_pre<2, 4, 2, false>(g, qh, w, kc, score, T, n, ld, pos0);
        else idx_score_pre<1, 4, 2, false>(g, qh, w, kc, score, T, n, ld, pos0);
    } else {
        idx_score_pre<1, 4, 1, false>(g, qh, w, kc, score, T, n, ld, pos0);
    }
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
