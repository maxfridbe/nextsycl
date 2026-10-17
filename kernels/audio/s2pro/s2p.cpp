// s2p.cpp: the Fish Audio S2 Pro engine's own kernels (s2p.h). The decode reads every matrix once a frame (the slow
// model's 36 layers, the fast model's 4 layers ten times), so its product keeps the file's Q8_0 blocks - 1.06 bytes a
// weight, not half's 2 - with the int8 codes and the block scales in two arrays so a lane reads 8 aligned codes; the
// rest are plain item-per-row kernels. Sub-groups of 16 (Xe2's native width).
#include "ns.h"
#include "ns_internal.hpp"
#include "s2p.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;
}  // namespace

extern "C" {

int ns_audio_s2p_gemv_q8(ns_gpu* g, const float* x, int64_t M, int64_t K, const int8_t* q, const void* s, int64_t N, float* out, int64_t ldo,
                         int acc) {
    NS_TRY
    if (M <= 0 || N <= 0) return 0;
    if (M > 8 || K % 256 != 0) return ns_fail("ns_audio_s2p_gemv_q8: at most 8 rows, K a multiple of 256");
    // each sub-group R output rows, its lanes 8 codes at a time across K (8 codes sit in one block of 32)
    constexpr int NSG = 8, R = 2, MM = 8, V = 8;
    const half* sc = (const half*) s;
    const int64_t nb = K / 32;
    const bool add = acc != 0;
    const int m_rows = (int) M;
    const int64_t groups = (N + NSG * R - 1) / (NSG * R);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SG * NSG), sycl::range<1>(SG * NSG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t n0 = ((int64_t) it.get_group(0) * NSG + sg.get_group_linear_id()) * R;
        float a[MM][R];
#pragma unroll
        for (int m = 0; m < MM; ++m)
#pragma unroll
            for (int r = 0; r < R; ++r) a[m][r] = 0.0f;
        for (int64_t k = (int64_t) lane * V; k < K; k += SG * V) {
            float wf[R][V];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int64_t n = n0 + r < N ? n0 + r : N - 1;
                const sycl::vec<int8_t, V> wv = *reinterpret_cast<const sycl::vec<int8_t, V>*>(q + n * K + k);
                const float d = (float) sc[n * nb + k / 32];
#pragma unroll
                for (int i = 0; i < V; ++i) wf[r][i] = d * (float) wv[i];
            }
#pragma unroll
            for (int m = 0; m < MM; ++m) {
                if (m < m_rows) {
                    const sycl::vec<float, 4> x0 = *reinterpret_cast<const sycl::vec<float, 4>*>(x + m * K + k);
                    const sycl::vec<float, 4> x1 = *reinterpret_cast<const sycl::vec<float, 4>*>(x + m * K + k + 4);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        a[m][r] += wf[r][0] * x0[0] + wf[r][1] * x0[1] + wf[r][2] * x0[2] + wf[r][3] * x0[3] +
                                   wf[r][4] * x1[0] + wf[r][5] * x1[1] + wf[r][6] * x1[2] + wf[r][7] * x1[3];
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            if (m < m_rows) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float v = sycl::reduce_over_group(sg, a[m][r], sycl::plus<float>());
                    const int64_t n = n0 + r;
                    if (lane == 0 && n < N) {
                        float* o = out + m * ldo + n;
                        *o = add ? *o + v : v;
                    }
                }
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_audio_s2p_dequant_q8(ns_gpu* g, const int8_t* q, const void* s, int64_t N, int64_t K, float* out) {
    NS_TRY
    if (K % 32 != 0) return ns_fail("ns_audio_s2p_dequant_q8: K a multiple of 32");
    if (N <= 0 || K <= 0) return 0;
    const half* sc = (const half*) s;
    g->q.parallel_for(sycl::range<1>((size_t) (N * K)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        out[i] = (float) sc[i / 32] * (float) q[i];
    });
    return 0;
    NS_CATCH
}

int ns_audio_s2p_half_float(ns_gpu* g, const void* x, int64_t n, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    const half* h = (const half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = (float) h[i]; });
    return 0;
    NS_CATCH
}

int ns_audio_s2p_norm_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* weight, float eps,
                           const float* cs, int64_t p0) {
    NS_TRY
    if (D > 256 || D % 2 != 0 || S <= 0) return ns_fail("ns_audio_s2p_norm_rope: D even, at most 256");
    if (rows <= 0 || H <= 0) return 0;
    const int64_t pairs = D / 2;
    // one work-item a (row, head): the norm over its D values, then each interleaved pair rotated (fish-speech's
    // apply_rotary_emb: (x0, x1) -> (x0 c - x1 s, x1 c + x0 s))
    g->q.parallel_for(sycl::range<1>((size_t) (rows * H)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / H, h = id[0] % H;
        float* v = x + r * xs + h * D;
        if (weight) {
            float ss = 0.0f;
            for (int64_t d = 0; d < D; ++d) ss += v[d] * v[d];
            const float inv = sycl::rsqrt(ss / (float) D + eps);
            for (int64_t d = 0; d < D; ++d) v[d] = v[d] * inv * weight[d];
        }
        const float* t = cs + (p0 + r % S) * pairs * 2;
        for (int64_t j = 0; j < pairs; ++j) {
            const float c = t[2 * j], s = t[2 * j + 1], x0 = v[2 * j], x1 = v[2 * j + 1];
            v[2 * j] = x0 * c - x1 * s;
            v[2 * j + 1] = x1 * c + x0 * s;
        }
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
