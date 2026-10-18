// q4k.cpp: q4_K decode products against float32 activations (ds41.h: ns_ds41_q4k_mv) - one launch a product, no
// quantization of x first (the shared path quantizes to Q8_1 and launches again), and o_a's 8 groups in one launch: a row
// block's input chosen by its group.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <cstdint>

namespace {
struct BlockQ4K {
    sycl::half d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(BlockQ4K) == 144, "block_q4_K is 144 bytes");

inline void scale_min(int j, const uint8_t* q, float& sc, float& m) {
    if (j < 4) {
        sc = (float) (q[j] & 63);
        m = (float) (q[j + 4] & 63);
    } else {
        sc = (float) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (float) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}
// R rows a sub-group: a lane's 64 values of x read once for all R (the activations' reads, 8 bytes for each weight byte,
// were what bound the one-row form on the B65)
template <int R>
void q4k_rows(sycl::queue& q, const BlockQ4K* wb, int64_t rows, int64_t nb, const float* x, int64_t xcs, int64_t rpg, int64_t xgs, float* y, int64_t ycs,
              int64_t t) {
    constexpr int SG = 16, SPG = 8, MC = 8;
    const int64_t rb = (rows + R - 1) / R;
    q.parallel_for(sycl::nd_range<1>((rb + SPG - 1) / SPG * SPG * SG, SPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t r0 = (it.get_group(0) * SPG + sg.get_group_id()[0]) * R;
        const int lane = sg.get_local_id()[0];
        if (r0 >= rows) return;
        const int nr = (int) sycl::min<int64_t>(R, rows - r0);
        const float* xr = x + (rpg > 0 ? (r0 / rpg) * xgs : 0);
        float acc[R][MC] = {};
        for (int64_t i = lane; i < nb * 4; i += SG) {
            const int64_t b = i >> 2;
            const int k = (int) (i & 3);
            const int64_t x0 = b * 256 + k * 64;
            for (int c = 0; c < MC; ++c) {
                if (c >= t) break;
                const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(xr + c * xcs + x0);
                sycl::float4 xa[8], xb[8];
                float slo = 0.f, shi = 0.f;
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    xa[u] = xv[u];
                    xb[u] = xv[8 + u];
                    slo += xa[u].x() + xa[u].y() + xa[u].z() + xa[u].w();
                    shi += xb[u].x() + xb[u].y() + xb[u].z() + xb[u].w();
                }
#pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    if (rr >= nr) break;
                    const BlockQ4K& blk = wb[(r0 + rr) * nb + b];
                    const float d = (float) blk.d, dm = (float) blk.dmin;
                    float s0, m0, s1, m1;
                    scale_min(2 * k, blk.scales, s0, m0);
                    scale_min(2 * k + 1, blk.scales, s1, m1);
                    const uint32_t* qp = reinterpret_cast<const uint32_t*>(blk.qs + 32 * k);
                    float lo = 0.f, hi = 0.f;
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const uint32_t qq = qp[u];
                        lo += (float) (qq & 0xF) * xa[u].x() + (float) ((qq >> 8) & 0xF) * xa[u].y() + (float) ((qq >> 16) & 0xF) * xa[u].z() +
                              (float) ((qq >> 24) & 0xF) * xa[u].w();
                        hi += (float) ((qq >> 4) & 0xF) * xb[u].x() + (float) ((qq >> 12) & 0xF) * xb[u].y() + (float) ((qq >> 20) & 0xF) * xb[u].z() +
                              (float) ((qq >> 28) & 0xF) * xb[u].w();
                    }
                    acc[rr][c] += d * s0 * lo - dm * m0 * slo + d * s1 * hi - dm * m1 * shi;
                }
            }
        }
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            if (rr >= nr) break;
            for (int c = 0; c < t; ++c) {
                const float s = sycl::reduce_over_group(sg, acc[rr][c], sycl::plus<float>());
                if (lane == 0) y[c * ycs + r0 + rr] = s;
            }
        }
    });
}
}  // namespace

extern "C" {

int ns_ds41_q4k_mv(ns_gpu* g, const void* w, int64_t rows, int64_t cols, const float* x, int64_t xcs, int64_t rpg, int64_t xgs, float* y, int64_t ycs,
                   int64_t t) {
    NS_TRY
    if (cols % 256 != 0 || t < 1 || t > 8) return ns_fail("ns_ds41_q4k_mv: cols a multiple of 256, 1..8 columns");
    static const int rows_sg = [] { const char* v = std::getenv("NS_DS41_Q4K_R"); return v ? std::atoi(v) : 4; }();
    if (rows_sg == 2 || rows_sg == 4) {
        if (rpg > 0 && rpg % rows_sg != 0) return ns_fail("ns_ds41_q4k_mv: a group's rows a multiple of the rows a sub-group");
        const BlockQ4K* wb = static_cast<const BlockQ4K*>(w);
        if (rows_sg == 2) q4k_rows<2>(g->q, wb, rows, cols / 256, x, xcs, rpg, xgs, y, ycs, t);
        else q4k_rows<4>(g->q, wb, rows, cols / 256, x, xcs, rpg, xgs, y, ycs, t);
        return 0;
    }
    constexpr int SG = 16, RPG = 8, MC = 8;
    const int64_t nb = cols / 256;
    const BlockQ4K* wb = static_cast<const BlockQ4K*>(w);
    g->q.parallel_for(sycl::nd_range<1>((rows + RPG - 1) / RPG * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0) * RPG + sg.get_group_id()[0];
        const int lane = sg.get_local_id()[0];
        if (r >= rows) return;
        const float* xr = x + (rpg > 0 ? (r / rpg) * xgs : 0);
        float acc[MC] = {};
        // a block a lane-group of 4: each lane 64 of its 256 values (one sub-block pair)
        for (int64_t i = lane; i < nb * 4; i += SG) {
            const int64_t b = i >> 2;
            const int k = (int) (i & 3);
            const BlockQ4K& blk = wb[r * nb + b];
            const float d = (float) blk.d, dm = (float) blk.dmin;
            float s0, m0, s1, m1;
            scale_min(2 * k, blk.scales, s0, m0);
            scale_min(2 * k + 1, blk.scales, s1, m1);
            const float a0 = d * s0, c0 = dm * m0, a1 = d * s1, c1 = dm * m1;
            const uint32_t* q = reinterpret_cast<const uint32_t*>(blk.qs + 32 * k);
            const int64_t x0 = b * 256 + k * 64;
#pragma unroll
            for (int c = 0; c < MC; ++c) {
                if (c >= t) break;
                const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(xr + c * xcs + x0);
                float lo = 0.f, hi = 0.f, slo = 0.f, shi = 0.f;
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const uint32_t qq = q[u];
                    const sycl::float4 xa = xv[u], xb = xv[8 + u];
                    lo += (float) (qq & 0xF) * xa.x() + (float) ((qq >> 8) & 0xF) * xa.y() + (float) ((qq >> 16) & 0xF) * xa.z() + (float) ((qq >> 24) & 0xF) * xa.w();
                    hi += (float) ((qq >> 4) & 0xF) * xb.x() + (float) ((qq >> 12) & 0xF) * xb.y() + (float) ((qq >> 20) & 0xF) * xb.z() + (float) ((qq >> 28) & 0xF) * xb.w();
                    slo += xa.x() + xa.y() + xa.z() + xa.w();
                    shi += xb.x() + xb.y() + xb.z() + xb.w();
                }
                acc[c] += a0 * lo - c0 * slo + a1 * hi - c1 * shi;
            }
        }
        for (int c = 0; c < t; ++c) {
            const float s = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
            if (lane == 0) y[c * ycs + r] = s;
        }
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
