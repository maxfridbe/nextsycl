// fmv.cpp: the inkling2 silo's float products at decode width (silo.h: ns_ink2s_fmv) - the float32 router and loop
// controllers, the BF16 sink and convolution-tap projections - against float32 rows. A sub-group takes R rows; its
// lanes take 8 values of a row at a time (one or two 16-byte loads), the row's 8 inputs loaded once for the R rows; the
// lanes' sums added in a fixed order (a row's value the same whatever rows share a pass).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

namespace {
constexpr int SG = 16, R = 4, NSG = 4;

template <int TY>
inline void wload(const uint8_t* row, int64_t c, float* v) {
    if constexpr (TY == 0) {
        const sycl::float4* p = (const sycl::float4*) (row + c * 32);
        const sycl::float4 a = p[0], b = p[1];
        v[0] = a.x(); v[1] = a.y(); v[2] = a.z(); v[3] = a.w();
        v[4] = b.x(); v[5] = b.y(); v[6] = b.z(); v[7] = b.w();
    } else {
        const sycl::uint4 a = *(const sycl::uint4*) (row + c * 16);
        const uint32_t u[4] = {a.x(), a.y(), a.z(), a.w()};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i] = sycl::bit_cast<float>(u[i] << 16);
            v[2 * i + 1] = sycl::bit_cast<float>(u[i] & 0xFFFF0000u);
        }
    }
}

template <int TY, int TT>
void fmv_launch(sycl::queue& q, const uint8_t* w, const float* x, float* y, int64_t n_in, int64_t rows) {
    const int64_t nc = n_in / 8, rb = TY == 0 ? n_in * 4 : n_in * 2;
    const int64_t blocks = (rows + R - 1) / R, wgs = (blocks + NSG - 1) / NSG;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) wgs * NSG * SG), sycl::range<1>(NSG * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t r0 = (it.get_group(0) * NSG + sg.get_group_linear_id()) * R;
        if (r0 >= rows) return;
        const int nr = (int) sycl::min<int64_t>(R, rows - r0);
        float acc[R][TT];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < TT; ++t) acc[r][t] = 0.0f;
        for (int64_t c = lane; c < nc; c += SG) {
            float wv[R][8];
#pragma unroll
            for (int r = 0; r < R; ++r) wload<TY>(w + (r0 + sycl::min(r, nr - 1)) * rb, c, wv[r]);
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const sycl::float4* xp = (const sycl::float4*) (x + t * n_in + c * 8);
                const sycl::float4 a = xp[0], b = xp[1];
                const float xv[8] = {a.x(), a.y(), a.z(), a.w(), b.x(), b.y(), b.z(), b.w()};
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float s = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) s = sycl::fma(wv[r][i], xv[i], s);
                    acc[r][t] += s;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const float v = sycl::reduce_over_group(sg, acc[r][t], sycl::plus<float>());
                if (lane == 0 && r < nr) y[t * rows + r0 + r] = v;
            }
    });
}

template <int TY>
void fmv_t(sycl::queue& q, const uint8_t* w, const float* x, float* y, int64_t n_in, int64_t rows, int64_t T) {
    switch (T) {
        case 1: fmv_launch<TY, 1>(q, w, x, y, n_in, rows); break;
        case 2: fmv_launch<TY, 2>(q, w, x, y, n_in, rows); break;
        case 3: fmv_launch<TY, 3>(q, w, x, y, n_in, rows); break;
        case 4: fmv_launch<TY, 4>(q, w, x, y, n_in, rows); break;
        case 5: fmv_launch<TY, 5>(q, w, x, y, n_in, rows); break;
        case 6: fmv_launch<TY, 6>(q, w, x, y, n_in, rows); break;
        case 7: fmv_launch<TY, 7>(q, w, x, y, n_in, rows); break;
        case 8: fmv_launch<TY, 8>(q, w, x, y, n_in, rows); break;
        case 9: fmv_launch<TY, 9>(q, w, x, y, n_in, rows); break;
        case 10: fmv_launch<TY, 10>(q, w, x, y, n_in, rows); break;
        case 11: fmv_launch<TY, 11>(q, w, x, y, n_in, rows); break;
        case 12: fmv_launch<TY, 12>(q, w, x, y, n_in, rows); break;
        case 13: fmv_launch<TY, 13>(q, w, x, y, n_in, rows); break;
        case 14: fmv_launch<TY, 14>(q, w, x, y, n_in, rows); break;
        case 15: fmv_launch<TY, 15>(q, w, x, y, n_in, rows); break;
        default: fmv_launch<TY, 16>(q, w, x, y, n_in, rows); break;
    }
}
}  // namespace

extern "C" {

int ns_ink2s_fmv_supported(int type, int64_t n_in) {
    return (type == 0 || type == 30) && n_in % 8 == 0;
}

int ns_ink2s_fmv(ns_gpu* g, int type, const void* w, const float* x, float* y, int64_t n_in, int64_t rows, int64_t T) {
    NS_TRY
    if (!ns_ink2s_fmv_supported(type, n_in) || T < 1 || T > 16) return ns_fail("ns_ink2s_fmv: a type, width or row count it does not take");
    if (rows <= 0) return 0;
    if (type == 0) fmv_t<0>(g->q, (const uint8_t*) w, x, y, n_in, rows, T);
    else fmv_t<30>(g->q, (const uint8_t*) w, x, y, n_in, rows, T);
    return 0;
    NS_CATCH
}

}  // extern "C"
