// mmvq.cpp: the museglimmer silo (silo.h) - Muse Glimmer 30B's Q8_0 matrices repacked at load (quants apart from
// their scales: 16-byte aligned rows), the decode products against them (a sub-group a row, each lane whole blocks:
// two 16-byte loads of weights, dp4a against the activations in the same split layout), and their expansion to half
// for a prompt pass. The arithmetic is ggml's Q8_0 x Q8_1: a block's integer dot x the weight's and the activation's
// half scales.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdlib>

namespace {
using half = sycl::half;
constexpr int SG = 16, NSG = 4, WG = SG * NSG;

inline int dp4a(int a, int b, int c) { return dpct::dp4a(a, b, c); }
inline int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = dp4a(a.x(), b.x(), acc); acc = dp4a(a.y(), b.y(), acc); acc = dp4a(a.z(), b.z(), acc); return dp4a(a.w(), b.w(), acc);
}

size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// R rows a sub-group (the activations' loads shared), C columns
template <int R, int C>
void mmvq(sycl::queue& q, const int8_t* wq, const half* ws, const int8_t* xq, const float* xs, float* y, int64_t n_in, int64_t n_out) {
    const int64_t nb = n_in / 32;
    const int64_t rows_wg = (int64_t) NSG * R;
    const int64_t groups = (n_out + rows_wg - 1) / rows_wg;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
        const int64_t r0 = (it.get_group(0) * NSG + sgi) * R;
        if (r0 >= n_out) return;
        float acc[R][C];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < C; ++c) acc[r][c] = 0.0f;
        for (int64_t b = lane; b < nb; b += SG) {
            sycl::int4 xv[C][2];
            float xd[C];
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const sycl::int4* xp = reinterpret_cast<const sycl::int4*>(xq + c * n_in + 32 * b);
                xv[c][0] = xp[0];
                xv[c][1] = xp[1];
                xd[c] = xs[c * nb + b];
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int64_t row = sycl::min<int64_t>(r0 + r, n_out - 1);
                const sycl::int4* wp = reinterpret_cast<const sycl::int4*>(wq + row * n_in + 32 * b);
                const sycl::int4 w0 = wp[0], w1 = wp[1];
                const float wd = (float) ws[row * nb + b];
#pragma unroll
                for (int c = 0; c < C; ++c) acc[r][c] += (float) dp4a4(w1, xv[c][1], dp4a4(w0, xv[c][0], 0)) * (wd * xd[c]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const float v = sycl::reduce_over_group(sg, acc[r][c], sycl::plus<float>());
                if (lane == 0 && r0 + r < n_out) y[c * n_out + r0 + r] = v;
            }
        }
    });
}

template <int R>
void mmvq_cols(sycl::queue& q, const int8_t* wq, const half* ws, const int8_t* xq, const float* xs, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    switch (ncols) {
        case 1: mmvq<R, 1>(q, wq, ws, xq, xs, y, n_in, n_out); break;
        case 2: mmvq<R, 2>(q, wq, ws, xq, xs, y, n_in, n_out); break;
        case 3: mmvq<R, 3>(q, wq, ws, xq, xs, y, n_in, n_out); break;
        case 4: mmvq<R, 4>(q, wq, ws, xq, xs, y, n_in, n_out); break;
        case 5: mmvq<R, 5>(q, wq, ws, xq, xs, y, n_in, n_out); break;
        case 6: mmvq<R, 6>(q, wq, ws, xq, xs, y, n_in, n_out); break;
        case 7: mmvq<R, 7>(q, wq, ws, xq, xs, y, n_in, n_out); break;
        default: mmvq<R, 8>(q, wq, ws, xq, xs, y, n_in, n_out); break;
    }
}
}  // namespace

extern "C" {

size_t ns_mg_q8r_scales_at(int64_t rows, int64_t n_in) { return align256((size_t) (rows * n_in)); }
size_t ns_mg_q8r_bytes(int64_t rows, int64_t n_in) { return ns_mg_q8r_scales_at(rows, n_in) + (size_t) (rows * (n_in / 32)) * 2; }

int ns_mg_q8r_pack(ns_gpu* g, const void* q8_0, int64_t rows, int64_t n_in, void* dst) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_mg_q8r_pack: n_in a multiple of 32");
    const uint8_t* src = (const uint8_t*) q8_0;
    int8_t* oq = (int8_t*) dst;
    half* os = (half*) ((uint8_t*) dst + ns_mg_q8r_scales_at(rows, n_in));
    const int64_t nb = rows * (n_in / 32);
    // a work-item a block: its 2-byte scale, then its 32 quants
    g->q.parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> id) {
        const int64_t b = id[0];
        const uint8_t* blk = src + b * 34;
        os[b] = *reinterpret_cast<const half*>(blk);
        for (int i = 0; i < 32; ++i) oq[b * 32 + i] = (int8_t) blk[2 + i];
    });
    return 0;
    NS_CATCH
}

size_t ns_mg_xq_bytes(int64_t n_in, int64_t ncols) { return align256((size_t) (ncols * n_in)) + (size_t) (ncols * (n_in / 32)) * 4; }

int ns_mg_xq(ns_gpu* g, const float* a, const float* b, int op, void* xq, int64_t n_in, int64_t ncols) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_mg_xq: n_in a multiple of 32");
    if (op != 0 && !b) return ns_fail("ns_mg_xq: op needs b");
    int8_t* oq = (int8_t*) xq;
    float* os = (float*) ((uint8_t*) xq + align256((size_t) (ncols * n_in)));
    const int64_t nb = ncols * (n_in / 32);
    // a sub-group of 32 a block (ggml's quantize_q8_1: d = amax / 127, q = round(x / d), d kept as half)
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * 32), sycl::range<1>(32)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const int64_t blk = it.get_group(0);
        const int i = (int) sg.get_local_linear_id();
        const int64_t at = blk * 32 + i;
        float v = a[at];
        if (op == 1) v *= 1.0f / (1.0f + sycl::exp(-b[at]));
        else if (op == 2) v = v / (1.0f + sycl::exp(-v)) * b[at];
        const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        oq[at] = (int8_t) sycl::round(v * id);
        if (i == 0) os[blk] = (float) (half) d;
    });
    return 0;
    NS_CATCH
}

int ns_mg_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t rows, int64_t C, float eps) {
    NS_TRY
    if (C % 32 != 0) return ns_fail("ns_mg_rms_xq: C a multiple of 32");
    int8_t* oq = (int8_t*) xq;
    float* os = (float*) ((uint8_t*) xq + align256((size_t) (rows * C)));
    const int64_t nb = C / 32;
    constexpr int W = 256, S = 32;
    // a work-group a row: the sum of squares, then a sub-group a block at a time
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(S)]] {
        auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
        const float* xr = x + r * C;
        float ss = 0.0f;
        for (int64_t i = tid; i < C; i += W) ss += xr[i] * xr[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) C + eps);
        for (int64_t b = sgi; b < nb; b += W / S) {
            const int64_t i = b * 32 + lane;
            const float v = xr[i] * inv * w[i];
            if (y) y[r * C + i] = v;
            const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            oq[r * C + i] = (int8_t) sycl::round(v * id);
            if (lane == 0) os[r * nb + b] = (float) (half) d;
        }
    });
    return 0;
    NS_CATCH
}

int ns_mg_post_rms_xq(ns_gpu* g, const float* out, const float* wp, float ep, float* x, const float* wn, float en, void* xq, int64_t rows, int64_t C) {
    NS_TRY
    if (C % 32 != 0) return ns_fail("ns_mg_post_rms_xq: C a multiple of 32");
    int8_t* oq = (int8_t*) xq;
    float* os = (float*) ((uint8_t*) xq + align256((size_t) (rows * C)));
    const int64_t nb = C / 32;
    constexpr int W = 256, S = 32;
    // a work-group a row: out's sum of squares, x updated (each work-item its own elements), x's sum of squares, then
    // a sub-group a block at a time (x read back after the barrier: its elements were other work-items')
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(S)]] {
        auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0), lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
        const float* orow = out + r * C;
        float* xr = x + r * C;
        float so = 0.0f;
        for (int64_t i = tid; i < C; i += W) so += orow[i] * orow[i];
        const float inv_o = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), so, sycl::plus<float>()) / (float) C + ep);
        float sx = 0.0f;
        for (int64_t i = tid; i < C; i += W) {
            const float v = xr[i] + orow[i] * inv_o * wp[i];
            xr[i] = v;
            sx += v * v;
        }
        const float inv_x = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), sx, sycl::plus<float>()) / (float) C + en);
        sycl::group_barrier(it.get_group());
        for (int64_t b = sgi; b < nb; b += W / S) {
            const int64_t i = b * 32 + lane;
            const float v = xr[i] * inv_x * wn[i];
            const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            oq[r * C + i] = (int8_t) sycl::round(v * id);
            if (lane == 0) os[r * nb + b] = (float) (half) d;
        }
    });
    return 0;
    NS_CATCH
}

int ns_mg_mmvq(ns_gpu* g, const void* w, const void* wsc, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (ncols < 1 || ncols > 8 || n_in % 32 != 0) return ns_fail("ns_mg_mmvq: 1..8 columns, n_in a multiple of 32");
    const int8_t* wq = (const int8_t*) w;
    const half* ws = (const half*) wsc;
    const int8_t* xqq = (const int8_t*) xq;
    const float* xs = (const float*) ((const uint8_t*) xq + align256((size_t) (ncols * n_in)));
    static const int rows = [] { const char* v = std::getenv("NS_MG_MMVQ_ROWS"); return v ? std::atoi(v) : 1; }();
    if (rows >= 2) mmvq_cols<2>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols);
    else mmvq_cols<1>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols);
    return 0;
    NS_CATCH
}

int ns_mg_q8r_f16(ns_gpu* g, const void* w, const void* wsc, int64_t rows, int64_t n_in, uint16_t* dst) {
    NS_TRY
    const int8_t* wq = (const int8_t*) w;
    const half* ws = (const half*) wsc;
    half* o = (half*) dst;
    const int64_t n = rows * n_in;
    // a work-item 16 values (one 16-byte load)
    g->q.parallel_for(sycl::range<1>((size_t) (n / 16)), [=](sycl::id<1> id) {
        const int64_t i = id[0] * 16;
        const float d = (float) ws[i / 32];
        const sycl::vec<int8_t, 16> v = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(wq + i);
        sycl::vec<half, 16> h;
#pragma unroll
        for (int k = 0; k < 16; ++k) h[k] = (half) ((float) v[k] * d);
        *reinterpret_cast<sycl::vec<half, 16>*>(o + i) = h;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
