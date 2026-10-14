// mmvq.cpp: the museglimmer silo (silo.h) - Muse Glimmer 30B's Q8_0 matrices repacked at load (quants apart from
// their scales: 16-byte aligned rows), the decode products against them on XMX, and their expansion to half for a
// prompt pass. The activations of a decode pass (1..16 rows) are quantized as ggml's Q8_1 (amax / 127 a block of 32, the
// scale kept as half) straight into the XMX operand layout: block b's quants at b x 512 bytes, word w (quants 4w .. 4w
// + 3) of column c at w x 64 + c x 4 - every column's tile loads as one packed int8 B operand, columns past a pass's
// rows unused (their scale 0). The product: a weight tile of 8 rows x one block (int8 DPAS 8 x 32 x 16) against it, the
// int32 result x the rows' and columns' scales into float sums - a column's arithmetic the same however many columns a
// pass has, so a verify row's logits are its one-row pass's bit for bit (on dp4a, one row a lane, a 16-row pass was
// compute-bound: a 15-draft step cost 1.7 one-row passes).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"
#include "xq.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <dpct/dpct.hpp>
#include <cstdlib>

namespace {
using half = sycl::half;
constexpr int SG = 16, NSG = 4, WG = SG * NSG;


using mgxq::align256;
using mgxq::XCOLS;
// the activations' layout (xq.hpp)
inline int64_t xq_at(int64_t c, int64_t i) { return mgxq::at(c, i); }
inline size_t xq_scales(int64_t n_in) { return mgxq::scales(n_in); }

// ---- the products on XMX (int8 DPAS, 8 x 32 x 16): a weight tile of 8 rows x 32 (one Q8 block of each) against the
// 16 columns' quants of that block (packed by each sub-group into local memory, columns past ncols zero), the int32
// result scaled by the rows' and the columns' block scales into float sums. A column's arithmetic is the same however
// many columns a pass has (a verify row's sums are its one-row pass's, bit for bit). A sub-group takes RT tiles (8 RT
// rows), the work-group's SK sub-groups split the blocks between them, their sums added in a fixed order.
namespace jm = sycl::ext::oneapi::experimental::matrix;
template <typename T>
inline auto gpm(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

template <int RT, int SK, int KB = 1>
void mmvq_xmx(sycl::queue& q, const int8_t* wq, const half* ws, const int8_t* xq, const float* xs, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    const int64_t nb = n_in / 32;
    constexpr int ROWS = RT * 8;
    const int64_t groups = (n_out + ROWS - 1) / ROWS;
    q.submit([&](sycl::handler& h) {
        // the sums to add [SK][RT][8][16]
        sycl::local_accessor<float, 1> red(sycl::range<1>(SK * RT * 8 * 16), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SK * SG), sycl::range<1>(SK * SG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int64_t row0 = it.get_group(0) * ROWS;
            float acc[RT][8];
#pragma unroll
            for (int t = 0; t < RT; ++t)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[t][e] = 0.0f;
            const bool col = lane < ncols;
            // KB consecutive blocks an iteration (KB = 2: a weight row's whole 64-byte line)
            for (int64_t k0 = sgi * KB; k0 < nb; k0 += SK * KB)
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) {
                const int64_t k = k0 + kk;
                if (k >= nb) break;
                jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, 16, jm::layout::ext_intel_packed> bm;
                jm::joint_matrix_load(sg, bm, gpm(xq + k * 512), (size_t) 64);
                const float xsc = col ? xs[k * 16 + lane] : 0.0f;
#pragma unroll
                for (int t = 0; t < RT; ++t) {
                    const int64_t r = row0 + t * 8;
                    if (r >= n_out) break;
                    jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> am;
                    jm::joint_matrix_load(sg, am, gpm(wq + r * n_in + 32 * k), (size_t) n_in);
                    jm::joint_matrix<sycl::sub_group, int32_t, jm::use::accumulator, 8, 16> cm;
                    jm::joint_matrix_fill(sg, cm, 0);
                    jm::joint_matrix_mad(sg, cm, am, bm, cm);
                    // the 8 rows' scales of this block, each lane one, then to every lane
                    const float wl = lane < 8 ? (float) ws[(r + lane) * nb + k] : 0.0f;
                    float wr[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) wr[e] = sycl::group_broadcast(sg, wl, e) * xsc;
                    int e = 0;
                    jm::joint_matrix_apply(sg, cm, [&](int32_t& v) { acc[t][e] += (float) v * wr[e]; ++e; });
                }
            }
            // the sub-groups' sums added in order: element (row e, column lane) of each tile
            float* rd = red.get_multi_ptr<sycl::access::decorated::no>().get();
#pragma unroll
            for (int t = 0; t < RT; ++t)
#pragma unroll
                for (int e = 0; e < 8; ++e) rd[((sgi * RT + t) * 8 + e) * 16 + lane] = acc[t][e];
            sycl::group_barrier(it.get_group());
            for (int i = (int) it.get_local_id(0); i < RT * 8 * 16; i += SK * SG) {
                const int c = i % 16, re = i / 16;
                const int64_t row = row0 + re;
                if (c >= ncols || row >= n_out) continue;
                float v = 0.0f;
                for (int s2 = 0; s2 < SK; ++s2) v += rd[(s2 * RT * 8 + re) * 16 + c];
                y[c * n_out + row] = v;
            }
        });
    });
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

int ns_mg_q8r_from_f32(ns_gpu* g, const float* x, int64_t rows, int64_t n_in, void* dst) {
    NS_TRY
    if (n_in % 32 != 0) return ns_fail("ns_mg_q8r_from_f32: n_in a multiple of 32");
    int8_t* oq = (int8_t*) dst;
    half* os = (half*) ((uint8_t*) dst + ns_mg_q8r_scales_at(rows, n_in));
    const int64_t nb = rows * (n_in / 32);
    // a sub-group of 32 a block: Q8_0's d = amax / 127 (as half), q = round(x / d)
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nb * 32), sycl::range<1>(32)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const int64_t b = it.get_group(0);
        const int i = (int) sg.get_local_linear_id();
        const float v = x[b * 32 + i];
        const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
        const half d = (half) (amax / 127.0f);
        const float df = (float) d;
        oq[b * 32 + i] = (int8_t) (df != 0.0f ? sycl::clamp(sycl::round(v / df), -127.0f, 127.0f) : 0.0f);
        if (i == 0) os[b] = d;
    });
    return 0;
    NS_CATCH
}

size_t ns_mg_xq_bytes(int64_t n_in, int64_t ncols) { return mgxq::bytes(n_in); }

int ns_mg_xq(ns_gpu* g, const float* a, const float* b, int op, void* xq, int64_t n_in, int64_t ncols) {
    NS_TRY
    if (n_in % 32 != 0 || ncols > XCOLS) return ns_fail("ns_mg_xq: n_in a multiple of 32, at most 16 columns");
    if (op != 0 && !b) return ns_fail("ns_mg_xq: op needs b");
    int8_t* oq = (int8_t*) xq;
    float* os = (float*) ((uint8_t*) xq + xq_scales(n_in));
    float* osum = (float*) ((uint8_t*) xq + mgxq::sums(n_in));
    int8_t* oa = (int8_t*) xq + mgxq::rowq(n_in);
    const int64_t nbc = n_in / 32;
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
        const int qi = (int) sycl::round(v * id);
        oq[xq_at(blk / nbc, (blk % nbc) * 32 + i)] = (int8_t) qi;
        oa[mgxq::at_a(blk / nbc, (blk % nbc) * 32 + i)] = (int8_t) qi;
        const int qs = sycl::reduce_over_group(sg, qi, sycl::plus<int>());
        if (i == 0) {
            os[(blk % nbc) * 16 + blk / nbc] = (float) (half) d;
            osum[(blk % nbc) * 16 + blk / nbc] = (float) (half) d * (float) qs;
        }
    });
    return 0;
    NS_CATCH
}

int ns_mg_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t rows, int64_t C, float eps) {
    NS_TRY
    if (C % 32 != 0 || rows > XCOLS) return ns_fail("ns_mg_rms_xq: C a multiple of 32, at most 16 rows");
    int8_t* oq = (int8_t*) xq;
    float* os = (float*) ((uint8_t*) xq + xq_scales(C));
    float* osum = (float*) ((uint8_t*) xq + mgxq::sums(C));
    int8_t* oa = (int8_t*) xq + mgxq::rowq(C);
    const int64_t nb = C / 32;
    constexpr int W = 1024, S = 32;
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
            const int qi = (int) sycl::round(v * id);
            oq[xq_at(r, i)] = (int8_t) qi;
            oa[mgxq::at_a(r, i)] = (int8_t) qi;
            const int qs = sycl::reduce_over_group(sg, qi, sycl::plus<int>());
            if (lane == 0) {
                os[b * 16 + r] = (float) (half) d;
                osum[b * 16 + r] = (float) (half) d * (float) qs;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_mg_post_rms_xq(ns_gpu* g, const float* out, const float* wp, float ep, float* x, const float* wn, float en, void* xq, int64_t rows, int64_t C) {
    NS_TRY
    if (C % 32 != 0 || rows > XCOLS) return ns_fail("ns_mg_post_rms_xq: C a multiple of 32, at most 16 rows");
    int8_t* oq = (int8_t*) xq;
    float* os = (float*) ((uint8_t*) xq + xq_scales(C));
    float* osum = (float*) ((uint8_t*) xq + mgxq::sums(C));
    int8_t* oa = (int8_t*) xq + mgxq::rowq(C);
    const int64_t nb = C / 32;
    constexpr int W = 1024, S = 32;
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
            const int qi = (int) sycl::round(v * id);
            oq[xq_at(r, i)] = (int8_t) qi;
            oa[mgxq::at_a(r, i)] = (int8_t) qi;
            const int qs = sycl::reduce_over_group(sg, qi, sycl::plus<int>());
            if (lane == 0) {
                os[b * 16 + r] = (float) (half) d;
                osum[b * 16 + r] = (float) (half) d * (float) qs;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_mg_mmvq(ns_gpu* g, const void* w, const void* wsc, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (ncols < 1 || ncols > XCOLS || n_in % 32 != 0 || n_out % 8 != 0) return ns_fail("ns_mg_mmvq: 1..16 columns, n_in a multiple of 32, n_out of 8");
    const int8_t* wq = (const int8_t*) w;
    const half* ws = (const half*) wsc;
    const int8_t* xqq = (const int8_t*) xq;
    const float* xs = (const float*) ((const uint8_t*) xq + xq_scales(n_in));
    // NS_MG_XMX_SHAPE: tiles a sub-group, sub-groups a work-group, blocks an iteration (default 7: 1, 4, 2 - B70, a
    // 19968 x 6656 matrix: 576 GB/s against 474 for 2, 4, 1)
    static const int shape = [] { const char* v = std::getenv("NS_MG_XMX_SHAPE"); return v ? std::atoi(v) : 7; }();
    switch (shape) {
        case 1: mmvq_xmx<4, 4>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 2: mmvq_xmx<2, 8>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 3: mmvq_xmx<1, 8>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 4: mmvq_xmx<2, 4, 2>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 5: mmvq_xmx<1, 8, 2>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 6: mmvq_xmx<2, 8, 2>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 8: mmvq_xmx<1, 4, 4>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 9: mmvq_xmx<1, 2, 4>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 10: mmvq_xmx<1, 8, 4>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 11: mmvq_xmx<2, 2, 2>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        case 0: mmvq_xmx<2, 4>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
        default: mmvq_xmx<1, 4, 2>(g->q, wq, ws, xqq, xs, y, n_in, n_out, ncols); break;
    }
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
