// attnx.cpp: the minimaxm3 silo's prompt attention on XMX, the qwen35 silo's flash kernel (attn.cpp there) shaped for this model: a work-group is one (query row, key/value
// group) and its 16 query heads, 8 heads a DPAS row set, 128 values a head; the keys are the row's selected blocks (or all before the query on a dense layer), staged 16 at
// a time. Q is half, scaled by 1/sqrt(128) log2(e) (exp2 in the softmax); the scores and the output accumulate in float.
#include "ns.h"
#include "ns_internal.hpp"
#include "mm3s.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>
#include <cstdint>

namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
using half = sycl::half;
constexpr int D = 128, HALF_D = 64, SG = 16, M = 8, CH = 16, CHP = CH + 8, KS = CH, NG = 2;  // NG: DPAS row sets a group's 16 heads make

template <typename T>
inline auto lp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

template <typename T>
inline auto gp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

// local memory, in 4-byte words: K packed [D/2][CH], V packed [CH/2][D], the partial scores [NG][2][M][CH], P [NG][M][CHP] half, the row stats [NG][M] x 3; Q before the first chunk
struct Slm {
    static constexpr int k = D / 2 * KS, v = CH / 2 * D, s = NG * 2 * M * CH, p = NG * M * CHP / 2, st = 3 * NG * M;
    static constexpr int q = NG * M * HALF_D / 2;
    static constexpr int words = (k + v + s > q ? k + v + s : q) + p + st;
};
}  // namespace

extern "C" {

int ns_mm3s_attend_x(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T, int64_t H, int64_t G, int64_t pos0, float /*scale*/) {
    NS_TRY
    if (!ns_mm3s_attend_supported(H, G, D)) return ns_fail("ns_mm3s_attend_x: 16 heads a group of 128 values");
    constexpr int NSG = 2 * NG, WG = NSG * SG;
    const float qscale = 1.0f / sycl::sqrt((float) D) * 1.4426950408889634f;
    const uint8_t* kc8 = (const uint8_t*) kc;
    const uint8_t* vc8 = (const uint8_t*) vc;
    using L = Slm;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(L::words), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * G) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int hg = sgi >> 1, hd = sgi & 1;  // this sub-group's row set and its half of the features
            const int64_t gid = it.get_group(0), t = gid / G, gi = gid % G, pos = pos0 + t;
            uint32_t* base = slm.get_multi_ptr<sycl::access::decorated::no>().get();
            uint32_t* kp = base;
            uint32_t* vp = kp + L::k;
            float* sp = (float*) (vp + L::v);
            half* pp = (half*) (base + (L::k + L::v + L::s > L::q ? L::k + L::v + L::s : L::q));
            float* mrow = (float*) ((uint32_t*) pp + L::p);
            float* lrow = mrow + NG * M;
            float* crow = lrow + NG * M;
            half* qh = (half*) base;
            const int64_t total = sel ? K * 128 : pos + 1;

            for (int i = tid; i < NG * M; i += WG) {
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> qa[HALF_D / 16];
            for (int hh = 0; hh < 2; ++hh) {
                for (int i = tid; i < NG * M * HALF_D; i += WG) {
                    const int head = i / HALF_D, d = i % HALF_D;
                    qh[i] = (half) (q[(t * H + gi * 16 + head) * D + hh * HALF_D + d] * qscale);
                }
                sycl::group_barrier(it.get_group());
                if (hd == hh) {
#pragma unroll
                    for (int k = 0; k < HALF_D / 16; ++k) jm::joint_matrix_load(sg, qa[k], lp(qh + (hg * M) * HALF_D + k * 16), (size_t) HALF_D);
                }
                sycl::group_barrier(it.get_group());
            }
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> o[HALF_D / 16];
#pragma unroll
            for (int u = 0; u < HALF_D / 16; ++u) jm::joint_matrix_fill(sg, o[u], 0.0f);

            for (int64_t c0 = 0; c0 < total; c0 += CH) {
                // the chunk's first key; 16 keys in a row (a block is 128 of them)
                int64_t j0 = c0;
                bool none = false;
                if (sel) {
                    const int32_t b = sel[(t * G + gi) * K + c0 / 128];
                    if (b < 0) none = true;
                    j0 = (int64_t) b * 128 + c0 % 128;
                }
                const int n_here = (int) sycl::min<int64_t>(CH, total - c0);
                // the keys this query may see: j0 + n <= pos, n < n_here
                const auto vis = [=](int n) { return !none && n < n_here && j0 + n <= pos; };
                for (int i = tid; i < CH * (D / 8); i += WG) {  // K: key n, features 8c .. 8c + 7 -> words [4c .. 4c + 3][n]
                    const int n = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 w(0, 0, 0, 0);
                    if (vis(n)) w = *reinterpret_cast<const sycl::uint4*>(kc8 + ((j0 + n) * G + gi) * D * 2 + 16 * c);
                    kp[(4 * c + 0) * KS + n] = w.x();
                    kp[(4 * c + 1) * KS + n] = w.y();
                    kp[(4 * c + 2) * KS + n] = w.z();
                    kp[(4 * c + 3) * KS + n] = w.w();
                }
                for (int i = tid; i < (CH / 2) * (D / 8); i += WG) {  // V: keys 2p, 2p + 1, features 8c .. -> words [p][8c ..]
                    const int pr = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 a(0, 0, 0, 0), b(0, 0, 0, 0);
                    if (vis(2 * pr)) a = *reinterpret_cast<const sycl::uint4*>(vc8 + ((j0 + 2 * pr) * G + gi) * D * 2 + 16 * c);
                    if (vis(2 * pr + 1)) b = *reinterpret_cast<const sycl::uint4*>(vc8 + ((j0 + 2 * pr + 1) * G + gi) * D * 2 + 16 * c);
                    const uint32_t av[4] = {a.x(), a.y(), a.z(), a.w()}, bv[4] = {b.x(), b.y(), b.z(), b.w()};
                    uint32_t* dst = vp + pr * D + 8 * c;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        dst[2 * e] = (av[e] & 0xffffu) | (bv[e] << 16);
                        dst[2 * e + 1] = (av[e] >> 16) | (bv[e] & 0xffff0000u);
                    }
                }
                sycl::group_barrier(it.get_group());
                // partial scores over this sub-group's features
                const half* kph = (const half*) kp;
                {
                    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> s;
                    jm::joint_matrix_fill(sg, s, 0.0f);
#pragma unroll
                    for (int k = 0; k < HALF_D / 16; ++k) {
                        jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> kbm;
                        jm::joint_matrix_load(sg, kbm, lp(kph + ((hd * HALF_D + k * 16) / 2) * (KS * 2)), (size_t) (KS * 2));
                        jm::joint_matrix_mad(sg, s, qa[k], kbm, s);
                    }
                    jm::joint_matrix_store(sg, s, lp(sp + ((hg * 2 + hd) * M) * CH), (size_t) CH, jm::layout::row_major);
                }
                sycl::group_barrier(it.get_group());
                // softmax: sub-group (hg, hd) takes rows 4 hd .. 4 hd + 3 of the set, 4 lanes a row, CH / 4 keys a lane
                {
                    const int r = hd * 4 + lane / 4, part = lane % 4;
                    const float* s0 = sp + ((hg * 2 + 0) * M + r) * CH;
                    const float* s1 = sp + ((hg * 2 + 1) * M + r) * CH;
                    float sv[CH / 4], mx = -INFINITY;
#pragma unroll
                    for (int i = 0; i < CH / 4; ++i) {
                        const int n = part * (CH / 4) + i;
                        sv[i] = vis(n) ? s0[n] + s1[n] : -INFINITY;
                        mx = sycl::fmax(mx, sv[i]);
                    }
                    mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 1));
                    mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 2));
                    const int si = hg * M + r;
                    const float mo = mrow[si], mn = sycl::fmax(mo, mx);
                    const float c = (mn == -INFINITY || mo == -INFINITY) ? (mo == -INFINITY ? 0.0f : 1.0f) : sycl::exp2(mo - mn);
                    float sum = 0.0f;
                    half* prow = pp + (hg * M + r) * CHP;
#pragma unroll
                    for (int i = 0; i < CH / 4; ++i) {
                        const float pv = sv[i] == -INFINITY ? 0.0f : sycl::exp2(sv[i] - mn);
                        sum += pv;
                        prow[part * (CH / 4) + i] = (half) pv;
                    }
                    sum += sycl::permute_group_by_xor(sg, sum, 1);
                    sum += sycl::permute_group_by_xor(sg, sum, 2);
                    if (part == 0) {
                        mrow[si] = mn;
                        lrow[si] = lrow[si] * c + sum;
                        crow[si] = c;
                    }
                }
                sycl::group_barrier(it.get_group());
                // O = O c + P V over this sub-group's features
                {
                    const float* cr = crow + hg * M;
#pragma unroll
                    for (int u = 0; u < HALF_D / 16; ++u) {
                        int e = 0;
                        jm::joint_matrix_apply(sg, o[u], [&](float& x) { x *= cr[e]; ++e; });
                    }
                    const half* vph = (const half*) vp;
                    jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> pa;
                    jm::joint_matrix_load(sg, pa, lp(pp + (hg * M) * CHP), (size_t) CHP);
#pragma unroll
                    for (int u = 0; u < HALF_D / 16; ++u) {
                        jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> vbm;
                        jm::joint_matrix_load(sg, vbm, lp(vph + (hd * HALF_D + u * 16) * 2), (size_t) (D * 2));
                        jm::joint_matrix_mad(sg, o[u], pa, vbm, o[u]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            // the heads' outputs: O / l
            const float* lr = lrow + hg * M;
            float* ob = out + (t * H + gi * 16 + hg * M) * D + hd * HALF_D;
#pragma unroll
            for (int u = 0; u < HALF_D / 16; ++u) {
                int e = 0;
                jm::joint_matrix_apply(sg, o[u], [&](float& x) { x = lr[e] > 0.0f ? x / lr[e] : 0.0f; ++e; });
                jm::joint_matrix_store(sg, o[u], gp(ob + u * 16), (size_t) D, jm::layout::row_major);
            }
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
