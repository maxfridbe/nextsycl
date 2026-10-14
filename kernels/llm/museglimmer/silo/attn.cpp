// attn.cpp: the museglimmer silo's prompt attention (silo.h: ns_mg_attn_prompt) - flash attention on XMX for a prompt
// pass's rows against the half ring caches, causal from position p0 and within the window. Started from the qwen35
// silo's (kernels/llm/qwen35/silo/attn.cpp: heads of 256, 6 query heads a key head), changed for Muse Glimmer: heads
// of 128, the 16 query heads of a key head in one work-group, a row's keys from max(0, p - window + 1), a key's slot
// its position mod the ring. The engine's own kernel (mg.cpp: attn) reads every key once a row and head group on the
// vector units: a 4K prompt's attention was half its time. A work-group takes 8 rows and the G query heads of a key
// head - 8 G query vectors - and streams the keys through local memory in chunks of CH:
//
//   1. the chunk's K and V into local memory, in the packed (VNNI) layouts XMX reads as its B operand;
//   2. S = Q K^T on joint_matrix (half x half -> float, 8x16x16; sub-group 16): two sub-groups a head, each over half
//      of the 128 features, their partial scores summed in step 3;
//   3. the online softmax a row (the running max, its rescale c, the sum), P as half;
//   4. O = O c + P V on joint_matrix, each sub-group its half of the features (O in registers: 4 tiles of 8x16).
//
// Q is scaled by 1/sqrt(D) log2(e) as it is staged (exp2 in the softmax). An accumulator tile's element e of lane l is
// (row e, column l) on this hardware, which the row rescales rely on.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>
#include <cstdint>

namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
using half = sycl::half;
// keys a chunk (the qwen35 silo's measure: 16; more local memory a work-group holds fewer share an Xe core)
constexpr int D = 128, HALF_D = 64, SG = 16, M = 8, CH = 16, CHP = CH + 8, KS = CH;

template <typename T>
inline auto lp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto gp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

// local memory, in 4-byte words: K packed [D/2][CH], V packed [CH/2][D], the partial scores [G][2][M][CH] float, P
// [G][M][CHP] half, the row stats [G][M] x 3; Q staged as half [G][M][64] before the first chunk, over the K / V /
// score words
template <int G>
struct Slm {
    static constexpr int k = D / 2 * KS, v = CH / 2 * D, s = G * 2 * M * CH, p = G * M * CHP / 2, st = 3 * G * M;
    static constexpr int q = G * M * HALF_D / 2;
    static constexpr int words = (k + v + s > q ? k + v + s : q) + p + st;
};

template <int G>
void attn_prompt(sycl::queue& queue, const float* q, const half* kc, const half* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0,
                 int64_t win, float* out) {
    constexpr int NSG = 2 * G, WG = NSG * SG;
    using L = Slm<G>;
    const int64_t blocks = (T + M - 1) / M, groups = Hq / G, per_kv = Hq / Hkv;
    const int64_t qs = Hq * D;
    const float qscale = 1.0f / sycl::sqrt((float) D) * 1.4426950408889634f;
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(L::words), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (blocks * groups) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int hg = sgi >> 1, hd = sgi & 1;   // this sub-group's head in the group, its half of the features
            const int64_t gi = it.get_group(0), grp = gi % groups, r0 = (gi / groups) * M;
            const int64_t h0 = grp * G, hk = h0 / per_kv;
            uint32_t* base = slm.get_multi_ptr<sycl::access::decorated::no>().get();
            uint32_t* kp = base;
            uint32_t* vp = kp + L::k;
            float* sp = (float*) (vp + L::v);
            half* pp = (half*) (base + (L::k + L::v + L::s > L::q ? L::k + L::v + L::s : L::q));
            float* mrow = (float*) ((uint32_t*) pp + L::p);
            float* lrow = mrow + G * M;
            float* crow = lrow + G * M;
            half* qh = (half*) base;
            // the block's keys: from the first row's first to the last row's last (rows past T take the last valid row)
            const int64_t rl = sycl::min(r0 + M, T) - 1, jend = p0 + rl + 1;
            const int64_t jbeg = win > 0 ? sycl::max<int64_t>(0, p0 + r0 - win + 1) : 0;

            // ---- Q, scaled, as half: one half of the features at a time; the row stats
            for (int i = tid; i < G * M; i += WG) {
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> qa[HALF_D / 16];
            for (int hh = 0; hh < 2; ++hh) {
                for (int i = tid; i < G * M * HALF_D; i += WG) {
                    const int g = i / (M * HALF_D), r = (i / HALF_D) % M, d = i % HALF_D;
                    const int64_t row = sycl::min(r0 + r, T - 1);
                    qh[i] = (half) (q[row * qs + (h0 + g) * D + hh * HALF_D + d] * qscale);
                }
                sycl::group_barrier(it.get_group());
                if (hd == hh) {
#pragma unroll
                    for (int k = 0; k < HALF_D / 16; ++k)
                        jm::joint_matrix_load(sg, qa[k], lp(qh + (hg * M) * HALF_D + k * 16), (size_t) HALF_D);
                }
                sycl::group_barrier(it.get_group());
            }
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> o[HALF_D / 16];
#pragma unroll
            for (int t = 0; t < HALF_D / 16; ++t) jm::joint_matrix_fill(sg, o[t], 0.0f);

            const half* kb = kc + hk * cap * D;
            const half* vb = vc + hk * cap * D;
            for (int64_t j0 = jbeg; j0 < jend; j0 += CH) {
                // ---- 1. the chunk's keys packed (keys past the block's last are zeros: masked below)
                for (int i = tid; i < CH * (D / 8); i += WG) {   // K: key n, features 8c .. 8c + 7 -> words [4c .. 4c + 3][n]
                    const int n = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 w(0, 0, 0, 0);
                    if (j0 + n < jend) w = *reinterpret_cast<const sycl::uint4*>(kb + ((j0 + n) % cap) * D + 8 * c);
                    kp[(4 * c + 0) * KS + n] = w.x();
                    kp[(4 * c + 1) * KS + n] = w.y();
                    kp[(4 * c + 2) * KS + n] = w.z();
                    kp[(4 * c + 3) * KS + n] = w.w();
                }
                for (int i = tid; i < (CH / 2) * (D / 8); i += WG) {   // V: keys 2p, 2p + 1, features 8c .. -> words [p][8c ..]
                    const int pr = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 a(0, 0, 0, 0), b(0, 0, 0, 0);
                    if (j0 + 2 * pr < jend) a = *reinterpret_cast<const sycl::uint4*>(vb + ((j0 + 2 * pr) % cap) * D + 8 * c);
                    if (j0 + 2 * pr + 1 < jend) b = *reinterpret_cast<const sycl::uint4*>(vb + ((j0 + 2 * pr + 1) % cap) * D + 8 * c);
                    const uint32_t av[4] = {a.x(), a.y(), a.z(), a.w()}, bv[4] = {b.x(), b.y(), b.z(), b.w()};
                    uint32_t* dst = vp + pr * D + 8 * c;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        dst[2 * e] = (av[e] & 0xffffu) | (bv[e] << 16);
                        dst[2 * e + 1] = (av[e] >> 16) | (bv[e] & 0xffff0000u);
                    }
                }
                sycl::group_barrier(it.get_group());
                // ---- 2. partial scores over this sub-group's features
                const half* kph = (const half*) kp;
#pragma unroll
                for (int nt = 0; nt < CH / 16; ++nt) {
                    jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> s;
                    jm::joint_matrix_fill(sg, s, 0.0f);
#pragma unroll
                    for (int k = 0; k < HALF_D / 16; ++k) {
                        jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> kbm;
                        jm::joint_matrix_load(sg, kbm, lp(kph + ((hd * HALF_D + k * 16) / 2) * (KS * 2) + nt * 16 * 2), (size_t) (KS * 2));
                        jm::joint_matrix_mad(sg, s, qa[k], kbm, s);
                    }
                    jm::joint_matrix_store(sg, s, lp(sp + ((hg * 2 + hd) * M) * CH + nt * 16), (size_t) CH, jm::layout::row_major);
                }
                sycl::group_barrier(it.get_group());
                // ---- 3. softmax: sub-group (g, h) takes rows 4h .. 4h + 3 of head g, 4 lanes a row, CH / 4 keys a lane
                {
                    const int r = hd * 4 + lane / 4, part = lane % 4;
                    const int64_t pos = p0 + sycl::min(r0 + r, T - 1);   // the row's position: its last key
                    const int64_t first = win > 0 ? pos - win + 1 : 0;
                    const float* s0 = sp + ((hg * 2 + 0) * M + r) * CH;
                    const float* s1 = sp + ((hg * 2 + 1) * M + r) * CH;
                    float sv[CH / 4], mx = -INFINITY;
#pragma unroll
                    for (int i = 0; i < CH / 4; ++i) {
                        const int n = part * (CH / 4) + i;
                        const int64_t j = j0 + n;
                        sv[i] = j <= pos && j >= first && j < jend ? s0[n] + s1[n] : -INFINITY;
                        mx = sycl::fmax(mx, sv[i]);
                    }
                    mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 1));
                    mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 2));
                    const int si = hg * M + r;
                    const float mo = mrow[si], mn = sycl::fmax(mo, mx);
                    const float c = mn == -INFINITY ? 1.0f : sycl::exp2(mo - mn);
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
                // ---- 4. O = O c + P V over this sub-group's features
                {
                    const float* cr = crow + hg * M;
#pragma unroll
                    for (int t = 0; t < HALF_D / 16; ++t) {
                        int e = 0;
                        jm::joint_matrix_apply(sg, o[t], [&](float& x) { x *= cr[e]; ++e; });
                    }
                    const half* vph = (const half*) vp;
#pragma unroll
                    for (int ks = 0; ks < CH / 16; ++ks) {
                        jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> pa;
                        jm::joint_matrix_load(sg, pa, lp(pp + (hg * M) * CHP + ks * 16), (size_t) CHP);
#pragma unroll
                        for (int t = 0; t < HALF_D / 16; ++t) {
                            jm::joint_matrix<sycl::sub_group, half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> vbm;
                            jm::joint_matrix_load(sg, vbm, lp(vph + (ks * 16 / 2) * (D * 2) + (hd * HALF_D + t * 16) * 2), (size_t) (D * 2));
                            jm::joint_matrix_mad(sg, o[t], pa, vbm, o[t]);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());   // the chunk's words free for the next
            }
            // ---- the rows' outputs: O / l; rows past T are not written
            const float* lr = lrow + hg * M;
            float* ob = out + r0 * qs + (h0 + hg) * D + hd * HALF_D;
            const bool whole = r0 + M <= T;
            float* scratch = sp + sgi * (M * 16);   // a tile's staging for a partial block (the chunks are done)
#pragma unroll
            for (int t = 0; t < HALF_D / 16; ++t) {
                int e = 0;
                jm::joint_matrix_apply(sg, o[t], [&](float& x) { x = lr[e] > 0.0f ? x / lr[e] : 0.0f; ++e; });
                if (whole) {
                    jm::joint_matrix_store(sg, o[t], gp(ob + t * 16), (size_t) qs, jm::layout::row_major);
                } else {
                    jm::joint_matrix_store(sg, o[t], lp(scratch), (size_t) 16, jm::layout::row_major);
                    sycl::group_barrier(sg);
                    for (int r = 0; r < M; ++r)
                        if (r0 + r < T) ob[r * qs + t * 16 + lane] = scratch[r * 16 + lane];
                    sycl::group_barrier(sg);
                }
            }
        });
    });
}
}  // namespace

extern "C" {

int ns_mg_attn_prompt_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh) {
    return T > 8 && Dh == D && Hkv > 0 && Hq == 16 * Hkv;
}

int ns_mg_attn_prompt(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh, int64_t cap, int64_t p0,
                      int64_t win, float* out) {
    NS_TRY
    if (!ns_mg_attn_prompt_supported(T, Hq, Hkv, Dh)) return ns_fail("ns_mg_attn_prompt: more than 8 rows, heads of 128, 16 query heads a key head");
    const int64_t need = win > 0 ? sycl::min(p0 + T, win + T - 1) : p0 + T;
    if (need > cap) return ns_fail("ns_mg_attn_prompt: the cache's ring is shorter than the window and the pass");
    attn_prompt<16>(g->q, q, (const half*) kc, (const half*) vc, T, Hq, Hkv, cap, p0, win, out);
    return 0;
    NS_CATCH
}

}  // extern "C"
