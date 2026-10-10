// attn.cpp: the qwen35 silo's prompt attention (silo.h: ns_q35_attn_prompt) - flash attention on XMX for a prompt
// pass's rows against the half cache, causal from position p0. The engine's own kernel (q35.cpp: attn) reads every key
// once per row (each work-group one row and key head): 3% of the card at 8K. Here a work-group takes 8 rows and the G
// query heads of one key head - 8 G query vectors - and streams the keys through local memory in chunks of CH:
//
//   1. the chunk's K and V into local memory, in the packed (VNNI) layouts XMX reads as its B operand;
//   2. S = Q K^T on joint_matrix (half x half -> float, 8x16x16; sub-group 16): two sub-groups a head, each over half
//      of the 256 features, their partial scores summed in step 3;
//   3. the online softmax a row (the running max, its rescale c, the sum), P as half;
//   4. O = O c + P V on joint_matrix, each sub-group its half of the features (O in registers: 8 tiles of 8x16).
//
// Q is scaled by 1/sqrt(D) log2(e) as it is staged (exp2 in the softmax). An accumulator tile's element e of lane l is
// (row e, column l) on this hardware (checked: joint_matrix_apply's order), which the row rescales rely on.
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
constexpr int D = 256, HALF_D = 128, SG = 16, M = 8, CH = 32, CHP = CH + 8;   // CHP: P's row stride (bank spread)

template <typename T>
inline auto lp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto gp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

// local memory, in 4-byte words: K packed [D/2][CH] (a word = features 2i, 2i+1 of a key), V packed [CH/2][D] (a word =
// keys 2i, 2i+1 of a feature), the partial scores [G][2][M][CH] float, P [G][M][CHP] half, the row stats [G][M] x 3
template <int G>
struct Slm {
    static constexpr int k = D / 2 * CH, v = CH / 2 * D, s = G * 2 * M * CH, p = G * M * CHP / 2, st = 3 * G * M;
    // Q staged as half [G][M][D] before the first chunk, over the K / V / score words (all free then)
    static constexpr int q = G * M * D / 2;
    static constexpr int words = (k + v + s > q ? k + v + s : q) + p + st;
};

template <int G>
void attn_prompt(sycl::queue& queue, const float* q, int64_t qs, const half* kc, const half* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap,
                 int64_t p0, float* out) {
    constexpr int NSG = 2 * G, WG = NSG * SG;
    using L = Slm<G>;
    const int64_t blocks = (T + M - 1) / M;
    const float qscale = 1.0f / sycl::sqrt((float) D) * 1.4426950408889634f;
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(L::words), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (blocks * Hkv) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int hg = sgi >> 1, hd = sgi & 1;   // this sub-group's head in the group, its half of the features
            const int64_t gi = it.get_group(0), hk = gi % Hkv, r0 = (gi / Hkv) * M;
            uint32_t* base = slm.get_multi_ptr<sycl::access::decorated::no>().get();
            uint32_t* kp = base;
            uint32_t* vp = kp + L::k;
            float* sp = (float*) (vp + L::v);
            half* pp = (half*) (base + (L::k + L::v + L::s > L::q ? L::k + L::v + L::s : L::q));
            float* mrow = (float*) ((uint32_t*) pp + L::p);
            float* lrow = mrow + G * M;
            float* crow = lrow + G * M;
            half* qh = (half*) base;
            // the last row's last key (rows past T take the last valid row's keys; they are not written)
            const int64_t rl = sycl::min(r0 + M, T) - 1, kv_end = p0 + rl + 1;

            // ---- Q, scaled, as half [G][M][D]; the row stats
            for (int i = tid; i < G * M * D; i += WG) {
                const int g = i / (M * D), r = (i / D) % M, d = i % D;
                const int64_t row = sycl::min(r0 + r, T - 1);
                qh[i] = (half) (q[row * qs + (hk * G + g) * D + d] * qscale);
            }
            for (int i = tid; i < G * M; i += WG) {
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            sycl::group_barrier(it.get_group());
            jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> qa[HALF_D / 16];
#pragma unroll
            for (int k = 0; k < HALF_D / 16; ++k)
                jm::joint_matrix_load(sg, qa[k], lp(qh + (hg * M) * D + hd * HALF_D + k * 16), (size_t) D);
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, 16> o[HALF_D / 16];
#pragma unroll
            for (int t = 0; t < HALF_D / 16; ++t) jm::joint_matrix_fill(sg, o[t], 0.0f);
            sycl::group_barrier(it.get_group());   // Q's words are the chunks' from here

            const half* kb = kc + hk * cap * D;
            const half* vb = vc + hk * cap * D;
            for (int64_t j0 = 0; j0 < kv_end; j0 += CH) {
                // ---- 1. the chunk's keys packed (keys past the last row's are zeros: masked below)
                for (int i = tid; i < CH * (D / 8); i += WG) {   // K: key n, features 8c .. 8c + 7 -> words [4c .. 4c + 3][n]
                    const int n = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 w(0, 0, 0, 0);
                    if (j0 + n < kv_end) w = *reinterpret_cast<const sycl::uint4*>(kb + (j0 + n) * D + 8 * c);
                    kp[(4 * c + 0) * CH + n] = w.x();
                    kp[(4 * c + 1) * CH + n] = w.y();
                    kp[(4 * c + 2) * CH + n] = w.z();
                    kp[(4 * c + 3) * CH + n] = w.w();
                }
                for (int i = tid; i < (CH / 2) * (D / 8); i += WG) {   // V: keys 2p, 2p + 1, features 8c .. -> words [p][8c ..]
                    const int pr = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 a(0, 0, 0, 0), b(0, 0, 0, 0);
                    if (j0 + 2 * pr < kv_end) a = *reinterpret_cast<const sycl::uint4*>(vb + (j0 + 2 * pr) * D + 8 * c);
                    if (j0 + 2 * pr + 1 < kv_end) b = *reinterpret_cast<const sycl::uint4*>(vb + (j0 + 2 * pr + 1) * D + 8 * c);
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
                        jm::joint_matrix_load(sg, kbm, lp(kph + ((hd * HALF_D + k * 16) / 2) * (CH * 2) + nt * 16 * 2), (size_t) (CH * 2));
                        jm::joint_matrix_mad(sg, s, qa[k], kbm, s);
                    }
                    jm::joint_matrix_store(sg, s, lp(sp + ((hg * 2 + hd) * M) * CH + nt * 16), (size_t) CH, jm::layout::row_major);
                }
                sycl::group_barrier(it.get_group());
                // ---- 3. softmax: sub-group (g, h) takes rows 4h .. 4h + 3 of head g, 4 lanes a row, CH / 4 keys a lane
                {
                    const int r = hd * 4 + lane / 4, part = lane % 4;
                    const int64_t lim = p0 + sycl::min(r0 + r, T - 1);   // the row's last key
                    const float* s0 = sp + ((hg * 2 + 0) * M + r) * CH;
                    const float* s1 = sp + ((hg * 2 + 1) * M + r) * CH;
                    float sv[CH / 4], mx = -INFINITY;
#pragma unroll
                    for (int i = 0; i < CH / 4; ++i) {
                        const int n = part * (CH / 4) + i;
                        sv[i] = j0 + n <= lim ? s0[n] + s1[n] : -INFINITY;
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
            const int64_t ostride = Hq * D;
            float* ob = out + r0 * ostride + (hk * G + hg) * D + hd * HALF_D;
            const bool whole = r0 + M <= T;
            float* scratch = sp + sgi * (M * 16);   // a tile's staging for a partial block (the chunks are done)
#pragma unroll
            for (int t = 0; t < HALF_D / 16; ++t) {
                int e = 0;
                jm::joint_matrix_apply(sg, o[t], [&](float& x) { x = lr[e] > 0.0f ? x / lr[e] : 0.0f; ++e; });
                if (whole) {
                    jm::joint_matrix_store(sg, o[t], gp(ob + t * 16), (size_t) ostride, jm::layout::row_major);
                } else {
                    jm::joint_matrix_store(sg, o[t], lp(scratch), (size_t) 16, jm::layout::row_major);
                    sycl::group_barrier(sg);
                    for (int r = 0; r < M; ++r)
                        if (r0 + r < T) ob[r * ostride + t * 16 + lane] = scratch[r * 16 + lane];
                    sycl::group_barrier(sg);
                }
            }
        });
    });
}

// ---- decode rows (at most 8, each at its own position): a work-group per (row, key head, split of the keys), NSG
// sub-groups taking 16 keys at a time. Scores: a lane a key (its K row against the G query heads, from local memory) -
// one reduction a head for 16 keys, not one a key; then P V with a lane 16 features (V rows read whole, the
// probabilities broadcast). The sub-groups' (max, sum, values) merged in local memory; the splits by a second kernel.
constexpr int NSG_D = 4, KEYS_D = NSG_D * 16;

// keys a split: ~512 work-groups (key heads x splits) whatever the context, at least a work-group's step
int64_t dec_split(int64_t kv, int64_t Hkv) {
    const int64_t want = (kv * Hkv + 511) / 512;
    return sycl::max<int64_t>(KEYS_D, (want + KEYS_D - 1) / KEYS_D * KEYS_D);
}
int64_t dec_splits(int64_t kv, int64_t Hkv) {
    return (kv + dec_split(kv, Hkv) - 1) / dec_split(kv, Hkv);
}

template <int G>
void attn_decode(sycl::queue& queue, const float* q, int64_t qs, const half* kc, const half* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap,
                 int64_t p0, float* out, float* part) {
    constexpr int WG = NSG_D * SG, DL = D / SG;
    // the longest row's splits for all rows (a shorter row's last splits are empty)
    const int64_t ns = dec_splits(p0 + T, Hkv), kb = dec_split(p0 + T, Hkv);
    const float qscale = 1.0f / sycl::sqrt((float) D) * 1.4426950408889634f;
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qsl(sycl::range<1>(G * D), h);
        sycl::local_accessor<float, 1> red(sycl::range<1>(NSG_D * G * (D + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * Hkv * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, r = gi / Hkv;
            const int64_t kv_end = p0 + r + 1, j0 = sp * kb, j1 = sycl::min(kv_end, (sp + 1) * kb);
            for (int i = tid; i < G * D; i += WG) qsl[i] = q[r * qs + hk * G * D + i] * qscale;
            sycl::group_barrier(it.get_group());
            float m[G], l[G], acc[G][DL];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                m[g] = -INFINITY;
                l[g] = 0.0f;
#pragma unroll
                for (int i = 0; i < DL; ++i) acc[g][i] = 0.0f;
            }
            const half* kb_ = kc + hk * cap * D;
            const half* vb_ = vc + hk * cap * D;
            for (int64_t jb = j0 + sgi * 16; jb < j1; jb += KEYS_D) {
                // scores: lane = key jb + lane
                const int64_t j = jb + lane;
                const bool valid = j < j1;
                float sc[G];
#pragma unroll
                for (int g = 0; g < G; ++g) sc[g] = 0.0f;
                if (valid) {
                    const sycl::uint4* kr = reinterpret_cast<const sycl::uint4*>(kb_ + j * D);
#pragma unroll 4
                    for (int c = 0; c < D / 8; ++c) {
                        const sycl::uint4 w = kr[c];
                        const uint32_t wv[4] = {w.x(), w.y(), w.z(), w.w()};
                        float kf[8];
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            kf[2 * e] = (float) sycl::bit_cast<half>(uint16_t(wv[e] & 0xffff));
                            kf[2 * e + 1] = (float) sycl::bit_cast<half>(uint16_t(wv[e] >> 16));
                        }
#pragma unroll
                        for (int g = 0; g < G; ++g)
#pragma unroll
                            for (int e = 0; e < 8; ++e) sc[g] += qsl[g * D + 8 * c + e] * kf[e];
                    }
                }
                // the online softmax a head over these 16 keys
                float p[G];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float s = valid ? sc[g] : -INFINITY;
                    const float mx = sycl::reduce_over_group(sg, s, sycl::maximum<float>());
                    const float mn = sycl::fmax(m[g], mx);
                    const float c = mn == -INFINITY ? 1.0f : sycl::exp2(m[g] - mn);
                    p[g] = valid ? sycl::exp2(s - mn) : 0.0f;
                    l[g] = l[g] * c + sycl::reduce_over_group(sg, p[g], sycl::plus<float>());
                    m[g] = mn;
#pragma unroll
                    for (int i = 0; i < DL; ++i) acc[g][i] *= c;
                }
                // P V: lane = features lane DL .. + DL - 1, the 16 keys in turn
                const int nk = (int) sycl::min<int64_t>(16, j1 - jb);
                for (int k = 0; k < nk; ++k) {
                    const sycl::vec<half, DL> vv = *reinterpret_cast<const sycl::vec<half, DL>*>(vb_ + (jb + k) * D + lane * DL);
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const float pk = sycl::select_from_group(sg, p[g], k);
#pragma unroll
                        for (int i = 0; i < DL; ++i) acc[g][i] += pk * (float) vv[i];
                    }
                }
            }
            // the sub-groups merged; the split's (values, max, sum) out
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float* o = &red[(sgi * G + g) * (D + 2)];
#pragma unroll
                for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
                if (lane == 0) {
                    o[D] = m[g];
                    o[D + 1] = l[g];
                }
            }
            sycl::group_barrier(it.get_group());
            for (int idx = tid; idx < G * D; idx += WG) {
                const int g = idx / D, d = idx % D;
                float mx = -INFINITY;
                for (int k = 0; k < NSG_D; ++k) mx = sycl::fmax(mx, red[(k * G + g) * (D + 2) + D]);
                float Ls = 0.0f, A = 0.0f;
                if (mx != -INFINITY) {
                    for (int k = 0; k < NSG_D; ++k) {
                        const float* o = &red[(k * G + g) * (D + 2)];
                        const float e = sycl::exp2(o[D] - mx);
                        Ls += o[D + 1] * e;
                        A += o[d] * e;
                    }
                }
                const int64_t hq = hk * G + g;
                if (ns == 1) {
                    out[r * Hq * D + hq * D + d] = Ls > 0.0f ? A / Ls : 0.0f;
                } else {
                    float* pp = part + ((r * Hq + hq) * ns + sp) * (D + 2);
                    pp[d] = A;
                    if (d == 0) {
                        pp[D] = mx;
                        pp[D + 1] = Ls;
                    }
                }
            }
        });
    });
    if (ns > 1) {   // the splits merged (their maxima in log2 units)
        queue.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * Hq) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
            const int64_t rh = it.get_group(0), d = it.get_local_id(0);
            const float* pp = part + rh * ns * (D + 2);
            float mx = -INFINITY;
            for (int64_t k = 0; k < ns; ++k) mx = sycl::fmax(mx, pp[k * (D + 2) + D]);
            float Ls = 0.0f, A = 0.0f;
            if (mx != -INFINITY) {
                for (int64_t k = 0; k < ns; ++k) {
                    const float e = sycl::exp2(pp[k * (D + 2) + D] - mx);
                    Ls += pp[k * (D + 2) + D + 1] * e;
                    A += pp[k * (D + 2) + d] * e;
                }
            }
            out[rh * D + d] = Ls > 0.0f ? A / Ls : 0.0f;
        });
    }
}
}  // namespace

extern "C" {

int ns_q35_attn_prompt_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh) {
    return T > 8 && Dh == D && Hkv > 0 && Hq == 6 * Hkv;
}

int ns_q35_attn_prompt(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh,
                       int64_t cap, int64_t p0, float* out) {
    NS_TRY
    if (!ns_q35_attn_prompt_supported(T, Hq, Hkv, Dh)) return ns_fail("ns_q35_attn_prompt: more than 8 rows, heads of 256, 6 query heads a key head");
    attn_prompt<6>(g->q, q, qs, (const half*) kc, (const half*) vc, T, Hq, Hkv, cap, p0, out);
    return 0;
    NS_CATCH
}

int ns_q35_attn_decode_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh) {
    return T >= 1 && T <= 8 && Dh == D && Hkv > 0 && Hq == 6 * Hkv;
}

int64_t ns_q35_attn_decode_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh, int64_t p0) {
    const int64_t ns = dec_splits(p0 + T, Hkv);
    return ns > 1 ? T * Hq * ns * (Dh + 2) : 0;
}

int ns_q35_attn_decode(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh,
                       int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_TRY
    if (!ns_q35_attn_decode_supported(T, Hq, Hkv, Dh)) return ns_fail("ns_q35_attn_decode: 1..8 rows, heads of 256, 6 query heads a key head");
    if (ns_q35_attn_decode_scratch(T, Hq, Hkv, Dh, p0) > 0 && !scratch) return ns_fail("ns_q35_attn_decode: no scratch");
    attn_decode<6>(g->q, q, qs, (const half*) kc, (const half*) vc, T, Hq, Hkv, cap, p0, out, scratch);
    return 0;
    NS_CATCH
}

}  // extern "C"
