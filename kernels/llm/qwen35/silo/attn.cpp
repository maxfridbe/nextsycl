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
// Decode rows (ns_q35_attn_decode, at most 8) run the same kernel in its SPLIT form: the rows together as the DPAS
// rows, a work-group per key split, the splits merged after (B70, UD-Q4_K_M with MTP at 40K: 36 -> 47 tok/s - the
// lane-a-key kernel it replaced read the cache once a row and ran its products on the vector units).
//
// Q is scaled by 1/sqrt(D) log2(e) as it is staged (exp2 in the softmax). An accumulator tile's element e of lane l is
// (row e, column l) on this hardware (checked: joint_matrix_apply's order), which the row rescales rely on.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
using half = sycl::half;
// keys a chunk: the local memory a work-group holds sets how many share an Xe core, and that bounds this kernel
// (B70, a 40K prompt: 32 keys 76 s, 64 keys 106 s - one work-group a core -, 16 keys 50 s)
#ifndef Q35_ATTN_CH
#define Q35_ATTN_CH 16
#endif
#ifndef Q35_ATTN_KPAD
#define Q35_ATTN_KPAD 0
#endif
// CHP: P's row stride; KS: packed K's row stride in words - a key's 4 words go KS apart, and KS = CH would put a
// sub-group's stores on one bank (padding measured no faster at 16 keys: 0)
constexpr int D = 256, HALF_D = 128, SG = 16, M = 8, CH = Q35_ATTN_CH, CHP = CH + 8, KS = CH + Q35_ATTN_KPAD;

// the cache's row of a key head and position, by its form (KT): 0 half [256], 1 q8_0 (the 256 int8 values, then 8
// half scales: 272 bytes), 2 q4_0 (8 blocks of 16 bytes - byte j of a block holds value j low, value j + 16 high - then
// 8 half scales: 144 bytes); llama.cpp's q8_0 / q4_0 blocks with the scales moved behind the values
template <int KT> constexpr int64_t kv_row() { return KT == 0 ? 512 : KT == 1 ? 272 : 144; }

// features 8c .. 8c + 7 of a row as 8 halves (packed in a uint4)
template <int KT>
inline sycl::uint4 kv_load8(const uint8_t* row, int c) {
    if constexpr (KT == 0) {
        return *reinterpret_cast<const sycl::uint4*>(row + 16 * c);
    } else {
        float f[8];
        if constexpr (KT == 1) {
            const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(row + 8 * c);
            const float d = (float) reinterpret_cast<const half*>(row + 256)[c >> 2];
            const uint32_t ww[2] = {w.x(), w.y()};
#pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = (float) (int8_t) ((ww[i >> 2] >> (8 * (i & 3))) & 0xff) * d;
        } else {
            const int blk = c >> 2, sub = c & 3;
            const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(row + blk * 16 + (sub & 1) * 8);
            const float d = (float) reinterpret_cast<const half*>(row + 128)[blk];
            const int sh = sub >= 2 ? 4 : 0;
            const uint32_t ww[2] = {w.x(), w.y()};
#pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = (float) ((int) (((ww[i >> 2] >> (8 * (i & 3))) >> sh) & 15) - 8) * d;
        }
        uint32_t o[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half a = (half) f[2 * i], b = (half) f[2 * i + 1];
            o[i] = (uint32_t) sycl::bit_cast<uint16_t>(a) | ((uint32_t) sycl::bit_cast<uint16_t>(b) << 16);
        }
        return sycl::uint4(o[0], o[1], o[2], o[3]);
    }
}

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
    static constexpr int k = D / 2 * KS, v = CH / 2 * D, s = G * 2 * M * CH, p = G * M * CHP / 2, st = 3 * G * M;
    // Q staged as half [G][M][128] - one half of the features at a time - before the first chunk, over the K / V /
    // score words (all free then)
    static constexpr int q = G * M * HALF_D / 2;
    static constexpr int words = (k + v + s > q ? k + v + s : q) + p + st;
};

// SPLIT (decode: T <= 8 rows, one block): a work-group per (key head, split of kb keys), each row's partial (values
// unnormalized, max, sum) into `part` for the merge - or its output when the row has one split. A row's arithmetic
// does not depend on the other rows (the DPAS rows are independent, the masks and the softmax a row's own) nor on
// how many there are, and its splits come from its own position: a verify row is the one-row decode's, bit for bit.
int64_t dec_splits(int64_t kv, int64_t Hkv);
template <int G, bool SPLIT, int KT>
void attn_flash(sycl::queue& queue, const float* q, int64_t qs, const uint8_t* kc, const uint8_t* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap,
                int64_t p0, float* out, int64_t kbs, int64_t ns, float* part) {
    constexpr int64_t RB = kv_row<KT>();
    constexpr int NSG = 2 * G, WG = NSG * SG;
    using L = Slm<G>;
    const int64_t blocks = SPLIT ? ns : (T + M - 1) / M;
    const float qscale = 1.0f / sycl::sqrt((float) D) * 1.4426950408889634f;
    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(L::words), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (blocks * Hkv) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int hg = sgi >> 1, hd = sgi & 1;   // this sub-group's head in the group, its half of the features
            const int64_t gi = it.get_group(0), hk = gi % Hkv, r0 = SPLIT ? 0 : (gi / Hkv) * M, spl = SPLIT ? gi / Hkv : 0;
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
            // this work-group's keys (decode: its split)
            const int64_t jbeg = SPLIT ? spl * kbs : 0, jend = SPLIT ? sycl::min(kv_end, (spl + 1) * kbs) : kv_end;
            if (jbeg >= jend) return;   // the whole work-group, before any barrier

            // ---- Q, scaled, as half: one half of the features at a time ([G][M][128]), each sub-group loading its
            // half's fragments; the row stats
            for (int i = tid; i < G * M; i += WG) {
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            jm::joint_matrix<sycl::sub_group, half, jm::use::a, M, 16, jm::layout::row_major> qa[HALF_D / 16];
            for (int hh = 0; hh < 2; ++hh) {
                for (int i = tid; i < G * M * HALF_D; i += WG) {
                    const int g = i / (M * HALF_D), r = (i / HALF_D) % M, d = i % HALF_D;
                    const int64_t row = sycl::min(r0 + r, T - 1);
                    qh[i] = (half) (q[row * qs + (hk * G + g) * D + hh * HALF_D + d] * qscale);
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

            const uint8_t* kb = kc + hk * cap * RB;
            const uint8_t* vb = vc + hk * cap * RB;
            for (int64_t j0 = jbeg; j0 < jend; j0 += CH) {
                // ---- 1. the chunk's keys packed (keys past the last row's are zeros: masked below)
                for (int i = tid; i < CH * (D / 8); i += WG) {   // K: key n, features 8c .. 8c + 7 -> words [4c .. 4c + 3][n]
                    const int n = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 w(0, 0, 0, 0);
                    if (j0 + n < jend) w = kv_load8<KT>(kb + (j0 + n) * RB, c);
                    kp[(4 * c + 0) * KS + n] = w.x();
                    kp[(4 * c + 1) * KS + n] = w.y();
                    kp[(4 * c + 2) * KS + n] = w.z();
                    kp[(4 * c + 3) * KS + n] = w.w();
                }
                for (int i = tid; i < (CH / 2) * (D / 8); i += WG) {   // V: keys 2p, 2p + 1, features 8c .. -> words [p][8c ..]
                    const int pr = i / (D / 8), c = i % (D / 8);
                    sycl::uint4 a(0, 0, 0, 0), b(0, 0, 0, 0);
                    if (j0 + 2 * pr < jend) a = kv_load8<KT>(vb + (j0 + 2 * pr) * RB, c);
                    if (j0 + 2 * pr + 1 < jend) b = kv_load8<KT>(vb + (j0 + 2 * pr + 1) * RB, c);
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
                    const int64_t lim = p0 + sycl::min(r0 + r, T - 1);   // the row's last key
                    const float* s0 = sp + ((hg * 2 + 0) * M + r) * CH;
                    const float* s1 = sp + ((hg * 2 + 1) * M + r) * CH;
                    float sv[CH / 4], mx = -INFINITY;
#pragma unroll
                    for (int i = 0; i < CH / 4; ++i) {
                        const int n = part * (CH / 4) + i;
                        sv[i] = j0 + n <= lim && j0 + n < jend ? s0[n] + s1[n] : -INFINITY;
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
            if constexpr (SPLIT) {
                // ---- each row's partial (or, one split, its output); rows past T, or without keys here, not written
                const float* lr = lrow + hg * M;
                const float* mr = mrow + hg * M;
                const int64_t hq = hk * G + hg;
                float* scratch = sp + sgi * (M * 16);
#pragma unroll
                for (int t = 0; t < HALF_D / 16; ++t) {
                    jm::joint_matrix_store(sg, o[t], lp(scratch), (size_t) 16, jm::layout::row_major);
                    sycl::group_barrier(sg);
                    for (int r = 0; r < T; ++r) {
                        const int64_t nsr = dec_splits(p0 + r + 1, Hkv);
                        if (spl >= nsr) continue;
                        const int d = hd * HALF_D + t * 16 + lane;
                        const float v = scratch[r * 16 + lane];
                        if (nsr == 1) {
                            out[r * Hq * D + hq * D + d] = lr[r] > 0.0f ? v / lr[r] : 0.0f;
                        } else {
                            float* pr_ = part + ((r * Hq + hq) * ns + spl) * (D + 2);
                            pr_[d] = v;
                            if (d == 0) {
                                pr_[D] = mr[r];
                                pr_[D + 1] = lr[r];
                            }
                        }
                    }
                    sycl::group_barrier(sg);
                }
                return;
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
                if (whole) {   // (prompt only)
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

template <int G, int KT>
void attn_prompt(sycl::queue& queue, const float* q, int64_t qs, const uint8_t* kc, const uint8_t* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap,
                 int64_t p0, float* out) {
    attn_flash<G, false, KT>(queue, q, qs, kc, vc, T, Hq, Hkv, cap, p0, out, 0, 1, nullptr);
}

// ---- decode rows (at most 8): the flash kernel above in its SPLIT form, the keys split over ~512 work-groups.
constexpr int KEYS_D = 64;   // a split's keys come in multiples of this

// keys a split: ~512 work-groups (key heads x splits) whatever the context, at least a work-group's step
int64_t dec_split(int64_t kv, int64_t Hkv) {
    const int64_t want = (kv * Hkv + 511) / 512;
    return sycl::max<int64_t>(KEYS_D, (want + KEYS_D - 1) / KEYS_D * KEYS_D);
}
int64_t dec_splits(int64_t kv, int64_t Hkv) {
    return (kv + dec_split(kv, Hkv) - 1) / dec_split(kv, Hkv);
}

// decode on XMX: the rows of the pass together when their splits are the same size (all but where a row's context
// crosses a split step), else a row at a time - the same arithmetic either way - then the splits merged
template <int G, int KT>
void flash_decode(sycl::queue& queue, const float* q, int64_t qs, const uint8_t* kc, const uint8_t* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap,
                  int64_t p0, float* out, float* part) {
    int64_t ns = 1;
    bool same = true;
    for (int64_t r = 0; r < T; ++r) {
        ns = std::max(ns, dec_splits(p0 + r + 1, Hkv));
        same = same && dec_split(p0 + r + 1, Hkv) == dec_split(p0 + 1, Hkv);
    }
    if (same) {
        attn_flash<G, true, KT>(queue, q, qs, kc, vc, T, Hq, Hkv, cap, p0, out, dec_split(p0 + 1, Hkv), ns, part);
    } else {
        for (int64_t r = 0; r < T; ++r)
            attn_flash<G, true, KT>(queue, q + r * qs, qs, kc, vc, 1, Hq, Hkv, cap, p0 + r, out + r * Hq * D, dec_split(p0 + r + 1, Hkv), ns,
                                part ? part + r * Hq * ns * (D + 2) : nullptr);
    }
    if (ns > 1) {   // the splits merged (their maxima in log2 units)
        queue.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * Hq) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
            const int64_t rh = it.get_group(0), d = it.get_local_id(0);
            const int64_t nsr = dec_splits(p0 + rh / Hq + 1, Hkv);
            if (nsr == 1) return;   // written by the first kernel
            const float* pp = part + rh * ns * (D + 2);
            float mx = -INFINITY;
            for (int64_t k = 0; k < nsr; ++k) mx = sycl::fmax(mx, pp[k * (D + 2) + D]);
            float Ls = 0.0f, A = 0.0f;
            if (mx != -INFINITY) {
                for (int64_t k = 0; k < nsr; ++k) {
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
    attn_prompt<6, 0>(g->q, q, qs, (const uint8_t*) kc, (const uint8_t*) vc, T, Hq, Hkv, cap, p0, out);
    return 0;
    NS_CATCH
}

int ns_q35_attn_prompt_kv(ns_gpu* g, int kt, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh,
                          int64_t cap, int64_t p0, float* out) {
    NS_TRY
    if (!ns_q35_attn_prompt_supported(T, Hq, Hkv, Dh)) return ns_fail("ns_q35_attn_prompt_kv: more than 8 rows, heads of 256, 6 query heads a key head");
    const auto* k = (const uint8_t*) kc;
    const auto* v = (const uint8_t*) vc;
    if (kt == 0) attn_prompt<6, 0>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out);
    else if (kt == 1) attn_prompt<6, 1>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out);
    else if (kt == 2) attn_prompt<6, 2>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out);
    else return ns_fail("ns_q35_attn_prompt_kv: a cache form 0 (half), 1 (q8_0) or 2 (q4_0)");
    return 0;
    NS_CATCH
}

int ns_q35_attn_decode_supported(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh) {
    return T >= 1 && T <= 8 && Dh == D && Hkv > 0 && Hq == 6 * Hkv;
}

int64_t ns_q35_attn_decode_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh, int64_t p0) {
    int64_t ns = 1;
    for (int64_t r = 0; r < T; ++r) ns = std::max(ns, dec_splits(p0 + r + 1, Hkv));
    return ns > 1 ? T * Hq * ns * (Dh + 2) : 0;
}

int ns_q35_attn_decode(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh,
                       int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_TRY
    if (!ns_q35_attn_decode_supported(T, Hq, Hkv, Dh)) return ns_fail("ns_q35_attn_decode: 1..8 rows, heads of 256, 6 query heads a key head");
    if (ns_q35_attn_decode_scratch(T, Hq, Hkv, Dh, p0) > 0 && !scratch) return ns_fail("ns_q35_attn_decode: no scratch");
    flash_decode<6, 0>(g->q, q, qs, (const uint8_t*) kc, (const uint8_t*) vc, T, Hq, Hkv, cap, p0, out, scratch);
    return 0;
    NS_CATCH
}

int ns_q35_attn_decode_kv(ns_gpu* g, int kt, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t Dh,
                          int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_TRY
    if (!ns_q35_attn_decode_supported(T, Hq, Hkv, Dh)) return ns_fail("ns_q35_attn_decode_kv: 1..8 rows, heads of 256, 6 query heads a key head");
    if (ns_q35_attn_decode_scratch(T, Hq, Hkv, Dh, p0) > 0 && !scratch) return ns_fail("ns_q35_attn_decode_kv: no scratch");
    const auto* k = (const uint8_t*) kc;
    const auto* v = (const uint8_t*) vc;
    if (kt == 0) flash_decode<6, 0>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else if (kt == 1) flash_decode<6, 1>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else if (kt == 2) flash_decode<6, 2>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else return ns_fail("ns_q35_attn_decode_kv: a cache form 0 (half), 1 (q8_0) or 2 (q4_0)");
    return 0;
    NS_CATCH
}

}  // extern "C"
