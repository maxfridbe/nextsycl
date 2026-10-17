// q3s.cpp: Qwen3-TTS's silo (q3s.h) - its frame's kernels, fused where the shared ones (the MiniMax Music engine's mm3
// kernels and the diffusion kernels' norms) take a launch each, and the draws on the device. Started from mm3.cpp's
// gemv and attention (copied, then changed: rule 6). Sub-groups of 16 (Xe2's native width).
#include "ns.h"
#include "ns_internal.hpp"
#include "q3s.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>

namespace {
using half = sycl::half;
namespace X = sycl::ext::oneapi::experimental;
constexpr int SG = 16;
constexpr int HD = 128;   // a head's width

// ---- the decode's product. A work-group of 8 sub-groups; each sub-group R = 2 output rows over a share of K: KS
// sub-groups split a row pair's K (interleaved chunks of SG * V), their sums merged through local memory. Its lanes
// read V weights at a time (16 bytes: 8 halfs or 16 int8), U chunks a step (loads in flight). NORM: each x row's
// 1 / rms reduced over the work-group first, the row read normed (x * inv * weight); SWIGLU: the row pair is (gate n,
// up n) and the output silu(gate) * up.
constexpr int NSG = 8, R = 2, U = 2;
constexpr int64_t KMAX_NORM = 2048;   // the widest normed input (the talker's hidden)

template <typename WT, int MM>
void gemv(sycl::queue& q, const float* x, int M, int64_t K, const WT* w, const float* scale, int64_t N, const float* bias, float* out,
          int64_t ldo, int flags, const float* norm_w, float eps, int KS) {
    constexpr int V = sizeof(WT) == 1 ? 16 : 8;
    const bool norm = flags & NS_Q3S_NORM, acc = flags & NS_Q3S_ACC, swiglu = flags & NS_Q3S_SWIGLU;
    const int64_t pairs = swiglu ? N / 2 : (N + R - 1) / R;   // row pairs to make
    const int per_wg = NSG / KS;                              // row pairs a work-group
    const int64_t groups = (pairs + per_wg - 1) / per_wg;
    const int64_t Nh = N / 2;
    const int64_t step = (int64_t) KS * SG * V;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) NSG * MM * R), h);
        sycl::local_accessor<float, 1> inv(sycl::range<1>(MM), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SG * NSG), sycl::range<1>(SG * NSG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int ks = sgi % KS, rp = sgi / KS;
            const int64_t pair = (int64_t) it.get_group(0) * per_wg + rp;
            float iv[MM];
#pragma unroll
            for (int m = 0; m < MM; ++m) iv[m] = 1.0f;
            if (norm) {
                // each row's 1 / rms over the work-group (4 floats a work-item at a time)
                const int li = (int) it.get_local_id(0);
                for (int m = 0; m < M; ++m) {
                    float ss = 0.0f;
                    for (int64_t k = (int64_t) li * 4; k < K; k += SG * NSG * 4) {
                        const sycl::vec<float, 4> v = *reinterpret_cast<const sycl::vec<float, 4>*>(x + m * K + k);
                        ss += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
                    }
                    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                    if (li == 0) inv[m] = sycl::rsqrt(ss / (float) K + eps);
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int m = 0; m < MM; ++m) iv[m] = m < M ? inv[m] : 1.0f;
            }
            int64_t rows[R];
            if (swiglu) {
                rows[0] = pair < Nh ? pair : Nh - 1;
                rows[1] = rows[0] + Nh;
            } else {
#pragma unroll
                for (int r = 0; r < R; ++r) rows[r] = pair * R + r < N ? pair * R + r : N - 1;
            }
            float a[MM][R];
#pragma unroll
            for (int m = 0; m < MM; ++m)
#pragma unroll
                for (int r = 0; r < R; ++r) a[m][r] = 0.0f;
            for (int64_t k0 = ((int64_t) ks * SG + lane) * V; k0 < K; k0 += U * step) {
                float wf[U][R][V];
                bool in[U];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int64_t k = k0 + u * step;
                    in[u] = k < K;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        sycl::vec<WT, V> wv(0);
                        if (in[u]) wv = *reinterpret_cast<const sycl::vec<WT, V>*>(w + rows[r] * K + k);
#pragma unroll
                        for (int i = 0; i < V; ++i) wf[u][r][i] = (float) wv[i];
                    }
                }
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    if (!in[u]) continue;
                    const int64_t k = k0 + u * step;
                    float nw[V];
                    if (norm) {
#pragma unroll
                        for (int i = 0; i < V; i += 4) {
                            const sycl::vec<float, 4> n4 = *reinterpret_cast<const sycl::vec<float, 4>*>(norm_w + k + i);
                            nw[i] = n4[0];
                            nw[i + 1] = n4[1];
                            nw[i + 2] = n4[2];
                            nw[i + 3] = n4[3];
                        }
                    }
#pragma unroll
                    for (int m = 0; m < MM; ++m) {
                        if (m < M) {
                            float xv[V];
#pragma unroll
                            for (int i = 0; i < V; i += 4) {
                                const sycl::vec<float, 4> x4 = *reinterpret_cast<const sycl::vec<float, 4>*>(x + m * K + k + i);
                                xv[i] = x4[0];
                                xv[i + 1] = x4[1];
                                xv[i + 2] = x4[2];
                                xv[i + 3] = x4[3];
                            }
                            if (norm) {
#pragma unroll
                                for (int i = 0; i < V; ++i) xv[i] = nw[i] * (xv[i] * iv[m]);
                            }
#pragma unroll
                            for (int r = 0; r < R; ++r) {
                                float s = 0.0f;
#pragma unroll
                                for (int i = 0; i < V; ++i) s += wf[u][r][i] * xv[i];
                                a[m][r] += s;
                            }
                        }
                    }
                }
            }
            // the lanes' sums, then the K shares' through local memory
#pragma unroll
            for (int m = 0; m < MM; ++m)
#pragma unroll
                for (int r = 0; r < R; ++r) a[m][r] = sycl::reduce_over_group(sg, a[m][r], sycl::plus<float>());
            if (KS > 1) {
                if (lane == 0) {
#pragma unroll
                    for (int m = 0; m < MM; ++m)
#pragma unroll
                        for (int r = 0; r < R; ++r) red[(sgi * MM + m) * R + r] = a[m][r];
                }
                sycl::group_barrier(it.get_group());
                if (ks == 0 && lane == 0) {
#pragma unroll
                    for (int m = 0; m < MM; ++m)
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            float s = 0.0f;
                            for (int j = 0; j < KS; ++j) s += red[((sgi + j) * MM + m) * R + r];
                            a[m][r] = s;
                        }
                }
            }
            if (ks != 0 || lane != 0) return;
            if (swiglu) {
                if (pair >= Nh) return;
#pragma unroll
                for (int m = 0; m < MM; ++m) {
                    if (m < M) {
                        float gt = a[m][0], up = a[m][1];
                        if (scale) { gt *= scale[rows[0]]; up *= scale[rows[1]]; }
                        if (bias) { gt += bias[rows[0]]; up += bias[rows[1]]; }
                        float* o = out + m * ldo + pair;
                        const float v = gt / (1.0f + sycl::exp(-gt)) * up;
                        *o = acc ? *o + v : v;
                    }
                }
                return;
            }
#pragma unroll
            for (int m = 0; m < MM; ++m) {
                if (m < M) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const int64_t n = pair * R + r;
                        if (n < N) {
                            const float v = a[m][r] * (scale ? scale[n] : 1.0f) + (bias ? bias[n] : 0.0f);
                            float* o = out + m * ldo + n;
                            *o = acc ? *o + v : v;
                        }
                    }
                }
            }
        });
    });
}

// K shares a row pair: enough sub-groups in flight for a narrow product (about 2,048 of them), each share at least
// one chunk of SG * V
int k_shares(int64_t pairs, int64_t K, int V) {
    int ks = 1;
    while (ks < NSG && pairs * ks < 2048 && K / ((int64_t) (ks * 2) * SG * V) >= 1) ks *= 2;
    return ks;
}

// q / k of a row: 128 values a head over a sub-group (8 a lane, strided by 16), RMS-normed and rotated at `pos`
inline void norm_rope(float v[8], const float* weight, float eps, const float* inv_freq, float pos, int l, sycl::sub_group sg) {
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) ss += v[j] * v[j];
    const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / 128.0f + eps);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = weight[l + 16 * j] * (v[j] * inv);
    float o[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = l + 16 * j;
        const float a = pos * inv_freq[i], c = sycl::cos(a), s = sycl::sin(a);
        o[j] = v[j] * c - v[j + 4] * s;
        o[j + 4] = v[j + 4] * c + v[j] * s;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = o[j];
}

// splitmix64 on two 32-bit words (the device's integers), a uniform float in [0, 1) from its top 24 bits
inline float uniform(uint32_t st[2]) {
    uint64_t s = ((uint64_t) st[1] << 32) | st[0];
    s += 0x9e3779b97f4a7c15ull;
    st[0] = (uint32_t) s;
    st[1] = (uint32_t) (s >> 32);
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (float) (z >> 40) * (1.0f / 16777216.0f);
}

// a float's order as an unsigned key (larger float, larger key)
inline uint32_t key(float v) {
    const uint32_t b = sycl::bit_cast<uint32_t>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// the draw's work-group: up to 4,096 logits, 16 a lane (each lane a run of 16 in index order)
constexpr int DRAW_WG = 256, DRAW_PER = 16;
// the attention kernels' work-group: 8 sub-groups
constexpr int ATT_SG = 8, ATT_WG = ATT_SG * SG;

// a head's 128 values from a row (8 a lane, strided by 16) RMS-normed and rotated at `pos`, into local memory at
// their places (scaled)
inline void head_to_local(const float* src, const float* weight, float eps, const float* inv_freq, float pos, float scale, float* dst, int l,
                          sycl::sub_group sg) {
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = src[l + 16 * j];
    norm_rope(v, weight, eps, inv_freq, pos, l, sg);
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[l + 16 * j] = v[j] * scale;
}

// q . k over a sub-group: q in local memory, k a half row (8 contiguous a lane)
inline float dot_q(const float* q, const half* k, int l, sycl::sub_group sg) {
    const sycl::vec<half, 8> kv = *reinterpret_cast<const sycl::vec<half, 8>*>(k + l * 8);
    float d = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) d += q[l * 8 + i] * (float) kv[i];
    return sycl::reduce_over_group(sg, d, sycl::plus<float>());
}
inline float dot_ql(const float* q, const float* k, int l, sycl::sub_group sg) {
    float d = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) d += q[l * 8 + i] * (float) (half) k[l * 8 + i];
    return sycl::reduce_over_group(sg, d, sycl::plus<float>());
}
}  // namespace

extern "C" {

int ns_q3s_supported(int64_t head, int64_t group, int64_t k) {
    return head == HD && (group == 1 || group == 2 || group == 4 || group == 8) && k % 256 == 0;
}

int ns_q3s_gemv(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* w, int wt, const float* scale, int64_t N, const float* bias,
                float* out, int64_t ldo, int flags, const float* norm_w, float eps) {
    NS_TRY
    if (M <= 0 || N <= 0) return 0;
    if (M > 8 || K % 256 != 0) return ns_fail("ns_q3s_gemv: at most 8 rows, K a multiple of 256");
    if ((flags & NS_Q3S_NORM) && (K > KMAX_NORM || M > 2 || !norm_w)) return ns_fail("ns_q3s_gemv: a normed input of at most 2 rows of 2,048");
    if ((flags & NS_Q3S_SWIGLU) && N % 2) return ns_fail("ns_q3s_gemv: SwiGLU takes an even number of rows");
    const int V = wt == 1 ? 16 : 8;
    const int64_t pairs = (flags & NS_Q3S_SWIGLU) ? N / 2 : (N + R - 1) / R;
    const int ks = k_shares(pairs, K, V);
#define Q3S_GEMV(WT, MM) gemv<WT, MM>(g->q, x, (int) M, K, (const WT*) w, scale, N, bias, out, ldo, flags, norm_w, eps, ks)
    if (wt == 0) {
        if (M == 1) Q3S_GEMV(half, 1);
        else if (M == 2) Q3S_GEMV(half, 2);
        else Q3S_GEMV(half, 8);
    } else if (wt == 1) {
        if (M == 1) Q3S_GEMV(int8_t, 1);
        else if (M == 2) Q3S_GEMV(int8_t, 2);
        else Q3S_GEMV(int8_t, 8);
    } else {
        return ns_fail("ns_q3s_gemv: weights half (0) or int8 (1)");
    }
#undef Q3S_GEMV
    return 0;
    NS_CATCH
}

int64_t ns_q3s_attn_scratch(int64_t Hq, int64_t splits) {
    return splits > 1 ? Hq * splits * (HD + 2) : 0;
}

int ns_q3s_attn(ns_gpu* g, float* qkv, int64_t H, int64_t Hkv, const float* q_norm, const float* k_norm, float eps, const float* inv_freq,
                const int32_t* pos, void* kc, void* vc, int64_t T, int64_t splits, float* out, float* part, int32_t* counters) {
    NS_TRY
    if (splits < 1 || (splits > 1 && (!part || !counters))) return ns_fail("ns_q3s_attn: splits and their scratch");
    if (H % Hkv || H / Hkv > 4) return ns_fail("ns_q3s_attn: at most 4 query heads a key head");
    half* kco = (half*) kc;
    half* vco = (half*) vc;
    const int64_t G = H / Hkv, ns = splits;
    const float scale = 1.0f / sycl::sqrt((float) HD);
    // a work-group per (key head, split): the G query heads (and, in the split holding the new position, the new key
    // and value, stored) normed and rotated into local memory; then the split's keys a tile of TK at a time - the
    // tile's keys and values read into local memory by the whole work-group at once, its scores a sub-group a key,
    // a running softmax a work-item a feature. A split's partial (sum, max, total) merged with the others' by the key
    // head's last work-group to finish (a counter a key head, reset by it)
    constexpr int TK = 32;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) (G + 2) * HD), h);
        sycl::local_accessor<half, 1> kt(sycl::range<1>((size_t) TK * HD), h);
        sycl::local_accessor<half, 1> vt(sycl::range<1>((size_t) TK * HD), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>((size_t) (G * TK)), h);
        sycl::local_accessor<int, 1> last(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (Hkv * ns) * ATT_WG), sycl::range<1>(ATT_WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            auto grp = it.get_group();
            const int l = (int) sg.get_local_linear_id(), si = (int) sg.get_group_linear_id(), li = (int) it.get_local_id(0);
            const int64_t sp = it.get_group(0) % ns, hk = it.get_group(0) / ns;
            const int64_t p = *pos;
            const int64_t kv_end = p + 1, per = (kv_end + ns - 1) / ns;
            const int64_t j0 = sp * per, j1 = sycl::min(kv_end, (sp + 1) * per);
            const bool mine = j0 <= p && p < j1;   // this split holds the new position
            float* kn = &qs[G * HD];
            float* vn = &qs[(G + 1) * HD];
            if (si < G) {
                head_to_local(qkv + (hk * G + si) * HD, q_norm, eps, inv_freq, (float) p, scale, &qs[si * HD], l, sg);
            } else if (si == G && mine) {
                head_to_local(qkv + (H + hk) * HD, k_norm, eps, inv_freq, (float) p, 1.0f, kn, l, sg);
                half* kd = kco + (hk * T + p) * HD;
                half* vd = vco + (hk * T + p) * HD;
                const float* vr = qkv + (H + Hkv + hk) * HD;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int i = l + 16 * j;
                    kd[i] = (half) kn[i];
                    vd[i] = (half) vr[i];
                    vn[i] = vr[i];
                }
            }
            const half* kb = kco + hk * T * HD;
            const half* vb = vco + hk * T * HD;
            float m[4], L[4], acc[4];
#pragma unroll
            for (int gq = 0; gq < 4; ++gq) {
                m[gq] = -INFINITY;
                L[gq] = 0.0f;
                acc[gq] = 0.0f;
            }
            const int d = li;   // this work-item's feature (ATT_WG == HD)
            for (int64_t t0 = j0; t0 < j1; t0 += TK) {
                const int tn = (int) sycl::min((int64_t) TK, j1 - t0);
                sycl::group_barrier(grp);
                // the tile's keys and values: 16 pieces of 8 a row, every work-item a few, all at once (the new
                // position's from local memory, as the cache holds them: half)
                for (int idx = li; idx < tn * 16; idx += ATT_WG) {
                    const int j = idx / 16, c = (idx % 16) * 8;
                    sycl::vec<half, 8> kv, vv;
                    if (t0 + j == p) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            kv[i] = (half) kn[c + i];
                            vv[i] = (half) vn[c + i];
                        }
                    } else {
                        kv = *reinterpret_cast<const sycl::vec<half, 8>*>(kb + (t0 + j) * HD + c);
                        vv = *reinterpret_cast<const sycl::vec<half, 8>*>(vb + (t0 + j) * HD + c);
                    }
                    *reinterpret_cast<sycl::vec<half, 8>*>(&kt[j * HD + c]) = kv;
                    *reinterpret_cast<sycl::vec<half, 8>*>(&vt[j * HD + c]) = vv;
                }
                sycl::group_barrier(grp);
                // the scores: a sub-group a key
                for (int j = si; j < tn; j += ATT_SG) {
                    for (int gq = 0; gq < G; ++gq) {
                        float s = 0.0f;
#pragma unroll
                        for (int i = 0; i < 8; ++i) s += qs[gq * HD + l * 8 + i] * (float) kt[j * HD + l * 8 + i];
                        s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                        if (l == 0) sc[gq * TK + j] = s;
                    }
                }
                sycl::group_barrier(grp);
                // the running softmax, a work-item a feature of every head
                for (int gq = 0; gq < G; ++gq) {
                    float mt = m[gq];
                    for (int j = 0; j < tn; ++j) mt = sycl::fmax(mt, sc[gq * TK + j]);
                    const float c = sycl::exp(m[gq] - mt);
                    float lt = 0.0f, at = 0.0f;
                    for (int j = 0; j < tn; ++j) {
                        const float e = sycl::exp(sc[gq * TK + j] - mt);
                        lt += e;
                        at += e * (float) vt[j * HD + d];
                    }
                    L[gq] = L[gq] * c + lt;
                    acc[gq] = acc[gq] * c + at;
                    m[gq] = mt;
                }
            }
            for (int gq = 0; gq < G; ++gq) {
                const int64_t hq = hk * G + gq;
                if (ns == 1) {
                    out[hq * HD + d] = L[gq] > 0.0f ? acc[gq] / L[gq] : 0.0f;
                } else {
                    float* pp = part + (hq * ns + sp) * (HD + 2);
                    pp[d] = acc[gq];
                    if (d == 0) {
                        pp[HD] = m[gq];
                        pp[HD + 1] = L[gq];
                    }
                }
            }
            if (ns == 1) return;
            // the last of the key head's splits to finish merges them
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
            sycl::group_barrier(grp);
            if (li == 0) {
                sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space> cnt(counters[hk]);
                last[0] = cnt.fetch_add(1) == ns - 1;
            }
            sycl::group_barrier(grp);
            if (!last[0]) return;
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
            for (int gq = 0; gq < G; ++gq) {
                const int64_t hq = hk * G + gq;
                const float* pp = part + hq * ns * (HD + 2);
                float mx = -INFINITY;
                for (int64_t k = 0; k < ns; ++k) mx = sycl::fmax(mx, pp[k * (HD + 2) + HD]);
                float Lt = 0.0f, A = 0.0f;
                if (mx != -INFINITY) {
                    for (int64_t k = 0; k < ns; ++k) {
                        const float e = sycl::exp(pp[k * (HD + 2) + HD] - mx);
                        Lt += pp[k * (HD + 2) + HD + 1] * e;
                        A += pp[k * (HD + 2) + d] * e;
                    }
                }
                out[hq * HD + d] = Lt > 0.0f ? A / Lt : 0.0f;
            }
            if (li == 0) counters[hk] = 0;
        });
    });
    return 0;
    NS_CATCH
}

int ns_q3s_attn_small(ns_gpu* g, float* qkv, int64_t ws, int64_t S, int64_t p0, int64_t H, int64_t Hkv, const float* q_norm, const float* k_norm,
                      float eps, const float* inv_freq, void* kc, void* vc, int64_t T, float* out) {
    NS_TRY
    if (S < 1 || S > 2 || p0 + S > T || H % Hkv || T > 32) return ns_fail("ns_q3s_attn_small: 1 or 2 rows within a cache of at most 32");
    const int64_t G = H / Hkv;
    if (G > 3) return ns_fail("ns_q3s_attn_small: at most 3 query heads a key head");
    half* ko = (half*) kc;
    half* vo = (half*) vc;
    const float scale = 1.0f / sycl::sqrt((float) HD);
    // a work-group a key head: the cache's older keys and values read into local memory by the whole work-group at
    // once, while its S new ones are normed, rotated, stored and put beside them, and the S x G query heads; then
    // every (row, head, key) score, a sub-group each; then a work-item a feature its rows' and heads' softmax and values
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>((size_t) (S * G) * HD), h);
        sycl::local_accessor<float, 1> kt(sycl::range<1>((size_t) T * HD), h);
        sycl::local_accessor<float, 1> vt(sycl::range<1>((size_t) T * HD), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>((size_t) (S * G * T)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) Hkv * ATT_WG), sycl::range<1>(ATT_WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            auto grp = it.get_group();
            const int l = (int) sg.get_local_linear_id(), si = (int) sg.get_group_linear_id(), li = (int) it.get_local_id(0);
            const int64_t hk = it.get_group(0);
            const half* kb = ko + hk * T * HD;
            const half* vb = vo + hk * T * HD;
            // the older keys and values (as the cache holds them: half)
            for (int idx = li; idx < (int) p0 * 16; idx += ATT_WG) {
                const int j = idx / 16, c = (idx % 16) * 8;
                const sycl::vec<half, 8> kv = *reinterpret_cast<const sycl::vec<half, 8>*>(kb + j * HD + c);
                const sycl::vec<half, 8> vv = *reinterpret_cast<const sycl::vec<half, 8>*>(vb + j * HD + c);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    kt[j * HD + c + i] = (float) kv[i];
                    vt[j * HD + c + i] = (float) vv[i];
                }
            }
            if (si < S) {
                const int64_t pp = p0 + si;
                float* kd = &kt[pp * HD];
                head_to_local(qkv + si * ws + (H + hk) * HD, k_norm, eps, inv_freq, (float) pp, 1.0f, kd, l, sg);
                const float* vr = qkv + si * ws + (H + Hkv + hk) * HD;
                half* kg = ko + (hk * T + pp) * HD;
                half* vg = vo + (hk * T + pp) * HD;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int i = l + 16 * j;
                    const half kh = (half) kd[i], vh = (half) vr[i];
                    kg[i] = kh;
                    vg[i] = vh;
                    kd[i] = (float) kh;
                    vt[pp * HD + i] = (float) vh;
                }
            } else if (si < S + S * G) {
                const int q = si - (int) S, s = q / (int) G, gq = q % (int) G;
                head_to_local(qkv + s * ws + (hk * G + gq) * HD, q_norm, eps, inv_freq, (float) (p0 + s), scale, &qs[q * HD], l, sg);
            }
            sycl::group_barrier(grp);
            // every (row, head, key) score: keys 0..p0+row
            const int64_t nk = p0 + S;
            for (int64_t t = si; t < S * G * nk; t += ATT_SG) {
                const int64_t q = t / nk, j = t % nk, s = q / G;
                if (j > p0 + s) continue;
                float v = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) v += qs[q * HD + l * 8 + i] * kt[j * HD + l * 8 + i];
                v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
                if (l == 0) sc[q * T + j] = v;
            }
            sycl::group_barrier(grp);
            // a work-item a feature: each (row, head)'s softmax over its keys and the values
            const int d = li;
            for (int64_t q = 0; q < S * G; ++q) {
                const int64_t s = q / G, n = p0 + s + 1;
                float mx = -INFINITY;
                for (int64_t j = 0; j < n; ++j) mx = sycl::fmax(mx, sc[q * T + j]);
                float L = 0.0f, a = 0.0f;
                for (int64_t j = 0; j < n; ++j) {
                    const float e = sycl::exp(sc[q * T + j] - mx);
                    L += e;
                    a += e * vt[j * HD + d];
                }
                out[s * H * HD + (hk * G + q % G) * HD + d] = a / L;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_q3s_draw(ns_gpu* g, const float* logits, int64_t n, int gi, const ns_q3s_draws* d, ns_q3s_state* st, uint8_t* seen,
                const int32_t* force, const float* ptab, int64_t base, int64_t width, float* row) {
    NS_TRY
    if (n < 1 || n > (int64_t) DRAW_WG * DRAW_PER) return ns_fail("ns_q3s_draw: 1 to 4,096 logits");
    if (gi < 0 || gi > 15) return ns_fail("ns_q3s_draw: codebook 0 to 15");
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(DRAW_WG / SG * 256), h);   // a histogram a sub-group
        sycl::local_accessor<uint32_t, 1> sh(sycl::range<1>(4), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(DRAW_WG), sycl::range<1>(DRAW_WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto grp = it.get_group();
            const int li = (int) it.get_local_id(0), sgi = (int) it.get_sub_group().get_group_linear_id();
            const int who = gi == 0 ? 0 : 1;
            const int frame = st->frame;
            // the logits as the processors leave them (penalty, suppressed ids, eos held back: the talker's); the best
            float v[DRAW_PER];
            int best = INT32_MAX;
            float bv = -INFINITY;
#pragma unroll
            for (int j = 0; j < DRAW_PER; ++j) {
                const int i = li * DRAW_PER + j;
                float x = -INFINITY;
                if (i < n) {
                    x = logits[i];
                    if (gi == 0) {
                        if (d->penalty != 1.0f && seen[i]) x = x > 0.0f ? x / d->penalty : x * d->penalty;
                        if (i >= d->suppress_from && i != d->eos) x = -INFINITY;
                        if (i == d->eos && frame < d->min_frames) x = -INFINITY;
                    }
                }
                v[j] = x;
                if (x > bv) {
                    bv = x;
                    best = i;
                }
            }
            // the best: the largest, the first of equals
            const float gbv = sycl::reduce_over_group(grp, bv, sycl::maximum<float>());
            int code = sycl::reduce_over_group(grp, bv == gbv ? best : INT32_MAX, sycl::minimum<int>());
            if (code == INT32_MAX) code = 0;
            if (d->sample[who] && gbv != -INFINITY) {
                const float t = sycl::fmax(d->temperature[who], 1e-5f);
                int nfin = 0;
#pragma unroll
                for (int j = 0; j < DRAW_PER; ++j) {
                    v[j] = v[j] / t;
                    nfin += sycl::isfinite(v[j]) ? 1 : 0;
                }
                const int k = d->top_k[who];
                const int fin = sycl::reduce_over_group(grp, nfin, sycl::plus<int>());
                float thr = -INFINITY;
                if (k > 0 && fin > k) {
                    // the k-th largest finite value: a radix select over the keys, 8 bits a pass from the top; a
                    // work-item a bin, the bins counted from the top by a scan
                    uint32_t prefix = 0, mask = 0;
                    int want = k;
                    for (int pass = 3; pass >= 0; --pass) {
                        const int sh_ = pass * 8;
                        for (int j = li; j < DRAW_WG / SG * 256; j += DRAW_WG) hist[j] = 0;
                        sycl::group_barrier(grp);
#pragma unroll
                        for (int j = 0; j < DRAW_PER; ++j) {
                            if (sycl::isfinite(v[j])) {
                                const uint32_t kk = key(v[j]);
                                if ((kk & mask) == prefix) {
                                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                                     sycl::access::address_space::local_space>(hist[sgi * 256 + ((kk >> sh_) & 255)]).fetch_add(1);
                                }
                            }
                        }
                        sycl::group_barrier(grp);
                        // work-item li the bin 255 - li: the counts above it, its own
                        const int b = 255 - li;
                        int c = 0;
                        for (int q = 0; q < DRAW_WG / SG; ++q) c += (int) hist[q * 256 + b];
                        const int above = sycl::exclusive_scan_over_group(grp, c, sycl::plus<int>());
                        if (above < want && above + c >= want) {
                            sh[0] = (uint32_t) b;
                            sh[1] = (uint32_t) (want - above);
                        }
                        sycl::group_barrier(grp);
                        prefix |= sh[0] << sh_;
                        mask |= 255u << sh_;
                        want = (int) sh[1];
                        sycl::group_barrier(grp);
                    }
                    // the key back to its float
                    const uint32_t b = (prefix & 0x80000000u) ? (prefix & 0x7fffffffu) : ~prefix;
                    thr = sycl::bit_cast<float>(b);
                }
                const float mx = gbv / t;
                float w[DRAW_PER], ls = 0.0f;
#pragma unroll
                for (int j = 0; j < DRAW_PER; ++j) {
                    w[j] = (sycl::isfinite(v[j]) && v[j] >= thr) ? sycl::exp(v[j] - mx) : 0.0f;
                    ls += w[j];
                }
                const float total = sycl::reduce_over_group(grp, ls, sycl::plus<float>());
                const float before = sycl::exclusive_scan_over_group(grp, ls, sycl::plus<float>());
                // the generator's draw (every work-item steps a copy; work-item 0 keeps it)
                uint32_t r[2] = {st->rng[0], st->rng[1]};
                const float u = uniform(r) * total;
                // the work-item whose share holds u walks its own (the last weighted code if rounding runs past the end)
                int pick = INT32_MAX;
                if (ls > 0.0f && u >= before && u < before + ls) {
                    float c = before;
#pragma unroll
                    for (int j = 0; j < DRAW_PER; ++j) {
                        if (pick == INT32_MAX && w[j] > 0.0f) {
                            c += w[j];
                            if (u < c) pick = li * DRAW_PER + j;
                        }
                    }
                    if (pick == INT32_MAX) {
#pragma unroll
                        for (int j = DRAW_PER - 1; j >= 0; --j)
                            if (pick == INT32_MAX && w[j] > 0.0f) pick = li * DRAW_PER + j;
                    }
                }
                int lastw = -1;
#pragma unroll
                for (int j = 0; j < DRAW_PER; ++j)
                    if (w[j] > 0.0f) lastw = li * DRAW_PER + j;
                const int got = sycl::reduce_over_group(grp, pick, sycl::minimum<int>());
                const int lw = sycl::reduce_over_group(grp, lastw, sycl::maximum<int>());
                code = got != INT32_MAX ? got : (lw >= 0 ? lw : code);
                if (li == 0) {
                    st->rng[0] = r[0];
                    st->rng[1] = r[1];
                }
            }
            if (d->forced) {
                const int f = force[frame * 16 + gi];
                if (li == 0 && code != f && !(gi == 0 && f == d->eos)) st->misses[who] += 1;
                code = f;
            }
            if (li == 0) {
                st->codes[gi] = code;
                if (gi == 0 && code >= 0 && code < n) seen[code] = 1;
            }
            if (row) {
                const float* src = ptab + (base + (int64_t) code) * width;
                for (int64_t j = li * 4; j < width; j += DRAW_WG * 4)
                    *reinterpret_cast<sycl::vec<float, 4>*>(row + j) = *reinterpret_cast<const sycl::vec<float, 4>*>(src + j);
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_q3s_frame_in(ns_gpu* g, const void* table, int64_t D, int64_t v0, int64_t vg, const ns_q3s_state* st, const ns_q3s_draws* d,
                    const float* feed, float* out) {
    NS_TRY
    const half* t = (const half*) table;
    g->q.parallel_for(sycl::range<1>((size_t) D), [=](sycl::id<1> id) {
        const int64_t j = id[0];
        // the codes' rows summed as the shared path does (8 at a time, then the two sums), then the text row
        float s0 = 0.0f, s1 = 0.0f;
        for (int c = 0; c < 16; ++c) {
            const int64_t r = c == 0 ? st->codes[0] : v0 + (int64_t) (c - 1) * vg + st->codes[c];
            const float e = (float) t[r * D + j];
            if (c < 8) s0 += e;
            else s1 += e;
        }
        const int64_t f = st->frame < d->n_text ? st->frame : d->n_text;
        out[j] = (s0 + s1) + feed[f * D + j];
    });
    return 0;
    NS_CATCH
}

int ns_q3s_advance(ns_gpu* g, ns_q3s_state* st) {
    NS_TRY
    g->q.single_task([=] {
        st->frame += 1;
        st->pos += 1;
    });
    return 0;
    NS_CATCH
}

using Graph = X::command_graph<X::graph_state::modifiable>;
using Exec = X::command_graph<X::graph_state::executable>;

int ns_q3s_graph_begin(ns_gpu* g, void** graph) {
    NS_TRY
    auto* gr = new Graph(g->q.get_context(), g->q.get_device());
    gr->begin_recording(g->q);
    *graph = gr;
    return 0;
    NS_CATCH
}

int ns_q3s_graph_end(ns_gpu* g, void* graph, void** exec) {
    NS_TRY
    auto* gr = (Graph*) graph;
    gr->end_recording(g->q);
    *exec = new Exec(gr->finalize());
    delete gr;
    return 0;
    NS_CATCH
}

int ns_q3s_graph_launch(ns_gpu* g, void* exec) {
    NS_TRY
    g->q.ext_oneapi_graph(*(Exec*) exec);
    return 0;
    NS_CATCH
}

void ns_q3s_graph_free(void* exec) {
    delete (Exec*) exec;
}

int ns_q3s_read_async(ns_gpu* g, void* host, const void* src, size_t bytes) {
    NS_TRY
    g->q.memcpy(host, src, bytes);
    return 0;
    NS_CATCH
}

}  // extern "C"
