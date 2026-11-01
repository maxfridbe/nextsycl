// yue.cpp: the YuE2 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the decode's product
// read from the Q8_0 blocks themselves (nsd_linear_q makes the whole matrix half first: right for the thousands of rows
// of a prefill or a flow pass, five times the traffic for the decode's one or two), the embedding rows from the Q8_0
// table, the attention's inputs (q / k norm, RoPE, the cache's rows) in one pass, the decode's attention against the
// half cache, and the flow stage's few element-wise steps. After nextcuda's yue.cu.
//
// The rotary (cos, sin) table is made on the host (lm.rs: upstream's float32 angles, position * inv_freq, one row a
// position of the model's context), not on the card: kernels/build.sh builds audio/ with the compiler's fast math, and
// a device cosine of an angle in the tens of thousands need not agree with the reference's.
#include "ns.h"
#include "ns_internal.hpp"
#include "yue.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>

namespace {

using half = sycl::half;
constexpr int SG = 16;
constexpr int HD = 128;        // the head size
constexpr int HQ = 16, HKV = 8;
constexpr int QB = 34;         // bytes of a Q8_0 block: a half scale, then 32 int8

inline float h2f(uint16_t b) { return (float) sycl::bit_cast<half>(b); }

// ---- the Q8_0 product: a sub-group a pair of output rows, each lane a block of 32 at a time across K. A block's bytes
// are 2-aligned (34 a block): 17 two-byte reads; the silo's repacked matrices read 16 bytes at a time instead
constexpr int Q_NSG = 8, Q_R = 2, Q_MM = 8;

inline void q8_block(const uint8_t* p, float* o) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    const float d = h2f(h[0]);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint16_t u = h[1 + i];
        o[2 * i] = d * (float) (int8_t) (u & 0xff);
        o[2 * i + 1] = d * (float) (int8_t) (u >> 8);
    }
}

void gemv_q8(sycl::queue& q, const float* x, int M, int64_t K, const uint8_t* w, int64_t N, float* out, int64_t ldo, bool acc) {
    const int64_t groups = (N + Q_NSG * Q_R - 1) / (Q_NSG * Q_R);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * Q_NSG * SG), sycl::range<1>(Q_NSG * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
        const int64_t n0 = (it.get_group(0) * Q_NSG + sgi) * Q_R;
        const int64_t nb = K / 32, row_bytes = nb * QB;
        float a[Q_MM][Q_R];
#pragma unroll
        for (int m = 0; m < Q_MM; ++m)
#pragma unroll
            for (int r = 0; r < Q_R; ++r) a[m][r] = 0.0f;
        for (int64_t b = lane; b < nb; b += SG) {
            float wf[Q_R][32];
#pragma unroll
            for (int r = 0; r < Q_R; ++r) {
                const int64_t n = n0 + r < N ? n0 + r : N - 1;
                q8_block(w + n * row_bytes + b * QB, wf[r]);
            }
#pragma unroll
            for (int m = 0; m < Q_MM; ++m) {
                if (m < M) {
                    const sycl::float4* xp = reinterpret_cast<const sycl::float4*>(x + m * K + b * 32);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const sycl::float4 xv = xp[i];
#pragma unroll
                        for (int r = 0; r < Q_R; ++r)
                            a[m][r] += wf[r][4 * i] * xv.x() + wf[r][4 * i + 1] * xv.y() + wf[r][4 * i + 2] * xv.z() + wf[r][4 * i + 3] * xv.w();
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < Q_MM; ++m) {
            if (m < M) {
#pragma unroll
                for (int r = 0; r < Q_R; ++r) {
                    const float s = sycl::reduce_over_group(sg, a[m][r], sycl::plus<float>());
                    const int64_t n = n0 + r;
                    if (lane == 0 && n < N) {
                        float* o = out + m * ldo + n;
                        *o = acc ? *o + s : s;
                    }
                }
            }
        }
    });
}

// ---- q / k norm + RoPE + the cache's rows: a sub-group a (row, head) of the 16 q, 8 k and 8 v heads; each lane holds
// features l + 16 j (j < 8), the rotated pairs (i, 64 + i) are j and j + 4
void qkv(sycl::queue& q, const float* qkv, int64_t T, const float* qn, const float* kn, float eps, const float* cs, int64_t pos0, half* qo, half* kc,
         half* vc, int64_t row0) {
    constexpr int NH = HQ + 2 * HKV;
    const int64_t subs = T * NH;
    constexpr int NSG = 4;
    const int64_t groups = (subs + NSG - 1) / NSG;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * NSG * SG), sycl::range<1>(NSG * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t gw = it.get_group(0) * NSG + sg.get_group_linear_id();
        const int64_t t = gw / NH;
        const int hh = (int) (gw % NH);
        if (t >= T) return;   // the whole sub-group: it holds one (row, head)
        const float* p = qkv + t * NH * HD + hh * HD;
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = p[l + SG * j];
        if (hh >= HQ + HKV) {
            half* o = vc + (row0 + t) * HKV * HD + (hh - HQ - HKV) * HD;
#pragma unroll
            for (int j = 0; j < 8; ++j) o[l + SG * j] = (half) v[j];
            return;
        }
        const float* w = hh < HQ ? qn : kn;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) ss += v[j] * v[j];
        const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) HD + eps);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = (v[j] * inv) * w[l + SG * j];
        const float* crow = cs + (pos0 + t) * HD;
        float o[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = l + SG * j;
            const float c = crow[2 * i], s = crow[2 * i + 1];
            o[j] = v[j] * c - v[j + 4] * s;
            o[j + 4] = v[j + 4] * c + v[j] * s;
        }
        half* dst = hh < HQ ? qo + t * HQ * HD + hh * HD : kc + (row0 + t) * HKV * HD + (hh - HQ) * HD;
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[l + SG * j] = (half) o[j];
    });
}

// ---- one query row against the cache: a work-group per (key head, split of the keys), NSG_A sub-groups taking the
// split's keys in turn, each lane DL = 8 features of the G = 2 query heads that share the key head; the sub-groups'
// running (max, sum, values) merged in local memory, the splits by a second kernel (mm3.cpp's scheme, the cache
// rows-major here). The keys and values are read as 16-byte vectors; every loop bound by n.
constexpr int64_t KB = 64;     // keys a split
constexpr int NSG_A = 4, DL = 8, G = 2;

int64_t splits(int64_t n) { return n > KB ? (n + KB - 1) / KB : 1; }

void attn(sycl::queue& qu, const half* q, const half* kc, const half* vc, int64_t n, float* out, float* part) {
    constexpr int WG = SG * NSG_A;
    const int64_t ns = splits(n);
    const float scale = 1.0f / sycl::sqrt((float) HD);
    qu.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NSG_A * G * (HD + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (HKV * ns) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int64_t gi = it.get_group(0), sp = gi % ns, hk = gi / ns;
            const int64_t j0 = sp * (ns > 1 ? KB : n), j1 = sycl::min(n, ns > 1 ? (sp + 1) * KB : n);
            float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const half* qr = q + (hk * G + g) * HD + lane * DL;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    qv[g][i] = (float) qr[i] * scale;
                    acc[g][i] = 0.0f;
                }
                m[g] = -INFINITY;
                l[g] = 0.0f;
            }
            for (int64_t j = j0 + sgi; j < j1; j += NSG_A) {
                const sycl::vec<half, DL> kv = *reinterpret_cast<const sycl::vec<half, DL>*>(kc + (j * HKV + hk) * HD + lane * DL);
                const sycl::vec<half, DL> vv = *reinterpret_cast<const sycl::vec<half, DL>*>(vc + (j * HKV + hk) * HD + lane * DL);
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    float d = 0.0f;
#pragma unroll
                    for (int i = 0; i < DL; ++i) d += qv[g][i] * (float) kv[i];
                    const float sc = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                    const float mn = sycl::fmax(m[g], sc);
                    const float c = sycl::exp(m[g] - mn), e = sycl::exp(sc - mn);
#pragma unroll
                    for (int i = 0; i < DL; ++i) acc[g][i] = acc[g][i] * c + e * (float) vv[i];
                    l[g] = l[g] * c + e;
                    m[g] = mn;
                }
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float* o = &slm[(sgi * G + g) * (HD + 2)];
#pragma unroll
                for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
                if (lane == 0) {
                    o[HD] = m[g];
                    o[HD + 1] = l[g];
                }
            }
            sycl::group_barrier(it.get_group());
            for (int idx = (int) it.get_local_id(0); idx < G * HD; idx += WG) {
                const int g = idx / HD, d = idx % HD;
                float mx = -INFINITY;
                for (int k = 0; k < NSG_A; ++k) mx = sycl::fmax(mx, slm[(k * G + g) * (HD + 2) + HD]);
                float L = 0.0f, A = 0.0f;
                if (mx != -INFINITY) {
                    for (int k = 0; k < NSG_A; ++k) {
                        const float* o = &slm[(k * G + g) * (HD + 2)];
                        const float e = sycl::exp(o[HD] - mx);
                        L += o[HD + 1] * e;
                        A += o[d] * e;
                    }
                }
                const int64_t hq = hk * G + g;
                if (ns == 1) {
                    out[hq * HD + d] = L > 0.0f ? A / L : 0.0f;
                } else {
                    float* p = part + (hq * ns + sp) * (HD + 2);
                    p[d] = A;
                    if (d == 0) {
                        p[HD] = mx;
                        p[HD + 1] = L;
                    }
                }
            }
        });
    });
    if (ns > 1) {
        qu.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) HQ * HD), sycl::range<1>(HD)), [=](sycl::nd_item<1> it) {
            const int64_t hq = it.get_group(0), d = it.get_local_id(0);
            const float* p = part + hq * ns * (HD + 2);
            float mx = -INFINITY;
            for (int64_t k = 0; k < ns; ++k) mx = sycl::fmax(mx, p[k * (HD + 2) + HD]);
            float L = 0.0f, A = 0.0f;
            if (mx != -INFINITY) {
                for (int64_t k = 0; k < ns; ++k) {
                    const float e = sycl::exp(p[k * (HD + 2) + HD] - mx);
                    L += p[k * (HD + 2) + HD + 1] * e;
                    A += p[k * (HD + 2) + d] * e;
                }
            }
            out[hq * HD + d] = L > 0.0f ? A / L : 0.0f;
        });
    }
}

}  // namespace

extern "C" {

int ns_audio_yue2_gemv_q8(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* w, int64_t N, float* out, int64_t ldo, int acc) {
    NS_TRY
    if (M <= 0 || N <= 0) return 0;
    if (M > Q_MM || K % 32 != 0) return ns_fail("ns_audio_yue2_gemv_q8: at most 8 rows, K a multiple of 32");
    gemv_q8(g->q, x, (int) M, K, (const uint8_t*) w, N, out, ldo ? ldo : N, acc != 0);
    return 0;
    NS_CATCH
}

int ns_audio_yue2_embed_q8(ns_gpu* g, const void* table, int64_t K, const int32_t* ids, int64_t n, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    if (K % 32 != 0) return ns_fail("ns_audio_yue2_embed_q8: K a multiple of 32");
    const uint8_t* t = (const uint8_t*) table;
    const int64_t nb = K / 32;
    g->q.parallel_for(sycl::range<1>((size_t) (n * nb)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / nb, b = i % nb;
        float v[32];
        q8_block(t + (int64_t) ids[r] * nb * QB + b * QB, v);
#pragma unroll
        for (int j = 0; j < 32; ++j) out[r * K + b * 32 + j] = v[j];
    });
    return 0;
    NS_CATCH
}

int ns_audio_yue2_qkv(ns_gpu* g, const float* x, int64_t T, const float* qn, const float* kn, float eps, const float* cs, int64_t pos0, void* q_out,
                      void* kc, void* vc, int64_t row0) {
    NS_TRY
    if (T <= 0) return 0;
    qkv(g->q, x, T, qn, kn, eps, cs, pos0, (half*) q_out, (half*) kc, (half*) vc, row0);
    return 0;
    NS_CATCH
}

int64_t ns_audio_yue2_attn_scratch(int64_t n) {
    const int64_t ns = splits(n);
    return ns > 1 ? HQ * ns * (HD + 2) : 0;
}

int ns_audio_yue2_attn(ns_gpu* g, const void* q, const void* kc, const void* vc, int64_t n, float* out, float* part) {
    NS_TRY
    if (n <= 0) return ns_fail("ns_audio_yue2_attn: no keys");
    if (splits(n) > 1 && !part) return ns_fail("ns_audio_yue2_attn: the splits' scratch is missing");
    attn(g->q, (const half*) q, (const half*) kc, (const half*) vc, n, out, part);
    return 0;
    NS_CATCH
}

int ns_audio_yue2_heads(ns_gpu* g, const void* in, void* out, int64_t S, int to_grouped) {
    NS_TRY
    if (S <= 0) return 0;
    const uint16_t* a = (const uint16_t*) in;
    uint16_t* b = (uint16_t*) out;
    // a row's head at a time (16-byte groups): (row r, query head hk G + g, i) <-> (row 2 r + g, head hk, i)
    g->q.parallel_for(sycl::range<1>((size_t) S * HQ * (HD / 8)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], e = i % (HD / 8), rh = i / (HD / 8), r = rh / HQ, hq = rh % HQ;
        const int64_t hk = hq / G, gg = hq % G;
        const int64_t flat = (r * HQ + hq) * HD + e * 8;
        const int64_t grouped = ((r * G + gg) * HKV + hk) * HD + e * 8;
        const sycl::vec<uint16_t, 8> v = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(a + (to_grouped ? flat : grouped));
        *reinterpret_cast<sycl::vec<uint16_t, 8>*>(b + (to_grouped ? grouped : flat)) = v;
    });
    return 0;
    NS_CATCH
}

int ns_audio_yue2_add_vec_rows(ns_gpu* g, float* x, int64_t R, int64_t C, const float* a, const float* rows) {
    NS_TRY
    if (R * C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (R * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        x[i] += a[i % C] + rows[i];
    });
    return 0;
    NS_CATCH
}

int ns_audio_yue2_axpy(ns_gpu* g, float* out, const float* a, const float* b, int64_t n, float s) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { out[id] = a[id] + s * b[id]; });
    return 0;
    NS_CATCH
}

}  // extern "C"
