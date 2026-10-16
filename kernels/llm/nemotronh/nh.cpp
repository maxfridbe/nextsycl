// nh.cpp: the nemotronh engine's own kernels (nh.h) - what Nemotron-H has that the shared ones do not: the Mamba-2
// mixer's conv (with its bias), its scan and its gated group norm, relu^2 for the experts, and attention without
// positions (head 128, 16 query heads a key head) against a half cache. After llama.cpp's nemotron-h / mamba-base
// graphs and ggml's ssm_conv / ssm_scan.
#include "ns.h"
#include "ns_internal.hpp"
#include "nh.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;

// ---- cached causal attention (the qwen35 engine's design, q35.cpp): a work-group per (row, group of G query heads,
// split of the keys), NSG sub-groups taking the split's keys in turn, each lane DL features of the G heads (which
// share one key head); the sub-groups' (max, sum, weighted values) merged in local memory, the splits (decode) by a
// second kernel
constexpr int NSG_A = 4;

int64_t split_keys(int64_t S, int64_t kv, int64_t groups) {
    if (S != 1) return kv;
    const int64_t want = (kv * groups + 511) / 512;
    return sycl::max<int64_t>(32, (want + 31) / 32 * 32);
}
int64_t splits(int64_t S, int64_t kv, int64_t groups) {
    const int64_t kb = split_keys(S, kv, groups);
    return kv > kb ? (kv + kb - 1) / kb : 1;
}

template <int DL, int G>
void attn(sycl::queue& q, const float* qp, int64_t qs, const half* kc, const half* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t T, int64_t p0,
          float* out, float* part) {
    constexpr int D = SG * DL, WG = SG * NSG_A;
    const int64_t groups = Hq / G, rep = Hq / Hkv;
    const int64_t ns = splits(S, p0 + S, groups), KB = split_keys(S, p0 + S, groups);
    const float scale = 1.0f / sycl::sqrt((float) D);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NSG_A * G * (D + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * groups * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t vg = gi % groups, r = gi / groups;
            const int64_t hk = vg * G / rep;
            const int64_t kv_end = p0 + r + 1;
            const int64_t j0 = sp * (ns > 1 ? KB : kv_end), j1 = sycl::min(kv_end, ns > 1 ? (sp + 1) * KB : kv_end);
            float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float* qr = qp + r * qs + (vg * G + g) * D + lane * DL;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    qv[g][i] = qr[i] * scale;
                    acc[g][i] = 0.0f;
                }
                m[g] = -INFINITY;
                l[g] = 0.0f;
            }
            const half* kb = kc + hk * T * D + lane * DL;
            const half* vb = vc + hk * T * D + lane * DL;
            for (int64_t j = j0 + sgi; j < j1; j += NSG_A) {
                const sycl::vec<half, DL> kv = *reinterpret_cast<const sycl::vec<half, DL>*>(kb + j * D);
                const sycl::vec<half, DL> vv = *reinterpret_cast<const sycl::vec<half, DL>*>(vb + j * D);
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
                float* o = &slm[(sgi * G + g) * (D + 2)];
#pragma unroll
                for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
                if (lane == 0) {
                    o[D] = m[g];
                    o[D + 1] = l[g];
                }
            }
            sycl::group_barrier(it.get_group());
            for (int idx = (int) it.get_local_id(0); idx < G * D; idx += WG) {
                const int g = idx / D, d = idx % D;
                float mx = -INFINITY;
                for (int k = 0; k < NSG_A; ++k) mx = sycl::fmax(mx, slm[(k * G + g) * (D + 2) + D]);
                float L = 0.0f, A = 0.0f;
                if (mx != -INFINITY) {
                    for (int k = 0; k < NSG_A; ++k) {
                        const float* o = &slm[(k * G + g) * (D + 2)];
                        const float e = sycl::exp(o[D] - mx);
                        L += o[D + 1] * e;
                        A += o[d] * e;
                    }
                }
                const int64_t hq = vg * G + g;
                if (ns == 1) {
                    out[r * Hq * D + hq * D + d] = L > 0.0f ? A / L : 0.0f;
                } else {
                    float* p = part + ((r * Hq + hq) * ns + sp) * (D + 2);
                    p[d] = A;
                    if (d == 0) {
                        p[D] = mx;
                        p[D + 1] = L;
                    }
                }
            }
        });
    });
    if (ns > 1) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * Hq) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
            const int64_t rh = it.get_group(0), d = it.get_local_id(0);
            const float* p = part + rh * ns * (D + 2);
            float mx = -INFINITY;
            for (int64_t k = 0; k < ns; ++k) mx = sycl::fmax(mx, p[k * (D + 2) + D]);
            float L = 0.0f, A = 0.0f;
            if (mx != -INFINITY) {
                for (int64_t k = 0; k < ns; ++k) {
                    const float e = sycl::exp(p[k * (D + 2) + D] - mx);
                    L += p[k * (D + 2) + D + 1] * e;
                    A += p[k * (D + 2) + d] * e;
                }
            }
            out[rh * D + d] = L > 0.0f ? A / L : 0.0f;
        });
    }
}

// the query heads a work-group takes: 8 (a key head's 16 in two groups: the registers of 16 would spill)
constexpr int GA = 8;
}  // namespace

extern "C" {

int ns_nh_conv(ns_gpu* g, const float* x, int64_t xs, float* state, const float* w, const float* b, float* out, int64_t T, int64_t C, int k,
               float* snap) {
    NS_TRY
    if (T <= 0) return 0;
    if (k < 2 || k > 8) return ns_fail("ns_nh_conv: a kernel of 2 to 8");
    // a work-item a channel: its k-1 inputs before, then the rows in order (the state's last k-1 written back)
    g->q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = id[0];
        float win[8];
        for (int j = 0; j < k - 1; ++j) win[j] = state[j * C + c];
        const float bias = b ? b[c] : 0.0f;
        for (int64_t t = 0; t < T; ++t) {
            win[k - 1] = x[t * xs + c];
            float s = bias;
            for (int j = 0; j < k; ++j) s += win[j] * w[c * k + j];
            out[t * C + c] = s / (1.0f + sycl::exp(-s));
            for (int j = 0; j < k - 1; ++j) win[j] = win[j + 1];
            if (snap && t + 1 < T)
                for (int j = 0; j < k - 1; ++j) snap[(t * (k - 1) + j) * C + c] = win[j];
        }
        for (int j = 0; j < k - 1; ++j) state[j * C + c] = win[j];
    });
    return 0;
    NS_CATCH
}

int ns_nh_ssm(ns_gpu* g, const float* xbc, int64_t xs, const float* dt, int64_t dts, const float* dt_bias, const float* A, const float* Dp, float* S,
              float* y, int64_t T, int64_t H, int64_t P, int64_t N, int64_t G, float* snap) {
    NS_TRY
    if (T <= 0) return 0;
    if (N % SG != 0 || N > 256 || H % G != 0 || P > 64) return ns_fail("ns_nh_ssm: d_state a multiple of 16 up to 256, head dim up to 64, the heads a multiple of the groups");
    const int64_t hpg = H / G, xd = H * P;
    if (T <= 8) {
        // decode widths: a sub-group per (head, p), its lanes the state's N values (N / 16 each), B and C read from
        // global memory (a few rows: no staging) - the tiled kernel below costs 104 against 30 us for one row
        g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (H * P) * SG), sycl::range<1>(SG)),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            const int64_t hp = it.get_group(0), h = hp / P, p = hp % P, gr = h / hpg;
            const int64_t per = N / SG;
            float s[16];
            float* sp = S + (h * P + p) * N;
            for (int64_t i = 0; i < per; ++i) s[i] = sp[lane + i * SG];
            const float a = A[h], dd = Dp[h], bias = dt_bias[h];
            for (int64_t t = 0; t < T; ++t) {
                const float* row = xbc + t * xs;
                float d = dt[t * dts + h] + bias;
                d = d <= 20.0f ? sycl::log1p(sycl::exp(d)) : d;
                const float dA = sycl::exp(d * a);
                const float xv = row[h * P + p];
                const float xdt = xv * d;
                const float* B = row + xd + gr * N;
                const float* C = row + xd + G * N + gr * N;
                float acc = 0.0f;
                for (int64_t i = 0; i < per; ++i) {
                    const int64_t n = lane + i * SG;
                    s[i] = s[i] * dA + B[n] * xdt;
                    acc += s[i] * C[n];
                }
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0) y[t * xd + h * P + p] = acc + dd * xv;
                if (snap && t + 1 < T) {
                    float* o = snap + ((t * H + h) * P + p) * N;
                    for (int64_t i = 0; i < per; ++i) o[lane + i * SG] = s[i];
                }
            }
            for (int64_t i = 0; i < per; ++i) sp[lane + i * SG] = s[i];
        });
        return 0;
    }
    constexpr int TT = 32;   // rows a tile
    // a work-group a head: a sub-group per p, its lanes the state's N values (N / 16 each); the head's B, C, x, dt' and
    // decay for a tile of rows staged in local memory once (every p of the head reads them: from global memory each
    // read them 64 times), then the rows in order - decay and update, C . S reduced over the sub-group
    const size_t wg = (size_t) P * SG;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sb(sycl::range<1>((size_t) TT * N), h), scl(sycl::range<1>((size_t) TT * N), h);
        sycl::local_accessor<float, 1> sx(sycl::range<1>((size_t) TT * P), h), sdt(sycl::range<1>(TT), h), sda(sycl::range<1>(TT), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) H * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            const int64_t hh = it.get_group(0), p = sg.get_group_linear_id(), gr = hh / hpg;
            const int64_t li = it.get_local_id(0);
            const int64_t per = N / SG;
            float s[16];
            float* sp = S + (hh * P + p) * N;
            for (int64_t i = 0; i < per; ++i) s[i] = sp[lane + i * SG];
            const float a = A[hh], dd = Dp[hh], bias = dt_bias[hh];
            for (int64_t t0 = 0; t0 < T; t0 += TT) {
                const int64_t tn = sycl::min<int64_t>(TT, T - t0);
                sycl::group_barrier(it.get_group());
                for (int64_t i = li; i < tn * N; i += (int64_t) wg) {
                    const int64_t r = i / N, n = i % N;
                    const float* row = xbc + (t0 + r) * xs + xd + gr * N;
                    sb[i] = row[n];
                    scl[i] = row[G * N + n];
                }
                for (int64_t i = li; i < tn * P; i += (int64_t) wg) sx[i] = xbc[(t0 + i / P) * xs + hh * P + i % P];
                if (li < tn) {
                    float d = dt[(t0 + li) * dts + hh] + bias;
                    d = d <= 20.0f ? sycl::log1p(sycl::exp(d)) : d;
                    sdt[li] = d;
                    sda[li] = sycl::exp(d * a);
                }
                sycl::group_barrier(it.get_group());
                // the rows' partial sums kept, reduced after the tile (no reduction in the recurrence's way)
                float part[TT];
#pragma unroll
                for (int r = 0; r < TT; ++r) {
                    part[r] = 0.0f;
                    if (r < tn) {
                        const float xdt = sx[r * P + p] * sdt[r], dA = sda[r];
                        for (int64_t i = 0; i < per; ++i) {
                            const int64_t n = lane + i * SG;
                            s[i] = s[i] * dA + sb[r * N + n] * xdt;
                            part[r] += s[i] * scl[r * N + n];
                        }
                        if (snap && t0 + r + 1 < T) {
                            float* o = snap + (((t0 + r) * H + hh) * P + p) * N;
                            for (int64_t i = 0; i < per; ++i) o[lane + i * SG] = s[i];
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < TT; ++r) {
                    const float acc = sycl::reduce_over_group(sg, part[r], sycl::plus<float>());
                    if (lane == 0 && r < tn) y[(t0 + r) * xd + hh * P + p] = acc + dd * sx[r * P + p];
                }
            }
            for (int64_t i = 0; i < per; ++i) sp[lane + i * SG] = s[i];
        });
    });
    return 0;
    NS_CATCH
}

// ---- the chunked scan (Mamba-2's chunked form): rows in chunks of SL; within a chunk the outputs and the chunk's
// state contribution as small products, the state carried over the chunks after, each chunk's entering state into its
// outputs last. Decays within a chunk from differences of the log decays' running sums. Measured (8K prompt, a layer of
// 2,048 rows): B70 5.3 -> 3.7 ms, B65 10.2 -> 5.4 against row after row; of the B70's: C . B^T ~0.1, the chunks' work
// 1.8, the carry 0.2, the entering states' outputs 1.1 (the per-chunk states, 134 MB, through memory three times)
namespace {
constexpr int SL = 64, SWG = 256;
}

int64_t ns_nh_ssm_chunk_scratch(int64_t T, int64_t H, int64_t P, int64_t N) {
    const int64_t nc = (T + SL - 1) / SL;
    // the chunks' contributions / entering states, C . B^T a (chunk, group), a head's running decays, its whole decay
    return nc * H * P * N + nc * H * SL * SL + nc * H * SL + nc * H;
}

int ns_nh_ssm_chunk(ns_gpu* g, const float* xbc, int64_t xs, const float* dt, int64_t dts, const float* dt_bias, const float* A, const float* Dp,
                    float* S, float* y, int64_t T, int64_t H, int64_t P, int64_t N, int64_t G, float* scratch) {
    NS_TRY
    if (T <= 0) return 0;
    if (P != 64 || N != 128 || H % G != 0) return ns_fail("ns_nh_ssm_chunk: head dim 64, d_state 128");
    const int64_t hpg = H / G, xd = H * P, nc = (T + SL - 1) / SL;
    float* dS = scratch;                       // [nc][H][P][N]: the chunks' contributions, then their entering states
    float* cb = dS + nc * H * P * N;           // [nc][G][SL][SL]: C_t . B_s
    float* eL = cb + nc * H * SL * SL;         // [nc][H][SL]: exp(the head's log decays summed to row t)
    float* dec = eL + nc * H * SL;             // [nc][H]: a chunk's whole decay
    // 0: C . B^T a (chunk, group), both tiles in local memory; a lane 4 x 4 of it
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::float4, 1> sB(sycl::range<1>(SL * 32), h), sC(sycl::range<1>(SL * 32), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nc * G) * SWG), sycl::range<1>(SWG)), [=](sycl::nd_item<1> it) {
            const int li = (int) it.get_local_id(0);
            const int64_t c = it.get_group(0) / G, gr = it.get_group(0) % G, r0 = c * SL;
            const int lc = (int) sycl::min<int64_t>(SL, T - r0);
            for (int i = li; i < SL * 32; i += SWG) {
                const int r = i / 32, n4 = i % 32;
                const float* row = xbc + (r0 + r) * xs + xd + gr * 128 + 4 * n4;
                sB[i] = r < lc ? *reinterpret_cast<const sycl::float4*>(row) : sycl::float4(0.0f);
                sC[i] = r < lc ? *reinterpret_cast<const sycl::float4*>(row + G * 128) : sycl::float4(0.0f);
            }
            sycl::group_barrier(it.get_group());
            const int tb = li / 16, sb = li % 16;
            float acc[4][4] = {};
            for (int n4 = 0; n4 < 32; ++n4) {
                sycl::float4 cv[4], bv[4];
#pragma unroll
                for (int a = 0; a < 4; ++a) cv[a] = sC[(4 * tb + a) * 32 + n4];
#pragma unroll
                for (int b = 0; b < 4; ++b) bv[b] = sB[(4 * sb + b) * 32 + n4];
#pragma unroll
                for (int a = 0; a < 4; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b) acc[a][b] += sycl::dot(cv[a], bv[b]);
            }
            float* o = cb + ((c * G + gr) * SL) * SL;
#pragma unroll
            for (int a = 0; a < 4; ++a)
                *reinterpret_cast<sycl::float4*>(o + (4 * tb + a) * SL + 4 * sb) = sycl::float4(acc[a][0], acc[a][1], acc[a][2], acc[a][3]);
        });
    });
    // 1: a work-group a (chunk, head): the running log decays; W[t][s] = (C_t . B_s) exp(L_t - L_s) dt'_s; y = W x + D x;
    //    the contribution sum_s exp(L_end - L_s) dt'_s x_s B_s
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sW(sycl::range<1>(SL * SL), h), sX(sycl::range<1>(SL * 64), h);
        sycl::local_accessor<float, 1> sdt(sycl::range<1>(SL), h), sL(sycl::range<1>(SL), h), sel(sycl::range<1>(SL), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nc * H) * SWG), sycl::range<1>(SWG)), [=](sycl::nd_item<1> it) {
            const int li = (int) it.get_local_id(0);
            const int64_t c = it.get_group(0) / H, hh = it.get_group(0) % H, gr = hh / hpg, r0 = c * SL;
            const int lc = (int) sycl::min<int64_t>(SL, T - r0);
            // the log decays and their running sums (a work-group scan over the first 64 lanes' values)
            float la = 0.0f;
            if (li < SL) {
                float d = 0.0f;
                if (li < lc) {
                    d = dt[(r0 + li) * dts + hh] + dt_bias[hh];
                    d = d <= 20.0f ? sycl::log1p(sycl::exp(d)) : d;
                    la = d * A[hh];
                }
                sdt[li] = d;
            }
            const float run = sycl::inclusive_scan_over_group(it.get_group(), li < SL ? la : 0.0f, sycl::plus<float>());
            if (li < SL) sL[li] = run;
            for (int i = li; i < SL * 16; i += SWG) {
                const int r = i / 16, p4 = i % 16;
                *reinterpret_cast<sycl::float4*>(&sX[r * 64 + 4 * p4]) =
                    r < lc ? *reinterpret_cast<const sycl::float4*>(xbc + (r0 + r) * xs + hh * 64 + 4 * p4) : sycl::float4(0.0f);
            }
            sycl::group_barrier(it.get_group());
            const float Lend = sL[lc - 1];
            if (li < SL) {
                eL[(c * H + hh) * SL + li] = li < lc ? sycl::exp(sL[li]) : 0.0f;
                sel[li] = li < lc ? sycl::exp(Lend - sL[li]) * sdt[li] : 0.0f;
                if (li == 0) dec[c * H + hh] = sycl::exp(Lend);
            }
            {
                const float* cbg = cb + ((c * G + gr) * SL) * SL;
                for (int i = li; i < SL * SL; i += SWG) {
                    const int tt = i / SL, s = i % SL;
                    sW[i] = (s <= tt && tt < lc) ? cbg[i] * sycl::exp(sL[tt] - sL[s]) * sdt[s] : 0.0f;
                }
            }
            sycl::group_barrier(it.get_group());
            // y [t][p]: a lane 4 rows x 4 columns
            {
                const int tb = li / 16, pb = li % 16;
                float acc[4][4] = {};
                const int tmax = sycl::min(lc, 4 * tb + 4);
                for (int s = 0; s < tmax; ++s) {
                    const sycl::float4 xv = *reinterpret_cast<const sycl::float4*>(&sX[s * 64 + 4 * pb]);
#pragma unroll
                    for (int a = 0; a < 4; ++a) {
                        const float w = sW[(4 * tb + a) * SL + s];
                        acc[a][0] += w * xv.x(); acc[a][1] += w * xv.y(); acc[a][2] += w * xv.z(); acc[a][3] += w * xv.w();
                    }
                }
                const float dd = Dp[hh];
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const int tt = 4 * tb + a;
                    if (tt >= lc) continue;
                    const sycl::float4 xv = *reinterpret_cast<const sycl::float4*>(&sX[tt * 64 + 4 * pb]);
                    *reinterpret_cast<sycl::float4*>(y + (r0 + tt) * xd + hh * 64 + 4 * pb) =
                        sycl::float4(acc[a][0] + dd * xv.x(), acc[a][1] + dd * xv.y(), acc[a][2] + dd * xv.z(), acc[a][3] + dd * xv.w());
                }
            }
            // the contribution [p][n]: a lane 4 p x 8 n, B's rows from memory (the group's heads share them: cached)
            {
                const int pb = li / 16, nb = li % 16;
                float acc[4][8] = {};
                for (int s = 0; s < lc; ++s) {
                    const sycl::float4 xv = *reinterpret_cast<const sycl::float4*>(&sX[s * 64 + 4 * pb]);
                    const float e = sel[s];
                    const float xe[4] = {xv.x() * e, xv.y() * e, xv.z() * e, xv.w() * e};
                    const float* br = xbc + (r0 + s) * xs + xd + gr * 128 + 8 * nb;
                    const sycl::float4 b0 = *reinterpret_cast<const sycl::float4*>(br), b1 = *reinterpret_cast<const sycl::float4*>(br + 4);
                    const float bv[8] = {b0.x(), b0.y(), b0.z(), b0.w(), b1.x(), b1.y(), b1.z(), b1.w()};
#pragma unroll
                    for (int b = 0; b < 8; ++b)
#pragma unroll
                        for (int a = 0; a < 4; ++a) acc[a][b] += xe[a] * bv[b];
                }
                float* o = dS + ((c * H + hh) * 64) * 128;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    float* orow = o + (4 * pb + a) * 128 + 8 * nb;
                    *reinterpret_cast<sycl::float4*>(orow) = sycl::float4(acc[a][0], acc[a][1], acc[a][2], acc[a][3]);
                    *reinterpret_cast<sycl::float4*>(orow + 4) = sycl::float4(acc[a][4], acc[a][5], acc[a][6], acc[a][7]);
                }
            }
        });
    });
    // 2: the state over the chunks, an element a work-item: each chunk's entering state in place of its contribution
    const int64_t el = H * P * N;
    g->q.parallel_for(sycl::range<1>((size_t) el), [=](sycl::id<1> id) {
        const int64_t i = id[0], hh = i / (P * N);
        float s = S[i];
        for (int64_t c = 0; c < nc; ++c) {
            float* d = dS + c * el + i;
            const float v = *d;
            *d = s;
            s = dec[c * H + hh] * s + v;
        }
        S[i] = s;
    });
    // 3: a work-group a (chunk, head): y [t][p] += exp(L_t) C_t . S_in[p]; C's tile and the entering state in local memory
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::float4, 1> sC(sycl::range<1>(SL * 32), h), sS(sycl::range<1>(64 * 32), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nc * H) * SWG), sycl::range<1>(SWG)), [=](sycl::nd_item<1> it) {
            const int li = (int) it.get_local_id(0);
            const int64_t c = it.get_group(0) / H, hh = it.get_group(0) % H, gr = hh / hpg, r0 = c * SL;
            const int lc = (int) sycl::min<int64_t>(SL, T - r0);
            const float* sin = dS + ((c * H + hh) * 64) * 128;
            for (int i = li; i < SL * 32; i += SWG) {
                const int r = i / 32, n4 = i % 32;
                sC[i] = r < lc ? *reinterpret_cast<const sycl::float4*>(xbc + (r0 + r) * xs + xd + G * 128 + gr * 128 + 4 * n4) : sycl::float4(0.0f);
                sS[i] = *reinterpret_cast<const sycl::float4*>(sin + 4 * i);
            }
            sycl::group_barrier(it.get_group());
            const int tb = li / 16, pb = li % 16;
            float acc[4][4] = {};
            for (int n4 = 0; n4 < 32; ++n4) {
                sycl::float4 cv[4], sv[4];
#pragma unroll
                for (int a = 0; a < 4; ++a) cv[a] = sC[(4 * tb + a) * 32 + n4];
#pragma unroll
                for (int b = 0; b < 4; ++b) sv[b] = sS[(4 * pb + b) * 32 + n4];
#pragma unroll
                for (int a = 0; a < 4; ++a)
#pragma unroll
                    for (int b = 0; b < 4; ++b) acc[a][b] += sycl::dot(cv[a], sv[b]);
            }
            const float* el_ = eL + (c * H + hh) * SL;
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const int tt = 4 * tb + a;
                if (tt >= lc) continue;
                float* yo = y + (r0 + tt) * xd + hh * 64 + 4 * pb;
                const float e = el_[tt];
                sycl::float4 v = *reinterpret_cast<sycl::float4*>(yo);
                v += sycl::float4(acc[a][0], acc[a][1], acc[a][2], acc[a][3]) * e;
                *reinterpret_cast<sycl::float4*>(yo) = v;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_nh_gated_norm(ns_gpu* g, const float* y, const float* z, int64_t zs, const float* w, float* out, int64_t T, int64_t C, int64_t group, float eps) {
    NS_TRY
    if (T <= 0) return 0;
    if (C % group != 0) return ns_fail("ns_nh_gated_norm: the channels a multiple of the group");
    const int64_t ng = C / group;
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * ng) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t tg = it.get_group(0), t = tg / ng, gi = tg % ng;
        const int64_t l = it.get_local_id(0);
        const float* yr = y + t * C + gi * group;
        const float* zr = z + t * zs + gi * group;
        float ss = 0.0f;
        for (int64_t i = l; i < group; i += WG) {
            const float zv = zr[i];
            const float v = yr[i] * zv / (1.0f + sycl::exp(-zv));
            ss += v * v;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float inv = sycl::rsqrt(ss / (float) group + eps);
        float* o = out + t * C + gi * group;
        for (int64_t i = l; i < group; i += WG) {
            const float zv = zr[i];
            o[i] = yr[i] * zv / (1.0f + sycl::exp(-zv)) * inv * w[gi * group + i];
        }
    });
    return 0;
    NS_CATCH
}

int ns_nh_relu2(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = sycl::fmax(x[i], 0.0f);
        x[i] = v * v;
    });
    return 0;
    NS_CATCH
}

int ns_nh_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                   void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_nh_kv_store: past the cache's capacity");
    half* ko = (half*) kc;
    half* vo = (half*) vc;
    g->q.parallel_for(sycl::range<1>((size_t) (T * Hkv * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
        const int64_t at = (h * cap + p0 + t) * D + d;
        ko[at] = (half) k[t * ks + h * D + d];
        vo[at] = (half) v[t * vs + h * D + d];
    });
    return 0;
    NS_CATCH
}

int64_t ns_nh_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0) {
    (void) Hkv;
    (void) D;
    const int64_t ns = splits(T, p0 + T, Hq / GA);
    return ns > 1 ? T * Hq * ns * (D + 2) : 0;
}

int ns_nh_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap,
               int64_t p0, float* out, float* scratch) {
    NS_TRY
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_nh_attn: past the cache's capacity");
    if (D != 128 || Hq % GA != 0 || Hq % Hkv != 0 || (Hq / Hkv) % GA != 0) return ns_fail("ns_nh_attn: head 128, query heads a multiple of 8 a key head");
    if (ns_nh_attn_scratch(T, Hq, Hkv, D, p0) > 0 && !scratch) return ns_fail("ns_nh_attn: no scratch");
    attn<8, GA>(g->q, q, qs, (const half*) kc, (const half*) vc, T, Hq, Hkv, cap, p0, out, scratch);
    return 0;
    NS_CATCH
}

int ns_nh_q6k_f16(ns_gpu* g, const void* src, int64_t n, uint16_t* dst) {
    NS_TRY
    if (n % 256 != 0) return ns_fail("ns_nh_q6k_f16: n a multiple of 256");
    const uint8_t* s = static_cast<const uint8_t*>(src);
    half* y = reinterpret_cast<half*>(dst);
    // ggml's dequantize_row_q6_K: a work-item a (block, half, l), four values each
    g->q.parallel_for(sycl::range<1>((size_t) (n / 256 * 64)), [=](sycl::id<1> id) {
        const int64_t i = id[0], b = i / 64, hn = (i % 64) / 32, l = i % 32;
        const uint8_t* blk = s + b * 210;
        const uint8_t* ql = blk + 64 * hn;
        const uint8_t* qh = blk + 128 + 32 * hn;
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + 8 * hn;
        const float d = (float) *reinterpret_cast<const half*>(blk + 208);
        const int is = (int) l / 16;
        const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        half* o = y + b * 256 + 128 * hn;
        o[l] = (half) (d * sc[is] * q1);
        o[l + 32] = (half) (d * sc[is + 2] * q2);
        o[l + 64] = (half) (d * sc[is + 4] * q3);
        o[l + 96] = (half) (d * sc[is + 6] * q4);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
