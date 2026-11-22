// dec.cpp: the inkling2 silo's fused decode steps (silo.h) - a layer's small steps in few launches: the depth
// attention with the norms and the activations' quantization after it; the convolution of k and v with their clip /
// norm and the cache write, with q's and r's clips (and the full layers' log scale); the attention split by absolute
// key positions and its merge with the sink, the output gate and the quantization for the output product; the
// convolutions of the attention's and the FFN's outputs with the residual; the loop controllers in two launches; the
// float products of a layer in one. Every row's arithmetic is the same whatever rows share a pass.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;

inline size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// ggml's Q8_0 rounding of a block of 32 (16 lanes, 2 values each)
inline void quant_block(sycl::sub_group sg, int lane, const float* x, int8_t* q, float* d) {
    const float a = x[lane * 2], b = x[lane * 2 + 1];
    const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(a), sycl::fabs(b)), sycl::maximum<float>());
    const float dd = amax / 127.0f;
    const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
    q[lane * 2] = (int8_t) sycl::round(a * id);
    q[lane * 2 + 1] = (int8_t) sycl::round(b * id);
    if (lane == 0) *d = (float) half(dd);
}

inline int8_t* xqq(void* xq) { return (int8_t*) xq; }
// the scales after the quants (ns_ink2s_xq_scales_at, here for the device)
inline float* xqd(void* xq, int64_t T, int64_t n) { return (float*) ((uint8_t*) xq + align256((size_t) (T * n))); }

constexpr int MIXW = 1024, MAXS = 8, MAXQ = 4;

// a work-group's sums of NV values (each thread's partials): sub-group sums, then a thread a value over the sub-groups
template <int NV>
inline void group_sums(sycl::nd_item<1> it, const float* part, float* slm, float* out) {
    auto sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
    const int nsg = (int) (it.get_local_range(0) / SG);
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        const float s = sycl::reduce_over_group(sg, part[v], sycl::plus<float>());
        if (lane == 0) slm[sgi * NV + v] = s;
    }
    sycl::group_barrier(it.get_group());
    const int tid = (int) it.get_local_id(0);
    if (tid < NV) {
        float s = 0.0f;
        for (int i = 0; i < nsg; ++i) s += slm[i * NV + tid];
        slm[nsg * NV + tid] = s;
    }
    sycl::group_barrier(it.get_group());
#pragma unroll
    for (int v = 0; v < NV; ++v) out[v] = slm[nsg * NV + v];
    sycl::group_barrier(it.get_group());
}

struct MixArgs {
    const float* st[MAXS];
    int ns, nq;
    const float* query;
    const float* nw[MAXQ];   // a query's norm weight (null: the mix unnormed)
    float* out;              // [nq, T, C]
    void* xq[MAXQ];          // a normed query's quantized rows (nullable)
    int64_t T, C;
    float eps_mix, eps_norm;
};

struct Ptr8 {
    const float* p[8];
};

// the depth attention's kernel for S states and Q queries (the sums and loops sized to them), W threads a row
template <int S, int Q, int W>
void mix_launch(sycl::queue& q, MixArgs a, int64_t T) {
    q.submit([&](sycl::handler& h) {
        constexpr int NV = S + Q * S;
        sycl::local_accessor<float, 1> slm(sycl::range<1>(W / SG * NV + NV + 64), h);
        sycl::local_accessor<float, 1> rowb(sycl::range<1>((size_t) Q * 2304), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t t = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const int64_t C_ = a.C, c4 = C_ / 4;
            float* sl = slm.get_multi_ptr<sycl::access::decorated::no>().get();
            float* rb = rowb.get_multi_ptr<sycl::access::decorated::no>().get();
            float part[NV];
#pragma unroll
            for (int v = 0; v < NV; ++v) part[v] = 0.0f;
            for (int64_t i = tid; i < c4; i += W) {
                sycl::float4 qv[Q];
#pragma unroll
                for (int qi = 0; qi < Q; ++qi)
                    if (qi < a.nq) qv[qi] = ((const sycl::float4*) (a.query + qi * C_))[i];
#pragma unroll
                for (int s = 0; s < S; ++s) {
                    if (s >= a.ns) break;
                    const sycl::float4 x = ((const sycl::float4*) (a.st[s] + t * C_))[i];
                    part[s] += x.x() * x.x() + x.y() * x.y() + x.z() * x.z() + x.w() * x.w();
#pragma unroll
                    for (int qi = 0; qi < Q; ++qi)
                        if (qi < a.nq) part[S + qi * S + s] += x.x() * qv[qi].x() + x.y() * qv[qi].y() + x.z() * qv[qi].z() + x.w() * qv[qi].w();
                }
            }
            float tot[NV];
            group_sums<NV>(it, part, sl, tot);
            float inv[S], pr[Q][S];
#pragma unroll
            for (int s = 0; s < S; ++s) inv[s] = s < a.ns ? sycl::rsqrt(tot[s] / (float) C_ + a.eps_mix) : 0.0f;
            const float sc = 1.0f / sycl::sqrt((float) C_);
#pragma unroll
            for (int qi = 0; qi < Q; ++qi) {
                if (qi >= a.nq) break;
                float mx = -INFINITY;
#pragma unroll
                for (int s = 0; s < S; ++s)
                    if (s < a.ns) {
                        pr[qi][s] = tot[S + qi * S + s] * inv[s] * sc;
                        mx = sycl::fmax(mx, pr[qi][s]);
                    }
                float sum = 0.0f;
#pragma unroll
                for (int s = 0; s < S; ++s)
                    if (s < a.ns) {
                        pr[qi][s] = sycl::exp(pr[qi][s] - mx);
                        sum += pr[qi][s];
                    }
#pragma unroll
                for (int s = 0; s < S; ++s)
                    if (s < a.ns) pr[qi][s] = pr[qi][s] / sum * inv[s];
            }
            // every query's mix, and its sum of squares
            float ssq[Q];
#pragma unroll
            for (int qi = 0; qi < Q; ++qi) ssq[qi] = 0.0f;
            for (int64_t i = tid; i < c4; i += W) {
                sycl::float4 y[Q];
#pragma unroll
                for (int qi = 0; qi < Q; ++qi) y[qi] = sycl::float4(0.0f);
#pragma unroll
                for (int s = 0; s < S; ++s) {
                    if (s >= a.ns) break;
                    const sycl::float4 x = ((const sycl::float4*) (a.st[s] + t * C_))[i];
#pragma unroll
                    for (int qi = 0; qi < Q; ++qi)
                        if (qi < a.nq) y[qi] += pr[qi][s] * x;
                }
#pragma unroll
                for (int qi = 0; qi < Q; ++qi)
                    if (qi < a.nq) {
                        ((sycl::float4*) (rb + qi * C_))[i] = y[qi];
                        ssq[qi] += y[qi].x() * y[qi].x() + y[qi].y() * y[qi].y() + y[qi].z() * y[qi].z() + y[qi].w() * y[qi].w();
                    }
            }
            float tq[Q];
            group_sums<Q>(it, ssq, sl, tq);
            for (int64_t i = tid; i < c4; i += W) {
#pragma unroll
                for (int qi = 0; qi < Q; ++qi) {
                    if (qi >= a.nq) break;
                    sycl::float4 y = ((const sycl::float4*) (rb + qi * C_))[i];
                    if (a.nw[qi]) {
                        const float ninv = sycl::rsqrt(tq[qi] / (float) C_ + a.eps_norm);
                        y = y * ninv * ((const sycl::float4*) a.nw[qi])[i];
                        ((sycl::float4*) (rb + qi * C_))[i] = y;
                    }
                    ((sycl::float4*) (a.out + (qi * a.T + t) * C_))[i] = y;
                }
            }
            sycl::group_barrier(it.get_group());
            // the normed rows quantized: a sub-group a block of 32 in turn, over the queries
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int64_t nb = C_ / 32;
            for (int64_t bq = sgi; bq < a.nq * nb; bq += W / SG) {
                const int qi = (int) (bq / nb);
                const int64_t b = bq % nb;
                if (!a.nw[qi] || !a.xq[qi]) continue;
                quant_block(sg, lane, rb + qi * C_ + b * 32, xqq(a.xq[qi]) + t * C_ + b * 32, xqd(a.xq[qi], a.T, C_) + t * nb + b);
            }
        });
    });
}

}  // namespace

extern "C" {

int ns_ink2s_mix_norm(ns_gpu* g, const float* const* states, int64_t ns, const float* query, int64_t nq, const float* const* norm_w, float* out,
                      void* const* xq, int64_t T, int64_t C, float eps_mix, float eps_norm) {
    NS_TRY
    if (ns < 1 || ns > MAXS || nq < 1 || nq > MAXQ || C > 2304 || C % 128) return ns_fail("ns_ink2s_mix_norm: 1..8 states, 1..4 queries, rows of a multiple of 128 to 2304");
    MixArgs a{};
    for (int i = 0; i < ns; ++i) a.st[i] = states[i];
    for (int i = 0; i < nq; ++i) {
        a.nw[i] = norm_w ? norm_w[i] : nullptr;
        a.xq[i] = xq ? xq[i] : nullptr;
    }
    a.ns = (int) ns;
    a.nq = (int) nq;
    a.query = query;
    a.out = out;
    a.T = T;
    a.C = C;
    a.eps_mix = eps_mix;
    a.eps_norm = eps_norm;
    if (T <= 0) return 0;
    // a work-group a row: one pass over the channels (16-byte loads) for the states' sums of squares and the (query,
    // state) dots; every query's mix in a second, with its sum of squares; then each normed, written and quantized
    // the kernel sized to the call: states and queries as template arguments, 576 threads (C / 4 for 2304)
    switch (ns * 8 + nq) {
#define MIXCASE(S_, Q_) case S_ * 8 + Q_: mix_launch<S_, Q_, 576>(g->q, a, T); break;
        MIXCASE(1, 1) MIXCASE(2, 1) MIXCASE(3, 1) MIXCASE(4, 1) MIXCASE(5, 1) MIXCASE(6, 1) MIXCASE(7, 1) MIXCASE(8, 1)
        MIXCASE(1, 3) MIXCASE(2, 3) MIXCASE(3, 3) MIXCASE(4, 3) MIXCASE(5, 3) MIXCASE(6, 3) MIXCASE(7, 3) MIXCASE(8, 3)
        MIXCASE(1, 4) MIXCASE(2, 4) MIXCASE(3, 4) MIXCASE(4, 4) MIXCASE(5, 4) MIXCASE(6, 4) MIXCASE(7, 4) MIXCASE(8, 4)
#undef MIXCASE
        default: mix_launch<MAXS, MAXQ, MIXW>(g->q, a, T); break;
    }
    return 0;
    NS_CATCH
}

int ns_ink2s_conv_res(ns_gpu* g, const float* x, float* state, const float* w, float* snap, const float* base, float* dst, float* out, int64_t T, int64_t C,
                      float inv) {
    NS_TRY
    if (T < 1 || T > 16) return ns_fail("ns_ink2s_conv_res: 1..16 rows");
    // a work-item a channel, its rows in order: the state read, then replaced, by its own work-item only
    g->q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = id[0];
        float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
        const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
        if (snap) {
            snap[c] = h0;
            snap[C + c] = h1;
            snap[2 * C + c] = h2;
        }
        for (int64_t t = 0; t < T; ++t) {
            const float xc = x[t * C + c];
            if (snap) snap[(3 + t) * C + c] = xc;
            float s = 0.0f;
            s += h0 * w0;
            s += h1 * w1;
            s += h2 * w2;
            s += xc * w3;
            const float o = (xc + s) * inv;
            if (out) out[t * C + c] = o;
            dst[t * C + c] = base ? base[t * C + c] + o : o;
            h0 = h1;
            h1 = h2;
            h2 = xc;
        }
        state[c] = h0;
        state[C + c] = h1;
        state[2 * C + c] = h2;
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_post_qkv(ns_gpu* g, const Ink2sQkv* pp) {
    NS_TRY
    const Ink2sQkv p = *pp;
    if (p.T < 1 || p.T > 16 || p.D != 128 || p.R > SG) return ns_fail("ns_ink2s_post_qkv: 1..16 rows, heads of 128, a bias of at most 16");
    // work-groups: Hkv for k, Hkv for v (a work-item a channel, its rows in order), Hq for q, Hq for r
    constexpr int W = 128;
    const int64_t groups = 2 * p.Hkv + 2 * p.Hq;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sm(sycl::range<1>(16 * 128 + 16 * 4 + 64), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t gi = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            auto grp = it.get_group();
            float* rows = sm.get_multi_ptr<sycl::access::decorated::no>().get();
            const int64_t T = p.T, KD = p.Hkv * 128;
            auto tau = [&](int64_t t) {
                return p.alpha > 0.0f ? p.alpha * sycl::log(sycl::fmax((float) (p.p0 + t + 1) * (1.0f / p.floor), 1.0f)) + 1.0f : 1.0f;
            };
            // a BF16 row of C against a float row: the work-group's sum
            auto bdot = [&](const uint16_t* wr, const float* xr) {
                float acc = 0.0f;
                for (int64_t i = tid; i < p.C; i += W) acc += sycl::bit_cast<float>((uint32_t) wr[i] << 16) * xr[i];
                return sycl::reduce_over_group(grp, acc, sycl::plus<float>());
            };
            if (gi < 2 * p.Hkv) {
                const bool isv = gi >= p.Hkv;
                const int64_t hh = isv ? gi - p.Hkv : gi, c = hh * 128 + tid;
                const float* x = isv ? p.vraw : p.kraw;
                const float* dyn = isv ? p.vdyn : p.kdyn;
                // the head's input-dependent taps, made here: [T][4] in local memory (after the rows' space)
                const uint16_t* dw = isv ? p.vdw : p.kdw;
                float* dloc = rows + 16 * 128;
                if (dw) {
                    const float* nx = isv ? p.n2 : p.n1;
                    for (int64_t t = 0; t < T; ++t)
                        for (int l = 0; l < 4; ++l) {
                            const float v = bdot(dw + (hh * 4 + l) * p.C, nx + t * p.C);
                            if (tid == 0) dloc[t * 4 + l] = v;
                        }
                    sycl::group_barrier(grp);
                    dyn = dloc;
                }
                const int64_t dstride = dw ? 4 : p.Hkv * 4, doff = dw ? 0 : hh * 4;
                float* st = isv ? p.vstate : p.kstate;
                float* sn = isv ? p.vsnap : p.ksnap;
                const float* w = isv ? p.vw : p.kw;
                float h0 = st[c], h1 = st[KD + c], h2 = st[2 * KD + c];
                if (sn) {
                    sn[c] = h0;
                    sn[KD + c] = h1;
                    sn[2 * KD + c] = h2;
                }
                for (int64_t t = 0; t < T; ++t) {
                    const float xc = x[t * KD + c];
                    if (sn) sn[(3 + t) * KD + c] = xc;
                    float s = 0.0f;
                    s += h0 * w[c * 4];
                    s += h1 * w[c * 4 + 1];
                    s += h2 * w[c * 4 + 2];
                    s += xc * w[c * 4 + 3];
                    if (dyn) {
                        const float* dt = dyn + t * dstride + doff;
                        s += dt[0] * xc;
                        s += dt[1] * h2;
                        s += dt[2] * h1;
                        s += dt[3] * h0;
                    }
                    rows[t * 128 + tid] = (xc + s) * (dyn ? p.inv_dyn : p.inv);
                    h0 = h1;
                    h1 = h2;
                    h2 = xc;
                }
                st[c] = h0;
                st[KD + c] = h1;
                st[2 * KD + c] = h2;
                sycl::group_barrier(grp);
                // each row: k clipped to an L2 norm of at most thr, v RMS-normed; into the caches (and the float rows)
                for (int64_t t = 0; t < T; ++t) {
                    const float v = rows[t * 128 + tid];
                    const float ss = sycl::reduce_over_group(grp, v * v, sycl::plus<float>());
                    float y;
                    if (isv) {
                        y = v * sycl::rsqrt(ss / 128.0f + p.eps);
                    } else {
                        const float f = sycl::clamp(p.thr / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
                        y = v * f;
                    }
                    (isv ? p.vout : p.kout)[t * KD + c] = y;
                    const int64_t slot = (p.p0 + t) % p.cap;
                    half* cache = (half*) (isv ? p.vc : p.kc);
                    cache[(hh * p.cap + slot) * 128 + tid] = half(y);
                }
                return;
            }
            // q (heads of 128) or r (heads of R), clipped after their scale, the full layers' log scale after
            const bool isr = gi >= 2 * p.Hkv + p.Hq;
            const int64_t hh = gi - 2 * p.Hkv - (isr ? p.Hq : 0);
            const int64_t D = isr ? p.R : 128;
            float* x = isr ? p.r : p.q;
            const float* scl = isr ? p.rscale : p.qscale;
            const float thr = isr ? p.rthr : p.thr;
            if (!isr && p.sinkw) {
                // the row's sink logit for this head
                for (int64_t t = 0; t < T; ++t) {
                    const float v = bdot(p.sinkw + hh * p.C, p.n0 + t * p.C);
                    if (tid == 0) p.sink[t * p.Hq + hh] = v;
                }
            }
            for (int64_t t = 0; t < T; ++t) {
                float* xr = x + (t * p.Hq + hh) * D;
                const float v = tid < D ? xr[tid] * scl[hh * D + tid] : 0.0f;
                const float ss = sycl::reduce_over_group(grp, v * v, sycl::plus<float>());
                const float f = sycl::clamp(thr / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
                if (tid < D) xr[tid] = p.alpha > 0.0f ? (v * f) * tau(t) : v * f;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_merge_gate_xq(ns_gpu* g, const float* part, const float* sink, const float* gate, float* out, void* xq, int64_t T, int64_t Hq, int64_t ns) {
    NS_TRY
    constexpr int D = 128;
    if (T <= 0) return 0;
    const int64_t n = Hq * D;
    // a work-group a (row, head): its splits merged in order with the sink, gated, the head's four blocks quantized
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> vals(sycl::range<1>(D), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * Hq) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t rh = it.get_group(0), t = rh / Hq, hq = rh % Hq;
            const int dd = (int) it.get_local_id(0);
            const float* pp = part + rh * ns * (D + 2);
            const float sk = sink ? sink[rh] : -INFINITY;
            float M = sk;
            for (int64_t i = 0; i < ns; ++i) M = sycl::fmax(M, pp[i * (D + 2) + D]);
            float S = sink ? sycl::exp(sk - M) : 0.0f, A = 0.0f;
            for (int64_t i = 0; i < ns; ++i) {
                const float c = sycl::exp(pp[i * (D + 2) + D] - M);
                S += pp[i * (D + 2) + D + 1] * c;
                A += pp[i * (D + 2) + dd] * c;
            }
            const int64_t at = t * n + hq * D + dd;
            const float y = gate ? A / S * (1.0f / (1.0f + sycl::exp(-gate[at]))) : A / S;
            out[at] = y;
            vals[dd] = y;
            sycl::group_barrier(it.get_group());
            auto sg = it.get_sub_group();
            const int sgi = (int) sg.get_group_linear_id();
            if (sgi < D / 32) {
                const float* vp = vals.get_multi_ptr<sycl::access::decorated::no>().get();
                quant_block(sg, (int) sg.get_local_linear_id(), vp + sgi * 32, xqq(xq) + t * n + hq * D + sgi * 32, xqd(xq, T, n) + t * (n / 32) + hq * 4 + sgi);
            }
        });
    });
    return 0;
    NS_CATCH
}

// up to 4 controllers in one launch (the three attention inputs of a looped layer)
struct CtrlJobs {
    const float* cur[4];
    const float* msg[4];
    const float* down[4];
    const float* gain[4];
    const float* forget[4];
    const float* u[4];
    float* uo[4];
    float* out[4];
    int n;
};

int ns_ink2s_ctrl_down_many(ns_gpu* g, int n, const float* const* cur, const float* const* msg, const float* const* down, float* const* u, int64_t T,
                            int64_t C, int64_t rank, float eps) {
    NS_TRY
    constexpr int W = 256, RPW = 2;
    if (T <= 0) return 0;
    if (n < 1 || n > 4 || C % 4 || 2 * C > 8192) return ns_fail("ns_ink2s_ctrl_down_many: 1..4 controllers, rows of a multiple of 4, at most 4096");
    CtrlJobs jb{};
    for (int k = 0; k < n; ++k) {
        jb.cur[k] = cur[k];
        jb.msg[k] = msg[k];
        jb.down[k] = down[k];
        jb.uo[k] = u[k];
    }
    jb.n = n;
    const int64_t blocks = (rank + RPW - 1) / RPW;
    // a work-group a (controller, row, 2 of the rank's rows): z = [rms(cur) | rms(msg)] into local memory once, then each
    // rank row's dot over all the work-group's threads (16-byte loads), the sums added in a fixed order
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> z(sycl::range<1>(8192), h);
        sycl::local_accessor<float, 1> sl(sycl::range<1>(W / SG * RPW + 8), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * T * blocks) * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t r0 = (gi % blocks) * RPW;
            gi /= blocks;
            const int64_t t = gi % T;
            const int k = (int) (gi / T);
            const float* down = jb.down[k];
            const int tid = (int) it.get_local_id(0);
            const sycl::float4* a = (const sycl::float4*) (jb.cur[k] + t * C);
            const sycl::float4* b = (const sycl::float4*) (jb.msg[k] + t * C);
            float* zp = z.get_multi_ptr<sycl::access::decorated::no>().get();
            float* s = sl.get_multi_ptr<sycl::access::decorated::no>().get();
            const int64_t c4 = C / 4;
            float pa = 0.0f, pb = 0.0f;
            for (int64_t i = tid; i < c4; i += W) {
                const sycl::float4 va = a[i], vb = b[i];
                pa += va.x() * va.x() + va.y() * va.y() + va.z() * va.z() + va.w() * va.w();
                pb += vb.x() * vb.x() + vb.y() * vb.y() + vb.z() * vb.z() + vb.w() * vb.w();
                ((sycl::float4*) zp)[i] = va;
                ((sycl::float4*) (zp + C))[i] = vb;
            }
            const float ia = sycl::rsqrt(sycl::reduce_over_group(grp, pa, sycl::plus<float>()) / (float) C + eps);
            const float ib = sycl::rsqrt(sycl::reduce_over_group(grp, pb, sycl::plus<float>()) / (float) C + eps);
            sycl::group_barrier(grp);
            float acc[RPW];
#pragma unroll
            for (int r = 0; r < RPW; ++r) acc[r] = 0.0f;
            const int64_t n4 = 2 * C / 4;
            for (int64_t i = tid; i < n4; i += W) {
                const sycl::float4 zv = ((const sycl::float4*) zp)[i];
                const float f = i < c4 ? ia : ib;
#pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    if (r0 + r >= rank) break;
                    const sycl::float4 dv = ((const sycl::float4*) (down + (r0 + r) * 2 * C))[i];
                    acc[r] += (dv.x() * zv.x() + dv.y() * zv.y() + dv.z() * zv.z() + dv.w() * zv.w()) * f;
                }
            }
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const float v = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                if (lane == 0) s[sgi * RPW + r] = v;
            }
            sycl::group_barrier(grp);
            if (tid < RPW && r0 + tid < rank) {
                float v = 0.0f;
                for (int i = 0; i < W / SG; ++i) v += s[i * RPW + tid];
                jb.uo[k][t * rank + r0 + tid] = v;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_ctrl_down(ns_gpu* g, const float* cur, const float* msg, const float* down, float* u, int64_t T, int64_t C, int64_t rank, float eps) {
    return ns_ink2s_ctrl_down_many(g, 1, &cur, &msg, &down, &u, T, C, rank, eps);
}

int ns_ink2s_ctrl_apply_many(ns_gpu* g, int n, const float* const* cur, const float* const* msg, const float* const* u, const float* const* gain,
                             const float* const* forget, float* const* out, int64_t T, int64_t C, int64_t rank) {
    NS_TRY
    constexpr int W = 128, CPW = W / SG * 4;
    if (T <= 0) return 0;
    if (n < 1 || n > 4 || rank > 64 || rank % SG) return ns_fail("ns_ink2s_ctrl_apply_many: 1..4 controllers, a rank of at most 64, a multiple of 16");
    CtrlJobs jb{};
    for (int k = 0; k < n; ++k) {
        jb.cur[k] = cur[k];
        jb.msg[k] = msg[k];
        jb.u[k] = u[k];
        jb.gain[k] = gain[k];
        jb.forget[k] = forget[k];
        jb.out[k] = out[k];
    }
    jb.n = n;
    const int64_t blocks = (C + CPW - 1) / CPW;
    // a sub-group 4 channels: their gain and forget rows against u (the lanes over the rank)
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * T * blocks) * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
        int64_t gi = it.get_group(0);
        const int64_t c0 = (gi % blocks) * CPW + sgi * 4;
        gi /= blocks;
        const int64_t t = gi % T;
        const int k = (int) (gi / T);
        const float* u = jb.u[k];
        const float* gain = jb.gain[k];
        const float* forget = jb.forget[k];
        float uv[4];
        for (int i = 0; i < 4; ++i) uv[i] = lane + i * SG < rank ? u[t * rank + lane + i * SG] : 0.0f;
        for (int kk = 0; kk < 4; ++kk) {
            const int64_t c = c0 + kk;
            if (c >= C) break;
            float gs = 0.0f, fs = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const int64_t j = lane + i * SG;
                if (j < rank) {
                    gs = sycl::fma(gain[c * rank + j], uv[i], gs);
                    fs = sycl::fma(forget[c * rank + j], uv[i], fs);
                }
            }
            gs = sycl::reduce_over_group(sg, gs, sycl::plus<float>());
            fs = sycl::reduce_over_group(sg, fs, sycl::plus<float>());
            if (lane == 0) {
                const float x = jb.cur[k][t * C + c];
                jb.out[k][t * C + c] = (x - sycl::tanh(fs) * x) + sycl::tanh(gs) * jb.msg[k][t * C + c];
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_ctrl_apply(ns_gpu* g, const float* cur, const float* msg, const float* u, const float* gain, const float* forget, float* out, int64_t T, int64_t C,
                        int64_t rank) {
    return ns_ink2s_ctrl_apply_many(g, 1, &cur, &msg, &u, &gain, &forget, &out, T, C, rank);
}

int ns_ink2s_router(ns_gpu* g, const float* x, const float* w, const float* bias, float* logits, uint32_t* ctr, int32_t* ids, float* wts, int64_t T, int64_t E,
                    int64_t C, int64_t K, float scale) {
    NS_TRY
    constexpr int EPW = 8;
    if (T <= 0) return 0;
    if (E > EPW * SG || E % EPW || K > 16 || C % 4) return ns_fail("ns_ink2s_router: at most 128 experts (a multiple of 8), 16 chosen, rows of a multiple of 4");
    const int64_t nwg = E / EPW;
    // a work-group a (row, 8 experts): a sub-group an expert's logit (lanes over the row, 16-byte loads, a fixed order);
    // the row's last work-group to finish (a counter a row, reset by it) makes the choice as ns_ink2_route does
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> last(sycl::range<1>(1), h);
        sycl::local_accessor<float, 1> sv(sycl::range<1>(2 * EPW * SG + 16), h);
        sycl::local_accessor<int, 1> si(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * nwg) * EPW * SG), sycl::range<1>(EPW * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int64_t t = it.get_group(0) / nwg, e = (it.get_group(0) % nwg) * EPW + sgi;
            const sycl::float4* xr = (const sycl::float4*) (x + t * C);
            const sycl::float4* wr = (const sycl::float4*) (w + e * C);
            float acc = 0.0f;
            for (int64_t i = lane; i < C / 4; i += SG) {
                const sycl::float4 a = xr[i], b = wr[i];
                acc = sycl::fma(a.x(), b.x(), acc);
                acc = sycl::fma(a.y(), b.y(), acc);
                acc = sycl::fma(a.z(), b.z(), acc);
                acc = sycl::fma(a.w(), b.w(), acc);
            }
            const float lg = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) logits[t * E + e] = lg;
            sycl::group_barrier(it.get_group());
            if (it.get_local_id(0) == 0) {
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space> c(ctr[t]);
                const uint32_t n = c.fetch_add(1u);
                last[0] = n + 1 == (uint32_t) nwg;
                if (last[0]) c.store(0u);
            }
            sycl::group_barrier(it.get_group());
            if (!last[0]) return;
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
            // the choice, every expert a work-item: sigmoid, p + bias; an expert's rank the count of those larger (the
            // lower index first among equals); the K first by rank, their weights p / (sum in rank order + 1e-20) x scale
            const int e2 = (int) it.get_local_id(0);
            float* vv = sv.get_multi_ptr<sycl::access::decorated::no>().get();
            float* pp = sv.get_multi_ptr<sycl::access::decorated::no>().get() + EPW * SG;
            float* selp = pp + EPW * SG;
            int* seli = si.get_multi_ptr<sycl::access::decorated::no>().get();
            float pe = 0.0f, ve = -INFINITY;
            if (e2 < E) {
                pe = 1.0f / (1.0f + sycl::exp(-logits[t * E + e2]));
                ve = bias ? pe + bias[e2] : pe;
            }
            vv[e2] = ve;
            pp[e2] = pe;
            sycl::group_barrier(it.get_group());
            if (e2 < E) {
                int rank = 0;
                for (int j = 0; j < E; ++j) rank += vv[j] > ve || (vv[j] == ve && j < e2);
                if (rank < K) {
                    seli[rank] = e2;
                    selp[rank] = pe;
                }
            }
            sycl::group_barrier(it.get_group());
            if (e2 < K) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) sum += selp[k];
                sum += 1e-20f;
                ids[t * K + e2] = seli[e2];
                wts[t * K + e2] = selp[e2] / sum * scale;
            }
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
