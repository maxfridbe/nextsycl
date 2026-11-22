// draft.cpp: the inkling2 silo's DSpark drafter steps (silo.h) - Inkling 2 Mini's 3-layer block drafter (its
// mtp.safetensors; SGLang's models/inkling2_mtp.py, models/dspark.py, kernels/ops/speculative/inkling2_mtp.py): the
// target's features kept a row, their per-layer mixes normed, the block's SwiGLU quantized, the attention over the
// context and the whole block (not causal), and the Markov head's recurrent step and its bias on the logits.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;

inline size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// float32 -> BF16 bits, round to nearest even (torch's .to(bfloat16))
inline uint16_t to_bf16(float x) {
    uint32_t u = sycl::bit_cast<uint32_t>(x);
    if ((u & 0x7F800000u) == 0x7F800000u) return (uint16_t) (u >> 16);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
inline float bf16f(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }
inline float rbf(float x) { return bf16f(to_bf16(x)); }

inline void quant_block(sycl::sub_group sg, int lane, const float* x, int8_t* q, float* d) {
    const float a = x[lane * 2], b = x[lane * 2 + 1];
    const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(a), sycl::fabs(b)), sycl::maximum<float>());
    const float dd = amax / 127.0f;
    const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
    q[lane * 2] = (int8_t) sycl::round(a * id);
    q[lane * 2 + 1] = (int8_t) sycl::round(b * id);
    if (lane == 0) *d = (float) half(dd);
}
}  // namespace

extern "C" {

int ns_ink2s_feat_put(ns_gpu* g, const float* src, float* feats, int64_t T, int64_t C, int64_t nf, int64_t k) {
    NS_TRY
    if (T <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t t = id[0] / C, c = id[0] % C;
        feats[(t * nf + k) * C + c] = rbf(src[t * C + c]);
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_feat_norm(ns_gpu* g, const float* feats, int64_t nf, const float* alpha, const float* w, float eps, float* y, void* xq, int64_t T, int64_t C) {
    NS_TRY
    constexpr int W = 256;
    if (T <= 0) return 0;
    if (C % 32 || C > 4096 || nf > 8) return ns_fail("ns_ink2s_feat_norm: rows of whole blocks of 32, at most 4096; at most 8 features");
    int8_t* q = (int8_t*) xq;
    float* d = xq ? (float*) ((uint8_t*) xq + align256((size_t) (T * C))) : nullptr;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> row(sycl::range<1>(4096), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t t = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            float* rp = row.get_multi_ptr<sycl::access::decorated::no>().get();
            float ss = 0.0f;
            for (int64_t c = tid; c < C; c += W) {
                float m = 0.0f;
                for (int64_t k = 0; k < nf; ++k) m += feats[(t * nf + k) * C + c] * alpha[k];
                rp[c] = m;
                ss += m * m;
            }
            const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) C + eps);
            for (int64_t c = tid; c < C; c += W) {
                const float v = rp[c] * inv * w[c];
                rp[c] = v;
                if (y) y[t * C + c] = v;
            }
            sycl::group_barrier(it.get_group());
            if (q) {
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
                for (int64_t b = sgi; b < C / 32; b += W / SG) quant_block(sg, lane, rp + b * 32, q + t * C + b * 32, d + t * (C / 32) + b);
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_swiglu_xq(ns_gpu* g, const float* gate, const float* up, void* xq, int64_t T, int64_t F) {
    NS_TRY
    if (T <= 0) return 0;
    if (F % 32) return ns_fail("ns_ink2s_swiglu_xq: rows of whole blocks of 32");
    const int64_t nb = T * (F / 32);
    int8_t* q = (int8_t*) xq;
    float* d = (float*) ((uint8_t*) xq + align256((size_t) (T * F)));
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> v(sycl::range<1>(4 * 32), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((nb + 3) / 4) * 4 * SG), sycl::range<1>(4 * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int64_t b = it.get_group(0) * 4 + sgi;
            if (b >= nb) return;
            float* vp = v.get_multi_ptr<sycl::access::decorated::no>().get() + sgi * 32;
            for (int i = 0; i < 2; ++i) {
                const int64_t at = b * 32 + lane * 2 + i;
                const float gv = gate[at];
                vp[lane * 2 + i] = gv / (1.0f + sycl::exp(-gv)) * up[at];
            }
            sycl::group_barrier(sg);
            quant_block(sg, lane, vp, q + b * 32, d + b);
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_to_bf16(ns_gpu* g, const float* x, uint16_t* y, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { y[i] = to_bf16(x[i]); });
    return 0;
    NS_CATCH
}

int ns_ink2s_rnn_inputs(ns_gpu* g, const float* state, const int32_t* tok, const uint16_t* w1, int64_t rank, const float* hidden, const float* f_rnn,
                        int64_t C, float* out) {
    NS_TRY
    // [state | w1[token] | hidden | f_rnn], each rounded to BF16 (the reference's inputs are BF16)
    const int64_t n = 2 * rank + 2 * C;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        float v;
        if (i < rank) v = state[i];
        else if (i < 2 * rank) v = bf16f(w1[(int64_t) tok[0] * rank + (i - rank)]);
        else if (i < 2 * rank + C) v = hidden[i - 2 * rank];
        else v = f_rnn[i - 2 * rank - C];
        out[i] = rbf(v);
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_rnn_gates(ns_gpu* g, const float* proj, const float* bias, float* state, float* latent, int64_t rank) {
    NS_TRY
    // the projection (+ bias) rounded to BF16, then each op rounded as the reference's eager BF16 ops: gate =
    // sigmoid(g), state' = gate state + (1 - gate) tanh(cand), latent = tanh(out)
    g->q.parallel_for(sycl::range<1>((size_t) rank), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const float gr = rbf(proj[i] + bias[i]), cr = rbf(proj[rank + i] + bias[rank + i]), orw = rbf(proj[2 * rank + i] + bias[2 * rank + i]);
        const float gate = rbf(1.0f / (1.0f + sycl::exp(-gr)));
        const float kept = rbf(gate * state[i]);
        const float blend = rbf(rbf(1.0f - gate) * rbf(sycl::tanh(cr)));
        state[i] = rbf(kept + blend);
        latent[i] = rbf(sycl::tanh(orw));
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_bias_argmax(ns_gpu* g, const float* base, const float* bias, const float* fold_b, int64_t n, float cap, float* logits, int32_t* out,
                         float* prob, int32_t* stop, float pmin) {
    NS_TRY
    // the step's logits: softcap(base + BF16(fold_b + bias)), over the first n; the largest (the lowest index of equals)
    // and its probability (softmax at temperature 1: the online max and sum of exponentials, merged in a fixed order)
    constexpr int W = 1024;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> bv(sycl::range<1>(W / SG), h);
        sycl::local_accessor<float, 1> bs(sycl::range<1>(W / SG), h);
        sycl::local_accessor<int, 1> bi(sycl::range<1>(W / SG), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int tid = (int) it.get_local_id(0);
            // a block already ended (an earlier draft below p_min): nothing to do
            if (stop && stop[0]) return;
            float best = -INFINITY, se = 0.0f;
            int at = 0x7FFFFFFF;
            for (int64_t i = tid; i < n; i += W) {
                float v = base[i] + rbf(fold_b[i] + bias[i]);
                if (cap > 0.0f) v = cap * sycl::tanh(v / cap);
                if (logits) logits[i] = v;
                if (v > best) {
                    se = se * sycl::exp(best - v) + 1.0f;
                    best = v;
                    at = (int) i;
                } else {
                    se += sycl::exp(v - best);
                }
            }
            auto sg = it.get_sub_group();
            const float m = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
            const int a = sycl::reduce_over_group(sg, best == m ? at : 0x7FFFFFFF, sycl::minimum<int>());
            const float sm = sycl::reduce_over_group(sg, best > -INFINITY ? se * sycl::exp(best - m) : 0.0f, sycl::plus<float>());
            if (sg.get_local_linear_id() == 0) {
                bv[sg.get_group_linear_id()] = m;
                bs[sg.get_group_linear_id()] = sm;
                bi[sg.get_group_linear_id()] = a;
            }
            sycl::group_barrier(it.get_group());
            if (tid == 0) {
                float m2 = -INFINITY;
                int a2 = 0x7FFFFFFF;
                for (int s = 0; s < W / SG; ++s)
                    if (bv[s] > m2 || (bv[s] == m2 && bi[s] < a2)) {
                        m2 = bv[s];
                        a2 = bi[s];
                    }
                float st = 0.0f;
                for (int s = 0; s < W / SG; ++s)
                    if (bv[s] > -INFINITY) st += bs[s] * sycl::exp(bv[s] - m2);
                out[0] = a2;
                if (prob) prob[0] = 1.0f / st;
                if (stop && 1.0f / st < pmin) stop[0] = 1;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_qk_norm_rope(ns_gpu* g, float* x, int64_t T, int64_t H, const float* w, float eps, float theta, int64_t p0) {
    NS_TRY
    constexpr int D = 128;
    if (T <= 0) return 0;
    // a sub-group a (row, head): RMS-normed with w [D] (null: not normed), then NEOX-rotated (pairs i, i + D / 2) at
    // position p0 + t, angle pos theta^(-2i / D)
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t rh = it.get_group(0), t = rh / H;
        float* xr = x + rh * D;
        float v[D / SG];
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < D / SG; ++i) {
            v[i] = xr[lane + i * SG];
            ss += v[i] * v[i];
        }
        const float inv = w ? sycl::rsqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) D + eps) : 1.0f;
#pragma unroll
        for (int i = 0; i < D / SG; ++i) v[i] = w ? v[i] * inv * w[lane + i * SG] : v[i];
        const float pos = (float) (p0 + t);
        // lane + i * SG for i < D / SG / 2 is the first half (dims 0..63), its pair i + D / SG / 2
#pragma unroll
        for (int i = 0; i < D / SG / 2; ++i) {
            const int d = lane + i * SG;
            const float ang = pos * sycl::pow(theta, -2.0f * (float) d / (float) D);
            const float c = sycl::cos(ang), s = sycl::sin(ang);
            const float a = v[i], b = v[i + D / SG / 2];
            v[i] = a * c - b * s;
            v[i + D / SG / 2] = b * c + a * s;
        }
#pragma unroll
        for (int i = 0; i < D / SG; ++i) xr[lane + i * SG] = v[i];
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_fold_mv(ns_gpu* g, const uint16_t* w, const float* x, float* y, int64_t rows, int64_t n, const int32_t* stop) {
    NS_TRY
    // y [rows] = w [rows, n] (BF16) . x [n]: a sub-group a row, its lanes over the row in a fixed order; nothing when the
    // block already ended (stop[0])
    constexpr int RPW = 16;
    if (rows <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((rows + RPW - 1) / RPW) * RPW * SG), sycl::range<1>(RPW * SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        if (stop && stop[0]) return;
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t r = it.get_group(0) * RPW + sg.get_group_linear_id();
        if (r >= rows) return;
        const uint16_t* wr = w + r * n;
        float acc = 0.0f;
        for (int64_t i = lane * 8; i < n; i += SG * 8) {
            const sycl::vec<uint16_t, 8> v = *(const sycl::vec<uint16_t, 8>*) (wr + i);
#pragma unroll
            for (int k = 0; k < 8; ++k) acc = sycl::fma(bf16f(v[k]), x[i + k], acc);
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) y[r] = acc;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
