// z3.cpp: the ZONOS3-Mini engine's own kernels (audio/zonos3, z3.h) - what the shared ones do not have: the backbone's
// embeddings, Full AttnRes, the gated DeltaNet's convolution and recurrence, the latent MoE's Q4_0 experts read from
// their blocks, a small-head attention (causal, windowed or full), the codec decoder's table lookups, causal ConvNeXt
// convolution, patch upsampling, iSTFT spectrum, overlap-add, FIR mixer and resampler, the speaker encoder's grouped
// convolutions and pooling, the reference encoder's Snake, GLU and relative-position attention (with the reference's
// BF16 rounding), FuseAlign's full attention, table RoPE and col2im, LACI's windowed attention, and the speech-edit
// LoRA on the routed experts.
//
// Origin: the arithmetic of Zyphra's ZONOS3 reference runtime (github.com/Zyphra/ZONOS3-Mini: zonos3.reference.lm,
// servers/ggml/src/zonos3-graph.cpp, zac.cpp, spk-encoder.cpp, ssl-encoder.cpp and the MLX port; MIT licence, the model
// Apache-2.0) and of FuseAlign (Zyphra/fusealign, Apache-2.0), written anew for SYCL; the shapes and launch plan follow
// our sister project nextcuda's port (kernels/audio/zonos3/z3.cu there). The gated delta rule's per-step order (decay,
// read, correct, write, read out) is the one of the reference's kernel.
//
// Precise arithmetic: the audio kinds build with icpx's default floating-point model, so this file asks for precise
// semantics itself (no reassociation, no reciprocal approximations) - the backbone is checked against the reference's
// logits and the decoder's PCM to 1e-5.
#pragma clang fp reassociate(off)
#include "ns.h"
#include "ns_internal.hpp"
#include "z3.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {

using half = sycl::half;
// a "warp": the sub-group of 32 the kernels below reduce over (Xe2 runs 16 or 32; 32 keeps the lane plans)
constexpr int WARP = 32;
using nd1 = sycl::nd_item<1>;
using nd2 = sycl::nd_item<2>;

inline size_t rup(int64_t n, int64_t t) { return (size_t) (((n + t - 1) / t) * t); }

inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float siluf_(float x) { return x / (1.0f + sycl::exp(-x)); }

inline float sg_sum(const sycl::sub_group& sg, float v) { return sycl::reduce_over_group(sg, v, sycl::plus<float>()); }

// one item an element: f(i) for i < n
template <class F>
void each(ns_gpu* g, int64_t n, F f) {
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { f((int64_t) id[0]); });
}

// round to bfloat16 (nearest, ties to even); inf and nan kept
inline float bf16r(float x) {
    uint32_t u = sycl::bit_cast<uint32_t>(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return sycl::bit_cast<float>(u & 0xffff0000u);
}

// Q4_0: a half scale and 16 bytes, value j of a block (q[j] & 15) - 8 for j < 16, (q[j - 16] >> 4) - 8 after
constexpr int Q4B = 18;
// 8 consecutive values (j0 a multiple of 8) of a Q4_0 row, dotted with x[0..8)
inline float q4_dot8(const uint8_t* row, int64_t j0, const float* x) {
    const uint8_t* blk = row + (j0 / 32) * Q4B;
    const float d = (float) sycl::bit_cast<half>((uint16_t) (blk[0] | (blk[1] << 8)));
    const int j = (int) (j0 % 32);
    const uint8_t* qs = blk + 2;
    float s = 0.0f;
    if (j < 16) {
#pragma unroll
        for (int i = 0; i < 8; ++i) s += (float) ((int) (qs[j + i] & 15) - 8) * x[i];
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) s += (float) ((int) (qs[j - 16 + i] >> 4) - 8) * x[i];
    }
    return s * d;
}

constexpr int MAXK = 8;                      // gdn_conv's taps at most
constexpr int GS = 128, GJ = GS / WARP, G_NW = 4;   // gdn_scan: heads of 128, 4 value rows a group
constexpr int M_NW = 8;                      // the experts: 8 sub-groups a group, an output each
constexpr int QB = 64;                       // attention: query rows a group (keys a tile: KT, 64; 32 for heads of 128)
constexpr int RD = 64, RQ = 64, RK = 32;     // relative attention: head size, query rows a group, keys a tile
constexpr int WB = 256;                      // windowed attention's group
constexpr int FB = 128;                      // full attention's group

template <int D, int KT = 64>
void attention_launch(ns_gpu* g, const float* q, int64_t qs, int64_t rows, const float* k, const float* v, int64_t kvs, int64_t keys, int Hq, int Hk,
                      float scale, int64_t pos0, int causal, int64_t window, float* out) {
    const size_t nb = (size_t) ((rows + QB - 1) / QB);
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ks(sycl::range<1>(KT * D), h), vs(sycl::range<1>(KT * D), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) Hq, nb * QB), sycl::range<2>(1, QB)), [=](nd2 it) {
            const int hh = (int) it.get_group(0), hk = hh / (Hq / Hk);
            const int tid = (int) it.get_local_id(1);
            const int64_t r0 = (int64_t) it.get_group(1) * QB, r = r0 + tid;
            const bool live = r < rows;
            const int64_t rl = (r0 + QB < rows ? r0 + QB : rows) - 1;
            const int64_t qp = pos0 + r;
            int64_t kend = keys, kbeg = 0;
            if (causal) kend = sycl::min(keys, pos0 + rl + 1);
            if (window > 0) kbeg = sycl::max((int64_t) 0, pos0 + r0 - window + 1);
            float qv[D], acc[D];
            float m = -INFINITY, l = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                qv[d] = live ? q[r * qs + (int64_t) hh * D + d] * scale : 0.0f;
                acc[d] = 0.0f;
            }
            for (int64_t t0 = kbeg; t0 < kend; t0 += KT) {
                sycl::group_barrier(it.get_group());
                for (int e = tid; e < KT * D; e += QB) {
                    const int kk = e / D, d = e % D;
                    const int64_t kp = t0 + kk;
                    ks[kk * D + d] = kp < kend ? k[kp * kvs + (int64_t) hk * D + d] : 0.0f;
                    vs[kk * D + d] = kp < kend ? v[kp * kvs + (int64_t) hk * D + d] : 0.0f;
                }
                sycl::group_barrier(it.get_group());
                if (!live) continue;
                const int n = (int) sycl::min((int64_t) KT, kend - t0);
                for (int kk = 0; kk < n; ++kk) {
                    const int64_t kp = t0 + kk;
                    if (causal && kp > qp) break;
                    if (window > 0 && qp - kp >= window) continue;
                    float s = 0.0f;
#pragma unroll
                    for (int d = 0; d < D; ++d) s += qv[d] * ks[kk * D + d];
                    if (s > m) {
                        const float c = sycl::exp(m - s);
                        l *= c;
#pragma unroll
                        for (int d = 0; d < D; ++d) acc[d] *= c;
                        m = s;
                    }
                    const float p = sycl::exp(s - m);
                    l += p;
#pragma unroll
                    for (int d = 0; d < D; ++d) acc[d] += p * vs[kk * D + d];
                }
            }
            if (!live) return;
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) out[r * (int64_t) Hq * D + (int64_t) hh * D + d] = acc[d] * inv;
        });
    });
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------------------------------- backbone

int ns_audio_z3_embed(ns_gpu* g, const void* audio, int64_t ncb, int64_t va, const void* text, int64_t vt, const int32_t* ids, int64_t rows, int64_t D,
                      float* out) {
    NS_TRY
    if (rows * D <= 0) return 0;
    const half* a = (const half*) audio;
    const half* tx = (const half*) text;
    const int nc = (int) ncb;
    each(g, rows * D, [=](int64_t i) {
        const int64_t r = i / D, d = i % D;
        const int32_t* id = ids + r * (nc + 1);
        float s = 0.0f;
        for (int c = 0; c < nc; ++c) {
            int64_t t = id[c];
            t = t < 0 ? 0 : (t >= va ? va - 1 : t);
            s += (float) a[((int64_t) c * va + t) * D + d];
        }
        int64_t t = id[nc];
        t = t < 0 ? 0 : (t >= vt ? vt - 1 : t);
        out[i] = s + (float) tx[t * D + d];
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_rms(ns_gpu* g, const float* x, int64_t xs, int64_t rows, int64_t C, const float* w, float eps, float* out) {
    NS_TRY
    if (rows * C <= 0) return 0;
    constexpr int T = 256;
    g->q.parallel_for(sycl::nd_range<1>((size_t) rows * T, T), [=](nd1 it) {
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const float* p = x + r * xs;
        float ss = 0.0f;
        for (int64_t c = tid; c < C; c += T) ss += p[c] * p[c];
        const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) C + eps);
        for (int64_t c = tid; c < C; c += T) out[r * C + c] = w ? p[c] * inv * w[c] : p[c] * inv;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_attnres_score(ns_gpu* g, const float* src, int64_t rows, int64_t D, const float* q, int64_t Q, float eps, float* scores) {
    NS_TRY
    if (rows <= 0) return 0;
    constexpr int T = 256, NW = T / WARP;
    g->q.parallel_for(sycl::nd_range<1>((size_t) rows * T, T), [=](nd1 it) [[sycl::reqd_sub_group_size(WARP)]] {
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id(), wi = (int) sg.get_group_linear_id();
        const float* p = src + r * D;
        float ss = 0.0f;
        for (int64_t c = tid; c < D; c += T) ss += p[c] * p[c];
        const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) D + eps);
        for (int64_t j = wi; j < Q; j += NW) {
            float s = 0.0f;
            for (int64_t c = lane; c < D; c += WARP) s += (p[c] * inv) * q[j * D + c];
            s = sg_sum(sg, s);
            if (lane == 0) scores[r * Q + j] = s;
        }
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_attnres_mix(ns_gpu* g, const float* bank, const float* scores, int64_t S, int64_t cap, int64_t rows, int64_t D, int64_t Q, int64_t qi,
                            float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    if (S < 1 || S > 64) return ns_fail("ns_audio_z3_attnres_mix: 1 to 64 sources");
    constexpr int T = 256;
    const int ns = (int) S;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> w(sycl::range<1>(64), h);
        h.parallel_for(sycl::nd_range<1>((size_t) rows * T, T), [=](nd1 it) {
            const int64_t r = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            if (tid == 0) {
                float m = -INFINITY;
                for (int s = 0; s < ns; ++s) m = sycl::fmax(m, scores[((int64_t) s * cap + r) * Q + qi]);
                float z = 0.0f;
                for (int s = 0; s < ns; ++s) {
                    w[s] = sycl::exp(scores[((int64_t) s * cap + r) * Q + qi] - m);
                    z += w[s];
                }
                for (int s = 0; s < ns; ++s) w[s] /= z;
            }
            sycl::group_barrier(it.get_group());
            for (int64_t d = tid; d < D; d += T) {
                float a = 0.0f;
                for (int s = 0; s < ns; ++s) a += w[s] * bank[((int64_t) s * cap + r) * D + d];
                out[r * D + d] = a;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_add_vec(ns_gpu* g, float* x, int64_t rows, int64_t C, const float* v) {
    NS_TRY
    if (rows * C > 0) each(g, rows * C, [=](int64_t i) { x[i] += v[i % C]; });
    return 0;
    NS_CATCH
}

int ns_audio_z3_gdn_conv(ns_gpu* g, const float* proj, int64_t rows, int64_t P, int64_t C, const float* w, int64_t K, float* hist, float* out) {
    NS_TRY
    if (rows * C <= 0) return 0;
    if (K < 2 || K > MAXK) return ns_fail("ns_audio_z3_gdn_conv: a kernel of 2 to 8 taps");
    const int kk = (int) K;
    // one item a channel, the rows in order (the window of the last K inputs in registers)
    each(g, C, [=](int64_t c) {
        float win[MAXK], wc[MAXK];
        for (int j = 0; j < kk; ++j) wc[j] = w[c * kk + j];
        for (int j = 0; j < kk - 1; ++j) win[j] = hist[c * (kk - 1) + j];
        for (int64_t t = 0; t < rows; ++t) {
            win[kk - 1] = proj[t * P + c];
            float s = 0.0f;
            for (int j = 0; j < kk; ++j) s += wc[j] * win[j];
            out[t * C + c] = siluf_(s);
            for (int j = 0; j < kk - 1; ++j) win[j] = win[j + 1];
        }
        for (int j = 0; j < kk - 1; ++j) hist[c * (kk - 1) + j] = win[j];
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_gdn_prep(ns_gpu* g, const float* conv, const float* proj, int64_t rows, int64_t P, int64_t ab0, int64_t H, int64_t S, const float* a_log,
                         const float* dt_bias, float* q, float* k, float* v, float* decay, float* beta) {
    NS_TRY
    if (rows <= 0) return 0;
    if (S % 32 || S > 1024) return ns_fail("ns_audio_z3_gdn_prep: heads of a multiple of 32 up to 1024");
    // a group a (row, head), S items
    g->q.parallel_for(sycl::nd_range<1>((size_t) (rows * H * S), (size_t) S), [=](nd1 it) {
        const int64_t r = it.get_group(0) / H;
        const int64_t h = it.get_group(0) % H;
        const int64_t i = it.get_local_id(0);
        const int64_t inner = H * S;
        const float* cr = conv + r * 3 * inner;
        const float qv = cr[h * S + i], kv = cr[inner + h * S + i], vv = cr[2 * inner + h * S + i];
        const float qn = sycl::reduce_over_group(it.get_group(), qv * qv, sycl::plus<float>());
        const float kn = sycl::reduce_over_group(it.get_group(), kv * kv, sycl::plus<float>());
        const int64_t o = (r * H + h) * S + i;
        q[o] = qv / sycl::sqrt(qn + 1e-6f) * (1.0f / sycl::sqrt((float) S));
        k[o] = kv / sycl::sqrt(kn + 1e-6f);
        v[o] = vv;
        if (i == 0) {
            const float a = proj[r * P + ab0 + h] + dt_bias[h];
            const float sp = a > 0.0f ? a + sycl::log1p(sycl::exp(-a)) : sycl::log1p(sycl::exp(a));
            decay[r * H + h] = sycl::exp(-sycl::exp(a_log[h]) * sp);
            beta[r * H + h] = sigmoidf_(proj[r * P + ab0 + H + h]);
        }
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_gdn_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* decay, const float* beta, int64_t rows, int64_t H,
                         int64_t S, float* state, float* y) {
    NS_TRY
    if (rows <= 0) return 0;
    if (S != GS) return ns_fail("ns_audio_z3_gdn_scan: heads of 128");
    const int hn = (int) H;
    // a sub-group a (head, value row); lane l holds the state's keys l + 32 j
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) H, (size_t) (GS / G_NW) * WARP * G_NW), sycl::range<2>(1, WARP * G_NW)),
                      [=](nd2 it) [[sycl::reqd_sub_group_size(WARP)]] {
        auto sg = it.get_sub_group();
        const int h = (int) it.get_group(0), lane = (int) sg.get_local_linear_id();
        const int vr = (int) it.get_group(1) * G_NW + (int) sg.get_group_linear_id();
        float* st = state + ((int64_t) h * GS + vr) * GS;
        float s[GJ];
#pragma unroll
        for (int j = 0; j < GJ; ++j) s[j] = st[lane + 32 * j];
        for (int64_t t = 0; t < rows; ++t) {
            const int64_t o = (t * hn + h) * GS;
            const float gg = decay[t * hn + h], b = beta[t * hn + h];
            float kv = 0.0f, kk[GJ], qq[GJ];
#pragma unroll
            for (int j = 0; j < GJ; ++j) {
                kk[j] = k[o + lane + 32 * j];
                qq[j] = q[o + lane + 32 * j];
                s[j] *= gg;
                kv += s[j] * kk[j];
            }
            kv = sg_sum(sg, kv);
            const float d = (v[o + vr] - kv) * b;
            float outv = 0.0f;
#pragma unroll
            for (int j = 0; j < GJ; ++j) {
                s[j] += kk[j] * d;
                outv += s[j] * qq[j];
            }
            outv = sg_sum(sg, outv);
            if (lane == 0) y[o + vr] = outv;
        }
#pragma unroll
        for (int j = 0; j < GJ; ++j) st[lane + 32 * j] = s[j];
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_gdn_out(ns_gpu* g, const float* y, const float* proj, int64_t P, int64_t z0, int64_t rows, int64_t H, int64_t S, const float* w, float eps,
                        float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    if (S % 32 || S > 1024) return ns_fail("ns_audio_z3_gdn_out: heads of a multiple of 32 up to 1024");
    g->q.parallel_for(sycl::nd_range<1>((size_t) (rows * H * S), (size_t) S), [=](nd1 it) {
        const int64_t r = it.get_group(0) / H, h = it.get_group(0) % H, i = it.get_local_id(0);
        const float yv = y[(r * H + h) * S + i];
        const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), yv * yv, sycl::plus<float>()) / (float) S + eps);
        const float z = proj[r * P + z0 + h * S + i];
        out[(r * H + h) * S + i] = yv * inv * w[i] * siluf_(z);
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_head_rms(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t H, int64_t D, const float* scale, float eps) {
    NS_TRY
    if (rows * H <= 0) return 0;
    // a sub-group a (row, head)
    const int64_t n = rows * H;
    g->q.parallel_for(sycl::nd_range<1>(rup(n * WARP, 256), 256), [=](nd1 it) [[sycl::reqd_sub_group_size(WARP)]] {
        auto sg = it.get_sub_group();
        const int64_t gw = (int64_t) it.get_global_id(0) / WARP;
        const int lane = (int) sg.get_local_linear_id();
        if (gw >= n) return;
        const int64_t r = gw / H, h = gw % H;
        float* p = x + r * xs + h * D;
        float ss = 0.0f;
        for (int64_t i = lane; i < D; i += WARP) ss += p[i] * p[i];
        const float inv = 1.0f / sycl::sqrt(sg_sum(sg, ss) / (float) D + eps);
        const float sc = scale ? scale[h] : 1.0f;
        for (int64_t i = lane; i < D; i += WARP) p[i] = p[i] * inv * sc;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t H, int64_t D, float theta, int64_t pos0) {
    NS_TRY
    const int64_t half_ = D / 2;
    if (rows * H * half_ <= 0) return 0;
    each(g, rows * H * half_, [=](int64_t i) {
        const int64_t r = i / (H * half_), h = (i / half_) % H, p = i % half_;
        float* e = x + r * xs + h * D + 2 * p;
        const float inv_freq = sycl::pow(theta, -2.0f * (float) p / (float) D);
        const float a = (float) (pos0 + r) * inv_freq, c = sycl::cos(a), s = sycl::sin(a);
        const float x0 = e[0], x1 = e[1];
        e[0] = x0 * c - x1 * s;
        e[1] = x0 * s + x1 * c;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_attention(ns_gpu* g, const float* q, int64_t qs, int64_t rows, const float* k, const float* v, int64_t kvs, int64_t keys, int64_t Hq,
                          int64_t Hk, int64_t D, float scale, int64_t pos0, int causal, int64_t window, float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    if (keys <= 0) return ns_fail("ns_audio_z3_attention: no keys");
    if (Hk <= 0 || Hq % Hk) return ns_fail("ns_audio_z3_attention: query heads a multiple of the key heads");
    switch (D) {
    case 16: attention_launch<16>(g, q, qs, rows, k, v, kvs, keys, (int) Hq, (int) Hk, scale, pos0, causal, window, out); break;
    case 32: attention_launch<32>(g, q, qs, rows, k, v, kvs, keys, (int) Hq, (int) Hk, scale, pos0, causal, window, out); break;
    case 64: attention_launch<64>(g, q, qs, rows, k, v, kvs, keys, (int) Hq, (int) Hk, scale, pos0, causal, window, out); break;
    default: return ns_fail("ns_audio_z3_attention: heads of 16, 32 or 64");
    }
    return 0;
    NS_CATCH
}

int ns_audio_z3_head_gate(ns_gpu* g, float* x, const float* gate, int64_t rows, int64_t H, int64_t D) {
    NS_TRY
    if (rows * H * D > 0) each(g, rows * H * D, [=](int64_t i) { x[i] *= sigmoidf_(gate[i / D]); });
    return 0;
    NS_CATCH
}

int ns_audio_z3_softcap(ns_gpu* g, float* x, int64_t n, float cap) {
    NS_TRY
    if (n > 0) each(g, n, [=](int64_t i) { x[i] = cap * sycl::tanh(x[i] / cap); });
    return 0;
    NS_CATCH
}

int ns_audio_z3_silu_mul(ns_gpu* g, const float* a, const float* b, float* out, int64_t n) {
    NS_TRY
    if (n > 0) each(g, n, [=](int64_t i) { out[i] = siluf_(a[i]) * b[i]; });
    return 0;
    NS_CATCH
}

int ns_audio_z3_topk(ns_gpu* g, const float* logits, const float* bias, int64_t rows, int64_t E, int64_t k, int32_t* idx, float* w) {
    NS_TRY
    if (rows <= 0) return 0;
    if (k < 1 || k > 8 || k > E) return ns_fail("ns_audio_z3_topk: 1 to 8 experts a row");
    const int ne = (int) E, kk = (int) k;
    each(g, rows, [=](int64_t r) {
        const float* lg = logits + r * ne;
        int sel[8];
        for (int s = 0; s < kk; ++s) {
            int best = -1;
            float bv = INFINITY;
            for (int e = 0; e < ne; ++e) {
                bool used = false;
                for (int u = 0; u < s; ++u) used |= sel[u] == e;
                if (used) continue;
                const float val = bias[e] - lg[e];
                if (val < bv) {
                    bv = val;
                    best = e;
                }
            }
            sel[s] = best < 0 ? 0 : best;
        }
        float m = -INFINITY;
        for (int s = 0; s < kk; ++s) m = sycl::fmax(m, lg[sel[s]]);
        float z = 0.0f, ex[8];
        for (int s = 0; s < kk; ++s) {
            ex[s] = sycl::exp(lg[sel[s]] - m);
            z += ex[s];
        }
        for (int s = 0; s < kk; ++s) {
            idx[r * kk + s] = sel[s];
            w[r * kk + s] = ex[s] / z;
        }
    });
    return 0;
    NS_CATCH
}

// the experts' up half: a sub-group an output f of a (row, slot), gate and up together
int ns_audio_z3_moe_up(ns_gpu* g, const float* x, int64_t rows, int64_t L, const void* gate, const void* up, const int32_t* idx, int64_t k, int64_t F,
                       float* h) {
    NS_TRY
    if (rows <= 0) return 0;
    if (L % 32) return ns_fail("ns_audio_z3_moe_up: rows of whole Q4_0 blocks");
    const uint8_t* G = (const uint8_t*) gate;
    const uint8_t* U = (const uint8_t*) up;
    const int kk = (int) k;
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (rows * k), (size_t) ((F + M_NW - 1) / M_NW) * WARP * M_NW), sycl::range<2>(1, WARP * M_NW)),
                      [=](nd2 it) [[sycl::reqd_sub_group_size(WARP)]] {
        auto sg = it.get_sub_group();
        const int64_t rs = it.get_group(0);
        const int64_t f = (int64_t) it.get_group(1) * M_NW + sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (f >= F) return;
        const int64_t e = idx[rs];
        const int64_t row_bytes = L / 32 * Q4B;
        const uint8_t* gr = G + (e * F + f) * row_bytes;
        const uint8_t* ur = U + (e * F + f) * row_bytes;
        const float* xr = x + (rs / kk) * L;
        float sgv = 0.0f, su = 0.0f;
        for (int64_t c = lane; c < L / 8; c += WARP) {
            sgv += q4_dot8(gr, c * 8, xr + c * 8);
            su += q4_dot8(ur, c * 8, xr + c * 8);
        }
        sgv = sg_sum(sg, sgv);
        su = sg_sum(sg, su);
        if (lane == 0) h[rs * F + f] = siluf_(sgv) * su;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_moe_down(ns_gpu* g, const float* h, int64_t rows, int64_t k, int64_t F, const void* down, const int32_t* idx, const float* w, int64_t L,
                         float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    if (F % 32) return ns_fail("ns_audio_z3_moe_down: rows of whole Q4_0 blocks");
    const uint8_t* Dn = (const uint8_t*) down;
    const int kk = (int) k;
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) rows, (size_t) ((L + M_NW - 1) / M_NW) * WARP * M_NW), sycl::range<2>(1, WARP * M_NW)),
                      [=](nd2 it) [[sycl::reqd_sub_group_size(WARP)]] {
        auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_group(1) * M_NW + sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (l >= L) return;
        const int64_t row_bytes = F / 32 * Q4B;
        float acc = 0.0f;
        for (int s = 0; s < kk; ++s) {
            const int64_t e = idx[r * kk + s];
            const uint8_t* dr = Dn + (e * L + l) * row_bytes;
            const float* hr = h + (r * kk + s) * F;
            float d = 0.0f;
            for (int64_t c = lane; c < F / 8; c += WARP) d += q4_dot8(dr, c * 8, hr + c * 8);
            acc += w[r * kk + s] * sg_sum(sg, d);
        }
        if (lane == 0) out[r * L + l] = acc;
    });
    return 0;
    NS_CATCH
}

// ---------------------------------------------------------------------------------------------------- decoder

int ns_audio_z3_lookup(ns_gpu* g, const int32_t* codes, int64_t T, int64_t ncol, int64_t col, const float* table, int64_t C, const float* bias, int acc,
                       float* out) {
    NS_TRY
    if (T * C <= 0) return 0;
    each(g, T * C, [=](int64_t i) {
        const int64_t t = i / C, c = i % C;
        const float v = table[(int64_t) codes[t * ncol + col] * C + c] + (bias ? bias[c] : 0.0f);
        out[i] = acc ? out[i] + v : v;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_dwconv_causal(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out) {
    NS_TRY
    if (T * C <= 0) return 0;
    each(g, T * C, [=](int64_t i) {
        const int64_t t = i / C, c = i % C;
        float s = bias ? bias[c] : 0.0f;
        for (int64_t j = 0; j < K; ++j) {
            const int64_t tt = t + j - (K - 1);
            if (tt >= 0) s += w[c * K + j] * x[tt * C + c];
        }
        out[i] = s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_dwconv_same(ns_gpu* g, const float* x, int64_t T, int64_t C, const float* w, int64_t K, const float* bias, float* out) {
    NS_TRY
    if (T * C <= 0) return 0;
    if (K % 2 == 0) return ns_fail("ns_audio_z3_dwconv_same: an odd kernel");
    each(g, T * C, [=](int64_t i) {
        const int64_t t = i / C, c = i % C;
        float s = bias ? bias[c] : 0.0f;
        for (int64_t j = 0; j < K; ++j) {
            const int64_t tt = t + j - K / 2;
            if (tt >= 0 && tt < T) s += w[c * K + j] * x[tt * C + c];
        }
        out[i] = s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_patch_up(ns_gpu* g, const float* x, int64_t T, int64_t C, int64_t h, float* out) {
    NS_TRY
    if (T * C * h <= 0) return 0;
    each(g, T * h * C, [=](int64_t i) {
        const int64_t o = i / C, c = i % C;  // out row o = t h + j
        const int64_t t = o / h, j = o % h;
        out[i] = x[t * C * h + c * h + j];
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_spec(ns_gpu* g, const float* y, int64_t F, int64_t bins, float maxlog, float* ri) {
    NS_TRY
    if (F * bins <= 0) return 0;
    each(g, F * bins, [=](int64_t i) {
        const int64_t f = i / bins, b = i % bins;
        const float mag = sycl::exp(sycl::fmin(y[f * 2 * bins + b], maxlog));
        const float ph = y[f * 2 * bins + bins + b];
        ri[f * 2 * bins + b] = mag * sycl::cos(ph);
        ri[f * 2 * bins + bins + b] = (b == 0 || b == bins - 1) ? 0.0f : mag * sycl::sin(ph);
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_ola(ns_gpu* g, const float* frames, int64_t F, int64_t n, int64_t hop, const float* win, float* out) {
    NS_TRY
    if (F * hop <= 0) return 0;
    each(g, F * hop, [=](int64_t i) {
        const int64_t p = i + n / 2;
        const int64_t f0 = p - n + 1 <= 0 ? 0 : (p - n + 1 + hop - 1) / hop;
        int64_t f1 = p / hop;
        if (f1 > F - 1) f1 = F - 1;
        float s = 0.0f, env = 0.0f;
        for (int64_t f = f0; f <= f1; ++f) {
            const int64_t j = p - f * hop;
            s += frames[f * n + j] * win[j];
            env += win[j] * win[j];
        }
        out[i] = env > 1e-11f ? s / env : s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_mixer(ns_gpu* g, const float* x0, const float* x1, int64_t L, const float* w, int64_t K, float* out) {
    NS_TRY
    if (L <= 0) return 0;
    each(g, L, [=](int64_t t) {
        const int64_t pad = (K - 1) / 2;
        float s = 0.0f;
        for (int64_t j = 0; j < K; ++j) {
            const int64_t tt = t + j - pad;
            if (tt < 0 || tt >= L) continue;
            s += w[j] * x0[tt] + w[K + j] * x1[tt];
        }
        out[t] = s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_resample(ns_gpu* g, const float* x, int64_t L, const float* kern, int64_t orig, int64_t nw, int64_t width, float* out, int64_t total) {
    NS_TRY
    if (total <= 0) return 0;
    each(g, total, [=](int64_t o) {
        const int64_t n = o / nw, i = o % nw, KL = 2 * width + orig;
        const float* kr = kern + i * KL;
        float s = 0.0f;
        for (int64_t j = 0; j < KL; ++j) {
            const int64_t src = n * orig + j - width;
            if (src >= 0 && src < L) s += kr[j] * x[src];
        }
        out[o] = s;
    });
    return 0;
    NS_CATCH
}

// ---------------------------------------------------------------------------------------------------- speaker encoder

int ns_audio_z3_conv3x3(ns_gpu* g, const float* x, int64_t Fi, int64_t Ti, int64_t Ci, const float* w, int64_t Co, int64_t a, int64_t sf, int64_t st,
                        const float* bias, float* out) {
    NS_TRY
    const int64_t Fo = (Fi - 1) / sf + 1, To = (Ti - 1) / st + 1;
    if (Fo * To * Co <= 0) return 0;
    if (a <= 0 || Ci % a || Co % (Ci / a)) return ns_fail("ns_audio_z3_conv3x3: the groups do not divide the channels");
    each(g, Fo * To * Co, [=](int64_t i) {
        const int64_t co = i % Co, to = (i / Co) % To, fo = i / (Co * To);
        const int64_t groups = Ci / a, per = Co / groups;
        const int64_t base = (co / per) * a;
        float s = bias ? bias[co] : 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
            const int64_t fi = fo * sf + kh - 1;
            if (fi < 0 || fi >= Fi) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int64_t ti = to * st + kw - 1;
                if (ti < 0 || ti >= Ti) continue;
                const float* xp = x + (fi * Ti + ti) * Ci + base;
                const float* wp = w + co * a * 9 + kh * 3 + kw;
                for (int64_t j = 0; j < a; ++j) s += wp[j * 9] * xp[j];
            }
        }
        out[i] = s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_affine(ns_gpu* g, const float* x, int64_t n, int64_t C, const float* scale, const float* shift, int relu, const float* add, float* out) {
    NS_TRY
    if (n * C <= 0) return 0;
    each(g, n * C, [=](int64_t i) {
        const int64_t c = i % C;
        float v = x[i] * scale[c] + shift[c];
        if (relu) v = sycl::fmax(v, 0.0f);
        if (add) v += add[i];
        out[i] = v;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_subsample(ns_gpu* g, const float* x, int64_t F, int64_t T, int64_t C, int64_t sf, int64_t st, float* out) {
    NS_TRY
    const int64_t Fo = (F + sf - 1) / sf, To = (T + st - 1) / st;
    if (Fo * To * C <= 0) return 0;
    each(g, Fo * To * C, [=](int64_t i) {
        const int64_t c = i % C, t = (i / C) % To, f = i / (C * To);
        out[i] = x[((f * sf) * T + t * st) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_swap01(ns_gpu* g, const float* x, int64_t A, int64_t B, int64_t C, float* out) {
    NS_TRY
    if (A * B * C <= 0) return 0;
    each(g, A * B * C, [=](int64_t i) {
        const int64_t c = i % C, a = (i / C) % A, b = i / (C * A);  // out [B, A, C]
        out[i] = x[(a * B + b) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_repeat_rows(ns_gpu* g, const float* x, int64_t rep, int64_t T, int64_t C, const float* add, float* out) {
    NS_TRY
    if (T * C <= 0) return 0;
    each(g, T * C, [=](int64_t i) {
        const int64_t t = i / C, c = i % C;
        out[i] = x[(t / rep) * C + c] + (add ? add[i] : 0.0f);
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_softmax(ns_gpu* g, float* x, int64_t n, int64_t C) {
    NS_TRY
    if (n <= 0) return 0;
    each(g, n, [=](int64_t r) {
        float* p = x + r * C;
        float m = -INFINITY;
        for (int64_t c = 0; c < C; ++c) m = sycl::fmax(m, p[c]);
        float z = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            p[c] = sycl::exp(p[c] - m);
            z += p[c];
        }
        for (int64_t c = 0; c < C; ++c) p[c] /= z;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_mix_experts(ns_gpu* g, const float* x, const float* r, int64_t T, int64_t E, int64_t J, float* out) {
    NS_TRY
    if (T * J <= 0) return 0;
    each(g, T * J, [=](int64_t i) {
        const int64_t t = i / J, j = i % J;
        float s = 0.0f;
        for (int64_t e = 0; e < E; ++e) s += r[t * E + e] * x[(t * E + e) * J + j];
        out[i] = s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_fuse4(ns_gpu* g, const float* x0, const float* x1, const float* x2, const float* x3, const float* w, int64_t T, int64_t C, float* out) {
    NS_TRY
    if (T * C <= 0) return 0;
    each(g, T * C, [=](int64_t i) {
        const int64_t c = i % C;
        out[i] = x0[i] * w[c] + x1[i] * w[C + c] + x2[i] * w[2 * C + c] + x3[i] * w[3 * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_astp(ns_gpu* g, const float* x, const float* att, int64_t T, int64_t C, float floor_, float* out) {
    NS_TRY
    if (C <= 0) return 0;
    if (T <= 0) return ns_fail("ns_audio_z3_astp: no frames");
    each(g, C, [=](int64_t c) {
        float m = -INFINITY;
        for (int64_t t = 0; t < T; ++t) m = sycl::fmax(m, att[t * C + c]);
        double z = 0.0, s1 = 0.0, s2 = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            const double a = sycl::exp((double) (att[t * C + c] - m));
            const double v = x[t * C + c];
            z += a;
            s1 += a * v;
            s2 += a * v * v;
        }
        const double mean = s1 / z, var = s2 / z - mean * mean;
        out[c] = (float) mean;
        out[C + c] = (float) sycl::sqrt(var > (double) floor_ ? var : (double) floor_);
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_tanh(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0) each(g, n, [=](int64_t i) { x[i] = sycl::tanh(x[i]); });
    return 0;
    NS_CATCH
}

// ---------------------------------------------------------------------------------------------------- reference encoder

int ns_audio_z3_snake_cf(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, const float* inv_alpha, float* out) {
    NS_TRY
    if (C * L <= 0) return 0;
    each(g, C * L, [=](int64_t i) {
        const int64_t c = i / L;
        const float sn = sycl::sin(alpha[c] * x[i]);
        out[i] = x[i] + inv_alpha[c] * sn * sn;
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    NS_TRY
    if (n > 0) each(g, n, [=](int64_t i) { x[i] += y[i]; });
    return 0;
    NS_CATCH
}

int ns_audio_z3_patch_down(ns_gpu* g, const float* x, int64_t T, int64_t C, int64_t h, float* out) {
    NS_TRY
    const int64_t To = T / h;
    if (To * C * h <= 0) return 0;
    each(g, To * C * h, [=](int64_t i) {
        const int64_t t = i / (C * h), r = i % (C * h), c = r / h, j = r % h;
        out[i] = x[(t * h + j) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_glu(ns_gpu* g, const float* x, int64_t T, int64_t C, float* out) {
    NS_TRY
    if (T * C <= 0) return 0;
    each(g, T * C, [=](int64_t i) {
        const int64_t t = i / C, c = i % C;
        out[i] = x[t * 2 * C + c] * sigmoidf_(x[t * 2 * C + C + c]);
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_silu(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0) each(g, n, [=](int64_t i) { x[i] = siluf_(x[i]); });
    return 0;
    NS_CATCH
}

// full attention with Wav2Vec2-BERT's relative positions: score = scale q . k_j + scale q . dist[clip(j - i, -left,
// right) + left], keys with valid[j] == 0 masked; q, k, v [T, H, 64] (rows qs apart), dist [left + right + 1, 64].
// Two passes over the keys (the row's max and sum, then the normalized probabilities times v), so that with bf != 0
// the scores, the probabilities and the products round to bfloat16 where the reference's BF16 recipe rounds them
// (ssl-encoder.cpp: content and relative scores each after the matmul and after the scale, their sum, the
// probabilities).
int ns_audio_z3_attn_rel(ns_gpu* g, const float* q, const float* k, const float* v, int64_t qs, int64_t T, int64_t H, int64_t D, const float* dist,
                         int64_t left, int64_t right, const float* valid, float scale, int bf, float* out) {
    NS_TRY
    if (T <= 0) return 0;
    if (D != RD) return ns_fail("ns_audio_z3_attn_rel: heads of 64");
    const size_t nb = (size_t) ((T + RQ - 1) / RQ);
    g->q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> ks(sycl::range<1>(RK * RD), hd), vs(sycl::range<1>(RK * RD), hd);
        hd.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) H, nb * RQ), sycl::range<2>(1, RQ)), [=](nd2 it) {
            const int64_t h = it.get_group(0);
            const int tid = (int) it.get_local_id(1);
            const int64_t i = (int64_t) it.get_group(1) * RQ + tid;
            const bool live = i < T;
            float qv[RD], acc[RD];
            float m = -INFINITY, l = 0.0f;
            for (int d = 0; d < RD; ++d) {
                qv[d] = live ? q[i * qs + h * RD + d] : 0.0f;
                acc[d] = 0.0f;
            }
            auto R = [bf](float x) { return bf ? bf16r(x) : x; };
            for (int pass = 0; pass < 2; ++pass) {
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                for (int64_t t0 = 0; t0 < T; t0 += RK) {
                    sycl::group_barrier(it.get_group());
                    for (int e = tid; e < RK * RD; e += RQ) {
                        const int kk = e / RD, d = e % RD;
                        const int64_t j = t0 + kk;
                        ks[kk * RD + d] = j < T ? k[j * qs + h * RD + d] : 0.0f;
                        vs[kk * RD + d] = j < T ? v[j * qs + h * RD + d] : 0.0f;
                    }
                    sycl::group_barrier(it.get_group());
                    if (!live) continue;
                    const int n = (int) sycl::min((int64_t) RK, T - t0);
                    for (int kk = 0; kk < n; ++kk) {
                        const int64_t j = t0 + kk;
                        if (valid && valid[j] == 0.0f) continue;
                        int64_t r = j - i;
                        r = r < -left ? -left : (r > right ? right : r);
                        const float* dr = dist + (r + left) * RD;
                        float c = 0.0f, rr = 0.0f;
#pragma unroll
                        for (int d = 0; d < RD; ++d) {
                            c += qv[d] * ks[kk * RD + d];
                            rr += qv[d] * dr[d];
                        }
                        const float s = R(R(R(c) * scale) + R(R(rr) * scale));
                        if (pass == 0) {
                            if (s > m) {
                                l *= sycl::exp(m - s);
                                m = s;
                            }
                            l += sycl::exp(s - m);
                        } else {
                            const float p = R(sycl::exp(s - m) * inv);
#pragma unroll
                            for (int d = 0; d < RD; ++d) acc[d] += p * vs[kk * RD + d];
                        }
                    }
                }
            }
            if (!live) return;
            for (int d = 0; d < RD; ++d) out[i * H * RD + h * RD + d] = R(acc[d]);
        });
    });
    return 0;
    NS_CATCH
}

int ns_audio_z3_bf16(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0) each(g, n, [=](int64_t i) { x[i] = bf16r(x[i]); });
    return 0;
    NS_CATCH
}

int ns_audio_z3_add_rows(ns_gpu* g, float* x, const float* bias, int64_t rows, int64_t cols) {
    NS_TRY
    if (rows * cols > 0) each(g, rows * cols, [=](int64_t i) { x[i] += bias[i / cols]; });
    return 0;
    NS_CATCH
}

// LACI's ACI mask (arXiv 2404.19723) on a decode row: the masked heads' attention recomputed over every key of the row
// (keys 0..keys-1) with the text keys outside each head's window blocked. One group a masked head; spec[m] = (head,
// lo, hi) in absolute key positions, text keys [ts, te). q [Hq hd], k / v rows kvs apart, out [Hq hd] overwritten.
// Each item sums its own keys' values; the group adds the items' sums (no atomics: the same answer every run)
int ns_audio_z3_attn_window(ns_gpu* g, const float* q, const float* k, const float* v, int64_t kvs, int64_t keys, int64_t Hq, int64_t Hk, int64_t hd,
                            float scale, const int* spec, int64_t n, int64_t ts, int64_t te, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    if (hd > 64) return ns_fail("ns_audio_z3_attn_window: heads of at most 64");
    const int64_t grp = Hq / Hk;
    g->q.parallel_for(sycl::nd_range<1>((size_t) n * WB, WB), [=](nd1 it) {
        const int64_t m_ = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const int head = spec[m_ * 3], lo = spec[m_ * 3 + 1], hi = spec[m_ * 3 + 2];
        const int64_t kv = head / grp;
        const float* qh = q + (int64_t) head * hd;
        auto score = [&](int64_t j) -> float {
            if (j >= ts && j < te && (j < lo || j > hi)) return -INFINITY;
            const float* kj = k + j * kvs + kv * hd;
            float s = 0.0f;
            for (int64_t d = 0; d < hd; ++d) s += qh[d] * kj[d];
            return s * scale;
        };
        float m = -INFINITY;
        for (int64_t j = tid; j < keys; j += WB) m = sycl::fmax(m, score(j));
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float acc[64];
        for (int d = 0; d < 64; ++d) acc[d] = 0.0f;
        float l = 0.0f;
        for (int64_t j = tid; j < keys; j += WB) {
            const float sc = score(j);
            if (sc == -INFINITY) continue;
            const float p = sycl::exp(sc - m);
            l += p;
            const float* vj = v + j * kvs + kv * hd;
            for (int64_t d = 0; d < hd; ++d) acc[d] += p * vj[d];
        }
        l = sycl::reduce_over_group(it.get_group(), l, sycl::plus<float>());
        for (int64_t d = 0; d < hd; ++d) {
            const float a = sycl::reduce_over_group(it.get_group(), acc[d], sycl::plus<float>());
            if (tid == 0) out[(int64_t) head * hd + d] = l > 0.0f ? a / l : 0.0f;
        }
    });
    return 0;
    NS_CATCH
}

// rotate-half RoPE from host tables (FuseAlign, Dia): x[r, h, :] = x * cos[r] + rotate_half(x) * sin[r], rotate_half =
// (-x2, x1); cos / sin [rows, D] (the angles of fractional positions, computed on the host as torch does)
int ns_audio_z3_rope_tab(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t H, int64_t D, const float* cs, const float* sn) {
    NS_TRY
    const int64_t half_ = D / 2;
    if (rows * H * half_ <= 0) return 0;
    each(g, rows * H * half_, [=](int64_t i) {
        const int64_t r = i / (H * half_), h = (i / half_) % H, p = i % half_;
        float* e = x + r * xs + h * D;
        const float a = e[p], b = e[p + half_];
        e[p] = a * cs[r * D + p] - b * sn[r * D + p];
        e[p + half_] = b * cs[r * D + p + half_] + a * sn[r * D + p + half_];
    });
    return 0;
    NS_CATCH
}

// ConvTranspose1d (kernel K, stride S, padding P) after its product: cols [T, K, C] (cols[t, k, c] = sum_i W[i, c, k]
// x[t, i]) -> y [To, C] + bias, y[j] = sum over (t, k) with t S - P + k = j
int ns_audio_z3_col2im(ns_gpu* g, const float* cols, int64_t T, int64_t K, int64_t C, int64_t S, int64_t P, const float* bias, int64_t To, float* y) {
    NS_TRY
    if (To * C <= 0) return 0;
    each(g, To * C, [=](int64_t i) {
        const int64_t j = i / C, c = i % C;
        float acc = bias ? bias[c] : 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            const int64_t n = j + P - k;
            if (n < 0 || n % S) continue;
            const int64_t t = n / S;
            if (t < T) acc += cols[(t * K + k) * C + c];
        }
        y[i] = acc;
    });
    return 0;
    NS_CATCH
}

// Full (non-causal) attention for heads of up to 128 (Dia's text encoder, FuseAlign's joint blocks): one group a
// (query, head), keys over the items, each item's sums added by the group; q, k, v [T, H hd] (rows qs apart)
int ns_audio_z3_attn_full(ns_gpu* g, const float* q, const float* k, const float* v, int64_t qs, int64_t T, int64_t H, int64_t hd, float scale, float* out) {
    NS_TRY
    if (T <= 0) return 0;
    if (hd > 128) return ns_fail("ns_audio_z3_attn_full: heads of at most 128");
    // the common head sizes: the tiled kernel (an item a query, the keys through local memory, an online softmax);
    // others: a group a (query, head)
    switch (hd) {
    case 32: attention_launch<32>(g, q, qs, T, k, v, qs, T, (int) H, (int) H, scale, 0, 0, 0, out); return 0;
    case 64: attention_launch<64>(g, q, qs, T, k, v, qs, T, (int) H, (int) H, scale, 0, 0, 0, out); return 0;
    case 128: attention_launch<128, 32>(g, q, qs, T, k, v, qs, T, (int) H, (int) H, scale, 0, 0, 0, out); return 0;
    default: break;
    }
    g->q.submit([&](sycl::handler& hh) {
        sycl::local_accessor<float, 1> qv(sycl::range<1>(128), hh), acc(sycl::range<1>(128), hh);
        hh.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) H, (size_t) T * FB), sycl::range<2>(1, FB)), [=](nd2 it) {
            const int64_t h = it.get_group(0), i = it.get_group(1);
            const int tid = (int) it.get_local_id(1);
            for (int64_t d = tid; d < hd; d += FB) {
                qv[d] = q[i * qs + h * hd + d];
                acc[d] = 0.0f;
            }
            sycl::group_barrier(it.get_group());
            float m = -INFINITY;
            for (int64_t j = tid; j < T; j += FB) {
                const float* kj = k + j * qs + h * hd;
                float s = 0.0f;
                for (int64_t d = 0; d < hd; ++d) s += qv[d] * kj[d];
                m = sycl::fmax(m, s * scale);
            }
            m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
            float l = 0.0f;
            for (int64_t j = tid; j < T; j += FB) {
                const float* kj = k + j * qs + h * hd;
                float s = 0.0f;
                for (int64_t d = 0; d < hd; ++d) s += qv[d] * kj[d];
                const float p = sycl::exp(s * scale - m);
                l += p;
                const float* vj = v + j * qs + h * hd;
                for (int64_t d = 0; d < hd; ++d) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> a(acc[d]);
                    a.fetch_add(p * vj[d]);
                }
            }
            l = sycl::reduce_over_group(it.get_group(), l, sycl::plus<float>());
            sycl::group_barrier(it.get_group());
            for (int64_t d = tid; d < hd; d += FB) out[i * H * hd + h * hd + d] = acc[d] / l;
        });
    });
    return 0;
    NS_CATCH
}

// The speech-edit LoRA on the routed experts (accents.gguf zonos-accent-lora/3: every routed (row, expert) pair adds
// up[e] @ (down[e] @ x) to its gate / up pre-activations and to its expert output).
// t[rs, r] = down[e, r, :] . x (x a row's: per_slot 0 -> row rs / k, 1 -> row rs); a sub-group a (rs, r)
int ns_audio_z3_moe_lora_t(ns_gpu* g, const float* x, int64_t rows, int64_t In, const int32_t* idx, int64_t k, int64_t per_slot, const float* down,
                           int64_t R, float* t) {
    NS_TRY
    if (rows <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (rows * k), (size_t) ((R + M_NW - 1) / M_NW) * WARP * M_NW), sycl::range<2>(1, WARP * M_NW)),
                      [=](nd2 it) [[sycl::reqd_sub_group_size(WARP)]] {
        auto sg = it.get_sub_group();
        const int64_t rs = it.get_group(0);
        const int64_t r = (int64_t) it.get_group(1) * M_NW + sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (r >= R) return;
        const int64_t e = idx[rs];
        const float* d = down + (e * R + r) * In;
        const float* xr = x + (per_slot ? rs : rs / k) * In;
        float acc = 0.0f;
        for (int64_t i = lane; i < In; i += WARP) acc += d[i] * xr[i];
        acc = sg_sum(sg, acc);
        if (lane == 0) t[rs * R + r] = acc;
    });
    return 0;
    NS_CATCH
}

// moe_up with the LoRA terms: gate += upg[e, f, :] . tg[rs], up += upu[e, f, :] . tu[rs]
int ns_audio_z3_moe_up_lora(ns_gpu* g, const float* x, int64_t rows, int64_t L, const void* gate, const void* up, const int32_t* idx, int64_t k, int64_t F,
                            const float* tg, const float* tu, const float* upg, const float* upu, int64_t R, float* h) {
    NS_TRY
    if (rows <= 0) return 0;
    if (L % 32) return ns_fail("ns_audio_z3_moe_up_lora: rows of whole Q4_0 blocks");
    const uint8_t* G = (const uint8_t*) gate;
    const uint8_t* U = (const uint8_t*) up;
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (rows * k), (size_t) ((F + M_NW - 1) / M_NW) * WARP * M_NW), sycl::range<2>(1, WARP * M_NW)),
                      [=](nd2 it) [[sycl::reqd_sub_group_size(WARP)]] {
        auto sg = it.get_sub_group();
        const int64_t rs = it.get_group(0);
        const int64_t f = (int64_t) it.get_group(1) * M_NW + sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (f >= F) return;
        const int64_t e = idx[rs];
        const int64_t row_bytes = L / 32 * Q4B;
        const uint8_t* gr = G + (e * F + f) * row_bytes;
        const uint8_t* ur = U + (e * F + f) * row_bytes;
        const float* xr = x + (rs / k) * L;
        float sgv = 0.0f, su = 0.0f;
        for (int64_t c = lane; c < L / 8; c += WARP) {
            sgv += q4_dot8(gr, c * 8, xr + c * 8);
            su += q4_dot8(ur, c * 8, xr + c * 8);
        }
        for (int64_t r = lane; r < R; r += WARP) {
            sgv += upg[(e * F + f) * R + r] * tg[rs * R + r];
            su += upu[(e * F + f) * R + r] * tu[rs * R + r];
        }
        sgv = sg_sum(sg, sgv);
        su = sg_sum(sg, su);
        if (lane == 0) h[rs * F + f] = siluf_(sgv) * su;
    });
    return 0;
    NS_CATCH
}

// moe_down with the LoRA term: an expert's output += upd[e, l, :] . td[rs]
int ns_audio_z3_moe_down_lora(ns_gpu* g, const float* h, int64_t rows, int64_t k, int64_t F, const void* down, const int32_t* idx, const float* w, int64_t L,
                              const float* td, const float* upd, int64_t R, float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    if (F % 32) return ns_fail("ns_audio_z3_moe_down_lora: rows of whole Q4_0 blocks");
    const uint8_t* Dn = (const uint8_t*) down;
    const int kk = (int) k;
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) rows, (size_t) ((L + M_NW - 1) / M_NW) * WARP * M_NW), sycl::range<2>(1, WARP * M_NW)),
                      [=](nd2 it) [[sycl::reqd_sub_group_size(WARP)]] {
        auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_group(1) * M_NW + sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        if (l >= L) return;
        const int64_t row_bytes = F / 32 * Q4B;
        float acc = 0.0f;
        for (int s = 0; s < kk; ++s) {
            const int64_t e = idx[r * kk + s];
            const uint8_t* dr = Dn + (e * L + l) * row_bytes;
            const float* hr = h + (r * kk + s) * F;
            float d = 0.0f;
            for (int64_t c = lane; c < F / 8; c += WARP) d += q4_dot8(dr, c * 8, hr + c * 8);
            for (int64_t q = lane; q < R; q += WARP) d += upd[(e * L + l) * R + q] * td[(r * kk + s) * R + q];
            acc += w[r * kk + s] * sg_sum(sg, d);
        }
        if (lane == 0) out[r * L + l] = acc;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
