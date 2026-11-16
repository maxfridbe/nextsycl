// haru.cpp: the Haru-4B engine's own kernels (haru.h) - what its language model (Qwen3.5's hybrid of Gated DeltaNet and
// gated full attention) needs beyond the VibeVoice engine's products and norms.
//
// A frame runs one position (a prompt up to 16 at a time), so these are shaped for few rows: the DeltaNet's recurrence
// keeps its state in device memory and walks it once a row (a work-group a value head, an item a value column: the
// state's column stays with its item, no item waits on another's), and attention splits the keys into pieces for the
// GPU to be busy at any context (a sub-group a key at a time, the four query heads of a key head together), merged by
// a second kernel that applies the output gate. Everything in float32; the cache in half.
#include "ns.h"
#include "ns_internal.hpp"
#include "haru.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;

inline float bf16f(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t) b << 16);
}

inline float silu(float x) {
    return x / (1.0f + sycl::exp(-x));
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + sycl::exp(-x));
}

// torch's softplus (beta 1, threshold 20)
inline float softplus(float x) {
    return x > 20.0f ? x : sycl::log1p(sycl::exp(x));
}
} // namespace

extern "C" int ns_audio_haru_embed(ns_gpu* g, const void* table, int64_t D, const int32_t* idx, int64_t n, float* out, int64_t ldo) {
    NS_TRY
    if (n <= 0 || D <= 0) return n == 0 ? 0 : ns_fail("ns_audio_haru_embed: bad sizes");
    const uint16_t* t = (const uint16_t*) table;
    g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) D), [=](sycl::id<2> i) {
        const int64_t r = i[0], d = i[1];
        out[r * ldo + d] = bf16f(t[(int64_t) idx[r] * D + d]);
    });
    return 0;
    NS_CATCH
}

extern "C" int ns_audio_haru_gdn_conv(ns_gpu* g, const float* x, int64_t ldx, int64_t R, int64_t C, int64_t K, const float* w, float* state, float* out) {
    NS_TRY
    if (K < 2 || K > 8) return ns_fail("ns_audio_haru_gdn_conv: a kernel of 2 to 8 taps");
    if (R <= 0 || C <= 0) return R == 0 ? 0 : ns_fail("ns_audio_haru_gdn_conv: bad sizes");
    g->q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> i) {
        const int64_t c = i[0];
        float wk[8], hist[8];
        for (int j = 0; j < K; j++) wk[j] = w[c * K + j];
        for (int j = 0; j < K - 1; j++) hist[j] = state[j * C + c];
        for (int64_t r = 0; r < R; r++) {
            const float v = x[r * ldx + c];
            float s = wk[K - 1] * v;
            for (int j = 0; j < K - 1; j++) s += wk[j] * hist[j];
            out[r * C + c] = silu(s);
            for (int j = 0; j < K - 2; j++) hist[j] = hist[j + 1];
            hist[K - 2] = v;
        }
        for (int j = 0; j < K - 1; j++) state[j * C + c] = hist[j];
    });
    return 0;
    NS_CATCH
}

extern "C" int ns_audio_haru_gdn_rule(ns_gpu* g, const float* qkv, int64_t ldq, const float* ba, int64_t ldb, const float* A_log, const float* dt_bias,
                                      float* state, int64_t R, int64_t Hk, int64_t Hv, int64_t Dk, int64_t Dv, float* out) {
    NS_TRY
    if (Dk != 128 || Dv != 128) return ns_fail("ns_audio_haru_gdn_rule: heads of 128 keys and values");
    if (Hk <= 0 || Hv % Hk != 0) return ns_fail("ns_audio_haru_gdn_rule: value heads a multiple of the key heads");
    if (R <= 0) return R == 0 ? 0 : ns_fail("ns_audio_haru_gdn_rule: bad rows");
    const int64_t rep = Hv / Hk, kd = Hk * Dk;
    const float scale = 1.0f / sycl::sqrt((float) Dk);
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qn(sycl::range<1>(128), h), kn(sycl::range<1>(128), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) Hv * 128), sycl::range<1>(128)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t hv = it.get_group(0), j = it.get_local_id(0), hk = hv / rep;
            float* S = state + hv * Dk * Dv;
            const float al = sycl::exp(A_log[hv]), db = dt_bias[hv];
            for (int64_t r = 0; r < R; r++) {
                const float* row = qkv + r * ldq;
                const float qv = row[hk * Dk + j], kv = row[kd + hk * Dk + j], vv = row[2 * kd + hv * Dv + j];
                const float sq = sycl::reduce_over_group(it.get_group(), qv * qv, sycl::plus<float>());
                const float sk = sycl::reduce_over_group(it.get_group(), kv * kv, sycl::plus<float>());
                qn[j] = qv * sycl::rsqrt(sq + 1e-6f) * scale;
                kn[j] = kv * sycl::rsqrt(sk + 1e-6f);
                sycl::group_barrier(it.get_group());
                const float beta = sigmoid(ba[r * ldb + hv]);
                const float decay = sycl::exp(-al * softplus(ba[r * ldb + Hv + hv] + db));
                float mem = 0.0f;
                for (int k = 0; k < 128; k++) mem += S[k * Dv + j] * kn[k];
                const float delta = (vv - decay * mem) * beta;
                float o = 0.0f;
                for (int k = 0; k < 128; k++) {
                    const float s = decay * S[k * Dv + j] + kn[k] * delta;
                    S[k * Dv + j] = s;
                    o += s * qn[k];
                }
                out[r * Hv * Dv + hv * Dv + j] = o;
                sycl::group_barrier(it.get_group());
            }
        });
    });
    return 0;
    NS_CATCH
}

extern "C" int ns_audio_haru_gated_norm(ns_gpu* g, const float* y, const float* z, int64_t ldz, const float* w, float eps, int64_t R, int64_t H, int64_t D,
                                        float* out) {
    NS_TRY
    if (D != 128) return ns_fail("ns_audio_haru_gated_norm: heads of 128");
    if (R <= 0) return R == 0 ? 0 : ns_fail("ns_audio_haru_gated_norm: bad rows");
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (R * H * D)), sycl::range<1>((size_t) D)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int64_t grp = it.get_group(0), r = grp / H, hh = grp % H, d = it.get_local_id(0);
        const float v = y[r * H * D + hh * D + d];
        const float ss = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
        const float n = v * sycl::rsqrt(ss / (float) D + eps);
        out[r * H * D + hh * D + d] = w[d] * n * silu(z[r * ldz + hh * D + d]);
    });
    return 0;
    NS_CATCH
}

extern "C" int ns_audio_haru_attn_prep(ns_gpu* g, const float* qkv, int64_t ldq, int64_t R, int64_t Hq, int64_t Hkv, int64_t D, int64_t rot, const float* qw,
                                       const float* kw, float eps, const float* inv_freq, int64_t p0, float* q, void* kc, void* vc, int64_t T) {
    NS_TRY
    if (D != 256 || rot <= 0 || rot > D || rot % 2) return ns_fail("ns_audio_haru_attn_prep: heads of 256, an even rotary part");
    if (R <= 0) return R == 0 ? 0 : ns_fail("ns_audio_haru_attn_prep: bad rows");
    if (p0 + R > T) return ns_fail("ns_audio_haru_attn_prep: past the cache's end");
    half* K = (half*) kc;
    half* V = (half*) vc;
    const int64_t heads = Hq + 2 * Hkv;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> xs(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (R * heads * D)), sycl::range<1>((size_t) D)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t grp = it.get_group(0), r = grp / heads, hh = grp % heads, d = it.get_local_id(0);
            const float* row = qkv + r * ldq;
            const int64_t pos = p0 + r;
            if (hh >= Hq + Hkv) {
                const int64_t hv = hh - Hq - Hkv;
                V[(hv * T + pos) * D + d] = (half) row[Hq * 2 * D + Hkv * D + hv * D + d];
                return;
            }
            const bool isq = hh < Hq;
            const float v = isq ? row[hh * 2 * D + d] : row[Hq * 2 * D + (hh - Hq) * D + d];
            const float ss = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
            const float n = v * sycl::rsqrt(ss / (float) D + eps) * (isq ? qw[d] : kw[d]);
            xs[d] = n;
            sycl::group_barrier(it.get_group());
            float o = n;
            const int64_t half_rot = rot / 2;
            if (d < rot) {
                const int64_t i = d < half_rot ? d : d - half_rot;
                const float a = (float) pos * inv_freq[i];
                const float c = sycl::cos(a), s = sycl::sin(a);
                o = d < half_rot ? n * c - xs[d + half_rot] * s : n * c + xs[d - half_rot] * s;
            }
            if (isq) q[r * Hq * D + hh * D + d] = o;
            else K[((hh - Hq) * T + pos) * D + d] = (half) o;
        });
    });
    return 0;
    NS_CATCH
}

namespace {
constexpr int NSG = 8;
constexpr int GMAX = 4;
} // namespace

extern "C" int ns_audio_haru_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t R, int64_t Hq, int64_t Hkv, int64_t D, int64_t T, int64_t p0,
                                  int64_t chunk, float* part) {
    NS_TRY
    if (D != 256) return ns_fail("ns_audio_haru_attn: heads of 256");
    if (Hkv <= 0 || Hq % Hkv != 0 || Hq / Hkv > GMAX) return ns_fail("ns_audio_haru_attn: at most 4 query heads a key head");
    if (R <= 0 || chunk <= 0) return R == 0 ? 0 : ns_fail("ns_audio_haru_attn: bad sizes");
    if (p0 + R > T) return ns_fail("ns_audio_haru_attn: past the cache's end");
    const int64_t G = Hq / Hkv, P = (p0 + R + chunk - 1) / chunk;
    const half* K = (const half*) kc;
    const half* V = (const half*) vc;
    const float scale = 1.0f / sycl::sqrt((float) D);
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> ml(sycl::range<1>(NSG * GMAX * 2), h);
        sycl::local_accessor<float, 1> os(sycl::range<1>(NSG * GMAX * 256), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (R * Hkv * P * NSG * SG)), sycl::range<1>(NSG * SG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t grp = it.get_group(0);
            const int64_t p = grp % P, hk = (grp / P) % Hkv, r = grp / (P * Hkv);
            auto sg = it.get_sub_group();
            const int sgi = (int) sg.get_group_linear_id(), lane = (int) sg.get_local_linear_id();
            const int64_t nk = p0 + r + 1, k0 = p * chunk, k1 = sycl::min(k0 + chunk, nk);
            float qr[GMAX][16], o[GMAX][16], m[GMAX], l[GMAX];
            for (int gq = 0; gq < GMAX; gq++) {
                m[gq] = -INFINITY;
                l[gq] = 0.0f;
                for (int i = 0; i < 16; i++) {
                    o[gq][i] = 0.0f;
                    qr[gq][i] = gq < G ? q[r * Hq * D + (hk * G + gq) * D + lane * 16 + i] * scale : 0.0f;
                }
            }
            for (int64_t t = k0 + sgi; t < k1; t += NSG) {
                const half* kp = K + (hk * T + t) * D + lane * 16;
                const half* vp = V + (hk * T + t) * D + lane * 16;
                float kf[16], vf[16];
                for (int i = 0; i < 16; i++) {
                    kf[i] = (float) kp[i];
                    vf[i] = (float) vp[i];
                }
                for (int gq = 0; gq < GMAX; gq++) {
                    if (gq >= G) break;
                    float s = 0.0f;
                    for (int i = 0; i < 16; i++) s += qr[gq][i] * kf[i];
                    s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                    const float mn = sycl::fmax(m[gq], s);
                    const float a = sycl::exp(m[gq] - mn), e = sycl::exp(s - mn);
                    l[gq] = l[gq] * a + e;
                    for (int i = 0; i < 16; i++) o[gq][i] = o[gq][i] * a + e * vf[i];
                    m[gq] = mn;
                }
            }
            for (int gq = 0; gq < GMAX; gq++) {
                if (lane == 0) {
                    ml[(sgi * GMAX + gq) * 2] = m[gq];
                    ml[(sgi * GMAX + gq) * 2 + 1] = l[gq];
                }
                for (int i = 0; i < 16; i++) os[(sgi * GMAX + gq) * 256 + lane * 16 + i] = o[gq][i];
            }
            sycl::group_barrier(it.get_group());
            const int li = (int) it.get_local_id(0);
            for (int gq = 0; gq < G; gq++) {
                float M = -INFINITY;
                for (int s = 0; s < NSG; s++) M = sycl::fmax(M, ml[(s * GMAX + gq) * 2]);
                float* dst = part + ((r * Hq + hk * G + gq) * P + p) * (D + 2);
                float L = 0.0f, O0 = 0.0f, O1 = 0.0f;
                if (M > -INFINITY) {
                    for (int s = 0; s < NSG; s++) {
                        const float w = sycl::exp(ml[(s * GMAX + gq) * 2] - M);
                        L += w * ml[(s * GMAX + gq) * 2 + 1];
                        O0 += w * os[(s * GMAX + gq) * 256 + li];
                        O1 += w * os[(s * GMAX + gq) * 256 + li + 128];
                    }
                }
                if (li == 0) {
                    dst[0] = M;
                    dst[1] = L;
                }
                dst[2 + li] = O0;
                dst[2 + li + 128] = O1;
            }
        });
    });
    return 0;
    NS_CATCH
}

extern "C" int ns_audio_haru_attn_merge(ns_gpu* g, const float* part, int64_t R, int64_t Hq, int64_t D, int64_t P, const float* gate, int64_t ldg, int64_t hs,
                                        float* out) {
    NS_TRY
    if (R <= 0 || P <= 0 || D <= 0) return R == 0 ? 0 : ns_fail("ns_audio_haru_attn_merge: bad sizes");
    g->q.parallel_for(sycl::range<2>((size_t) (R * Hq), (size_t) D), [=](sycl::id<2> i) {
        const int64_t rh = i[0], d = i[1], r = rh / Hq, hh = rh % Hq;
        const float* src = part + rh * P * (D + 2);
        float M = -INFINITY;
        for (int64_t p = 0; p < P; p++) M = sycl::fmax(M, src[p * (D + 2)]);
        float L = 0.0f, O = 0.0f;
        for (int64_t p = 0; p < P; p++) {
            const float m = src[p * (D + 2)];
            if (m == -INFINITY) continue;
            const float w = sycl::exp(m - M);
            L += w * src[p * (D + 2) + 1];
            O += w * src[p * (D + 2) + 2 + d];
        }
        out[r * Hq * D + hh * D + d] = (L > 0.0f ? O / L : 0.0f) * sigmoid(gate[r * ldg + hh * hs + d]);
    });
    return 0;
    NS_CATCH
}
