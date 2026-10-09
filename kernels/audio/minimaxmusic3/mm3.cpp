// mm3.cpp: the MiniMax Music 3 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the
// autoregressive stage's small-batch products and cached attention (two rows a frame: the prompt and its
// classifier-free twin), Qwen3's q / k norm with its rotary positions, the flow transformer's partial rotation, the
// codebooks' embeddings, the per-frame layer mix, and the sampler's step. On the GPU's queue; sub-groups of 16 (Xe2's
// native width).
#include "ns.h"
#include "ns_internal.hpp"
#include "mm3.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>

namespace {
using half = sycl::half;
constexpr int SG = 16;

inline float bf16f(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t) b << 16);
}

// ---- the decode's product: each sub-group R output rows, its lanes 8 features at a time across K
template <typename WT>
void gemv(sycl::queue& q, const float* x, int M, int64_t K, const WT* w, const float* scale, int64_t N, const float* bias, float* out,
          int64_t ldo, bool acc) {
    constexpr int NSG = 8, R = 2, MM = 8, V = 8;
    const int64_t groups = (N + NSG * R - 1) / (NSG * R);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SG * NSG), sycl::range<1>(SG * NSG)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t n0 = ((int64_t) it.get_group(0) * NSG + sg.get_group_linear_id()) * R;
        float a[MM][R];
#pragma unroll
        for (int m = 0; m < MM; ++m)
#pragma unroll
            for (int r = 0; r < R; ++r) a[m][r] = 0.0f;
        for (int64_t k = (int64_t) lane * V; k < K; k += SG * V) {
            float wf[R][V];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int64_t n = n0 + r < N ? n0 + r : N - 1;
                const sycl::vec<WT, V> wv = *reinterpret_cast<const sycl::vec<WT, V>*>(w + n * K + k);
#pragma unroll
                for (int i = 0; i < V; ++i) wf[r][i] = (float) wv[i];
            }
#pragma unroll
            for (int m = 0; m < MM; ++m) {
                if (m < M) {
                    const sycl::vec<float, 4> x0 = *reinterpret_cast<const sycl::vec<float, 4>*>(x + m * K + k);
                    const sycl::vec<float, 4> x1 = *reinterpret_cast<const sycl::vec<float, 4>*>(x + m * K + k + 4);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        a[m][r] += wf[r][0] * x0[0] + wf[r][1] * x0[1] + wf[r][2] * x0[2] + wf[r][3] * x0[3] +
                                   wf[r][4] * x1[0] + wf[r][5] * x1[1] + wf[r][6] * x1[2] + wf[r][7] * x1[3];
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            if (m < M) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float s = sycl::reduce_over_group(sg, a[m][r], sycl::plus<float>());
                    const int64_t n = n0 + r;
                    if (lane == 0 && n < N) {
                        float v = s * (scale ? scale[n] : 1.0f) + (bias ? bias[n] : 0.0f);
                        float* o = out + m * ldo + n;
                        *o = acc ? *o + v : v;
                    }
                }
            }
        }
    });
}

// ---- cached causal attention: a work-group per (row, key head, split of the keys), NSG sub-groups taking the
// split's keys in turn, each lane DL features of the G query heads that share the key head; the sub-groups' running
// (max, sum, weighted values) merged in local memory, the splits (decode only) by a second kernel
constexpr int64_t KB = 256;   // keys a split (decode)
constexpr int NSG_A = 4;

int64_t splits(int64_t S, int64_t kv) {
    return S == 1 && kv > KB ? (kv + KB - 1) / KB : 1;
}

template <int DL, int G>
void attn(sycl::queue& q, const float* qp, int64_t qs, const half* kc, const half* vc, int64_t B, int64_t S, int64_t Hq, int64_t Hkv,
          int64_t T, int64_t p0, float* out, float* part) {
    constexpr int D = SG * DL, WG = SG * NSG_A;
    const int64_t ns = splits(S, p0 + S);
    const float scale = 1.0f / sycl::sqrt((float) D);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NSG_A * G * (D + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * S * Hkv * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, r = gi / Hkv, b = r / S, s = r % S;
            const int64_t kv_end = p0 + s + 1;
            const int64_t j0 = sp * (ns > 1 ? KB : kv_end), j1 = sycl::min(kv_end, ns > 1 ? (sp + 1) * KB : kv_end);
            float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float* qr = qp + r * qs + (hk * G + g) * D + lane * DL;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    qv[g][i] = qr[i] * scale;
                    acc[g][i] = 0.0f;
                }
                m[g] = -INFINITY;
                l[g] = 0.0f;
            }
            const half* kb = kc + (b * Hkv + hk) * T * D + lane * DL;
            const half* vb = vc + (b * Hkv + hk) * T * D + lane * DL;
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
                const int64_t hq = hk * G + g;
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
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * S * Hq) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
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
}  // namespace

extern "C" {

int ns_audio_mm3_gemv(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* w, int wt, const float* scale, int64_t N,
                      const float* bias, float* out, int64_t ldo, int acc) {
    NS_TRY
    if (M <= 0 || N <= 0) return 0;
    if (M > 8 || K % 256 != 0) return ns_fail("ns_audio_mm3_gemv: at most 8 rows, K a multiple of 256");
    if (wt == 0) gemv(g->q, x, (int) M, K, (const half*) w, scale, N, bias, out, ldo, acc != 0);
    else if (wt == 1) gemv(g->q, x, (int) M, K, (const int8_t*) w, scale, N, bias, out, ldo, acc != 0);
    else return ns_fail("ns_audio_mm3_gemv: weights half (0) or int8 (1)");
    return 0;
    NS_CATCH
}

int64_t ns_audio_mm3_attn_scratch(int64_t B, int64_t S, int64_t Hq, int64_t D, int64_t p0) {
    const int64_t ns = splits(S, p0 + S);
    return ns > 1 ? B * S * Hq * ns * (D + 2) : 0;
}

int ns_audio_mm3_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t B, int64_t S, int64_t Hq,
                      int64_t Hkv, int64_t D, int64_t T, int64_t p0, float* out, float* part) {
    NS_TRY
    if (B <= 0 || S <= 0) return 0;
    if (p0 + S > T) return ns_fail("ns_audio_mm3_attn: past the cache's capacity");
    if (ns_audio_mm3_attn_scratch(B, S, Hq, D, p0) > 0 && !part) return ns_fail("ns_audio_mm3_attn: no scratch");
    const half* k = (const half*) kc;
    const half* v = (const half*) vc;
    if (D == 128 && Hq == 4 * Hkv) attn<8, 4>(g->q, q, qs, k, v, B, S, Hq, Hkv, T, p0, out, part);
    else if (D == 256 && Hq == Hkv) attn<16, 1>(g->q, q, qs, k, v, B, S, Hq, Hkv, T, p0, out, part);
    else return ns_fail("ns_audio_mm3_attn: D 128 with 4 query heads a key head, or D 256 with one");
    return 0;
    NS_CATCH
}

int ns_audio_mm3_kv_store(ns_gpu* g, const float* k, const float* v, int64_t ks, int64_t B, int64_t S, int64_t Hkv, int64_t D,
                          int64_t T, int64_t p0, void* kc, void* vc) {
    NS_TRY
    if (B <= 0 || S <= 0) return 0;
    if (p0 + S > T) return ns_fail("ns_audio_mm3_kv_store: past the cache's capacity");
    half* ko = (half*) kc;
    half* vo = (half*) vc;
    g->q.parallel_for(sycl::range<1>((size_t) (B * S * Hkv * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % Hkv, r = i / (D * Hkv), b = r / S, s = r % S;
        const int64_t at = ((b * Hkv + h) * T + p0 + s) * D + d;
        ko[at] = (half) k[r * ks + h * D + d];
        vo[at] = (half) v[r * ks + h * D + d];
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_qk_norm_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, const float* weight, float eps,
                              const float* inv_freq, int64_t p0) {
    NS_TRY
    if (rows <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (rows * H) * SG), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t rh = it.get_group(0), r = rh / H, h = rh % H;
        float* p = x + r * xs + h * 128;
        float v[8];
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = p[l + 16 * j];
            ss += v[j] * v[j];
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / 128.0f + eps);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = weight[l + 16 * j] * (v[j] * inv);
        const float pos = (float) (p0 + r % S);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = l + 16 * j;
            const float a = pos * inv_freq[i], c = sycl::cos(a), s = sycl::sin(a);
            p[i] = v[j] * c - v[j + 4] * s;
            p[i + 64] = v[j + 4] * c + v[j] * s;
        }
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_rope_partial(ns_gpu* g, void* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, int64_t rot,
                              const float* inv_freq) {
    NS_TRY
    if (rows <= 0) return 0;
    half* xp = (half*) x;
    const int64_t hr = rot / 2;
    g->q.parallel_for(sycl::range<1>((size_t) (rows * H * hr)), [=](sycl::id<1> id) {
        const int64_t i = id[0] % hr, h = (id[0] / hr) % H, r = id[0] / (hr * H);
        half* p = xp + r * xs + h * D;
        const float a = (float) (r % S) * inv_freq[i], c = sycl::cos(a), s = sycl::sin(a);
        const float x1 = (float) p[i], x2 = (float) p[i + hr];
        p[i] = (half) (x1 * c - x2 * s);
        p[i + hr] = (half) (x2 * c + x1 * s);
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_embed(ns_gpu* g, const void* table, int64_t D, const int32_t* idx, int n, float scale, float* out, int64_t ldo,
                       int copies) {
    NS_TRY
    if (n < 1 || n > 8) return ns_fail("ns_audio_mm3_embed: 1 to 8 rows");
    struct Ix { int32_t v[8]; } ix{};
    for (int i = 0; i < n; ++i) ix.v[i] = idx[i];
    const half* t = (const half*) table;
    g->q.parallel_for(sycl::range<1>((size_t) (D * copies)), [=](sycl::id<1> id) {
        const int64_t j = id[0] % D, c = id[0] / D;
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += (float) t[(int64_t) ix.v[i] * D + j];
        out[c * ldo + j] = s * scale;
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_mix(ns_gpu* g, const float* h, int n, int64_t D, const float* w, float scale, float* out) {
    NS_TRY
    if (n < 1 || n > 8) return ns_fail("ns_audio_mm3_mix: 1 to 8 rows");
    struct Wt { float v[8]; } wt{};
    for (int i = 0; i < n; ++i) wt.v[i] = w[i];
    g->q.parallel_for(sycl::range<1>((size_t) D), [=](sycl::id<1> id) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += wt.v[i] * h[i * D + id[0]];
        out[id[0]] = scale * s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_bf16_to_half(ns_gpu* g, const uint16_t* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) bf16f(x[i]); });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_bf16_to_float(ns_gpu* g, const uint16_t* x, float* out, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = bf16f(x[i]); });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_quant_rows(ns_gpu* g, const float* w, int64_t N, int64_t K, int8_t* q, float* scale) {
    NS_TRY
    constexpr int WG = 256;
    if (N <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) N * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_local_id(0);
        const float* row = w + r * K;
        float mx = 0.0f;
        for (int64_t i = l; i < K; i += WG) mx = sycl::fmax(mx, sycl::fabs(row[i]));
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        const float s = sycl::fmax(mx / 127.0f, 1e-30f);
        for (int64_t i = l; i < K; i += WG) q[r * K + i] = (int8_t) sycl::clamp(sycl::rint(row[i] / s), -127.0f, 127.0f);
        if (l == 0) scale[r] = s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_dequant_rows(ns_gpu* g, const int8_t* q, const float* scale, int64_t N, int64_t K, void* out) {
    NS_TRY
    half* o = (half*) out;
    if (N > 0 && K > 0) g->q.parallel_for(sycl::range<1>((size_t) (N * K)), [=](sycl::id<1> i) { o[i] = (half) ((float) q[i] * scale[i / K]); });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += y[i]; });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_dit_in(ns_gpu* g, const float* lat, const float* cond, int64_t L, int64_t C, int64_t Cc, float* out) {
    NS_TRY
    const int64_t W = 2 * C + Cc;
    if (L > 0) g->q.parallel_for(sycl::range<1>((size_t) (2 * L * W)), [=](sycl::id<1> id) {
        const int64_t c = id[0] % W, l = (id[0] / W) % L, b = id[0] / (W * L);
        float v = 0.0f;
        if (c < C) v = lat[l * C + c];
        else if (c >= 2 * C && b == 0) v = cond[l * Cc + c - 2 * C];
        out[id[0]] = v;
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_cfg_step(ns_gpu* g, float* lat, const float* vc, const float* vu, int64_t n, float cfg, float dt) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { lat[i] += dt * (vu[i] + cfg * (vc[i] - vu[i])); });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_blend(ns_gpu* g, float* x, const float* p, const float* q, int64_t n, float a, float b) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] = a * p[i] + b * q[i]; });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_nearest_rows(ns_gpu* g, const float* x, int64_t C, int64_t Li, int64_t Lo, float* out) {
    NS_TRY
    if (Lo <= 0 || Li <= 0) return 0;
    const float sc = (float) Li / (float) Lo;
    g->q.parallel_for(sycl::range<1>((size_t) (Lo * C)), [=](sycl::id<1> id) {
        const int64_t c = id[0] % C, j = id[0] / C;
        int64_t s;
        if (Lo == Li) s = j;
        else if (Lo == 2 * Li) s = j >> 1;
        else s = sycl::min((int64_t) sycl::floor((float) j * sc), Li - 1);
        out[id[0]] = x[c * Li + s];
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_transpose(ns_gpu* g, const float* x, int64_t R, int64_t C, float* out) {
    NS_TRY
    if (R > 0 && C > 0) g->q.parallel_for(sycl::range<1>((size_t) (R * C)), [=](sycl::id<1> id) {
        const int64_t r = id[0] % R, c = id[0] / R;
        out[id[0]] = x[r * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_mm3_tanh(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] = sycl::tanh(x[i]); });
    return 0;
    NS_CATCH
}

}  // extern "C"
