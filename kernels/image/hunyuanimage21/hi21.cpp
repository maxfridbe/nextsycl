// hi21.cpp: the HunyuanImage 2.1 engine's own kernels - what the shared diffusion kernels (nsd) do not have: Q4_0 (the
// text encoder's file), the modulated / affine layer norm, the activations, T5's gated GELU, the single block's
// concatenation, attention for short text sequences (the token refiner, byT5 with its position bias, the text
// encoder's causal pass), the VAE's group norm, upsample shuffle and channel-repeat shortcuts, the picture's bytes.
// Every loop is bounded by its arguments; each kernel is checked against the reference stage it serves.
#include "ns.h"
#include "ns_internal.hpp"
#include "hi21.h"

#include <sycl/sycl.hpp>
#include <cstring>
#include <limits>

namespace {
using half = sycl::half;

inline float ld(const void* p, int dt, size_t i) {
    return dt == 0 ? ((const float*) p)[i] : (float) ((const half*) p)[i];
}
inline void st(void* p, int dt, size_t i, float v) {
    if (dt == 0) ((float*) p)[i] = v;
    else ((half*) p)[i] = (half) v;
}
inline float gelu_tanh(float v) {
    return 0.5f * v * (1.0f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
}  // namespace

extern "C" {

int ns_image_hi21_q4_0_half(ns_gpu* g, const void* src, int64_t n, void* out) {
    NS_TRY
    if (n % 32 != 0) return ns_fail("ns_image_hi21_q4_0_half: a whole number of 32-value blocks");
    const uint8_t* b = (const uint8_t*) src;
    half* o = (half*) out;
    // two values a work-item: byte j of a block gives values j and j + 16
    g->q.parallel_for(sycl::range<1>((size_t) (n / 2)), [=](sycl::id<1> id) {
        const size_t blk = id[0] / 16, j = id[0] % 16;
        const uint8_t* x = b + blk * 18;
        half d;
        std::memcpy(&d, x, 2);
        const float df = (float) d;
        const uint8_t q = x[2 + j];
        o[blk * 32 + j] = (half) (df * (float) ((int) (q & 0xF) - 8));
        o[blk * 32 + j + 16] = (half) (df * (float) ((int) (q >> 4) - 8));
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                         const float* scale, const float* shift, void* out, int out_dt) {
    NS_TRY
    constexpr int WG = 256, PER = 32;            // a row of up to 8,192 values held in registers
    if (C > (int64_t) WG * PER) return ns_fail("ns_image_hi21_ln_mod: rows longer than 8192");
    if (M <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const float* row = x + r * C;
        float v[PER];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            v[j] = i < C ? row[i] : 0.0f;
            s += v[j];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float s2 = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            const float d = i < C ? v[j] - mean : 0.0f;
            s2 += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) {
                float y = (v[j] - mean) * inv;
                if (weight) y = y * weight[i] + (bias ? bias[i] : 0.0f);
                if (scale) y = y * (1.0f + scale[i]);
                if (shift) y += shift[i];
                st(out, out_dt, r * C + i, y);
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_act(ns_gpu* g, void* x, int dt, int64_t n, int kind) {
    NS_TRY
    if (kind < 0 || kind > 2) return ns_fail("ns_image_hi21_act: kind 0 (silu), 1 (gelu tanh) or 2 (gelu erf)");
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = ld(x, dt, i[0]);
        float y;
        if (kind == 0) y = v / (1.0f + sycl::exp(-v));
        else if (kind == 1) y = gelu_tanh(v);
        else y = 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
        st(x, dt, i[0], y);
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_geglu(ns_gpu* g, const void* x, int dt, int64_t M, int64_t C, void* out) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], c = id[1];
        st(out, dt, r * C + c, gelu_tanh(ld(x, dt, r * 2 * C + c)) * ld(x, dt, r * 2 * C + C + c));
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_cat_gelu(ns_gpu* g, const void* att, int64_t M, int64_t A, const void* mlp, int64_t mlp_stride, int64_t F, void* out) {
    NS_TRY
    const half* a = (const half*) att;
    const half* m = (const half*) mlp;
    half* o = (half*) out;
    const int64_t W = A + F;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) W), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        o[r * W + c] = c < A ? a[r * A + c] : (half) gelu_tanh((float) m[r * mlp_stride + (c - A)]);
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_attn_small(ns_gpu* g, const float* q, int64_t qs, const float* k, const float* v, int64_t kvs, int64_t L, int64_t Hq,
                             int64_t Hkv, int64_t D, float scale, const float* bias, int causal, float* out) {
    NS_TRY
    constexpr int WG = 256, MAXL = 4096;
    if (L <= 0) return 0;
    if (L > MAXL || D > WG || Hkv <= 0 || Hq % Hkv != 0) return ns_fail("ns_image_hi21_attn_small: L <= 4096, D <= 256, Hq a multiple of Hkv");
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(MAXL), h);
        sycl::local_accessor<float, 1> qv(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (Hq * L * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0), hh = gi / L, i = gi % L, l = it.get_local_id(0);
            const int64_t kh = hh / (Hq / Hkv);
            if (l < D) qv[l] = q[i * qs + hh * D + l];
            sycl::group_barrier(it.get_group());
            const int64_t n = causal ? i + 1 : L;
            // the scores: a lane a key
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t j = l; j < n; j += WG) {
                const float* kr = k + j * kvs + kh * D;
                float s = 0.0f;
                for (int64_t d = 0; d < D; ++d) s += qv[d] * kr[d];
                s *= scale;
                if (bias) s += bias[(hh * L + i) * L + j];
                sc[j] = s;
                mx = sycl::fmax(mx, s);
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t j = l; j < n; j += WG) {
                const float e = sycl::exp(sc[j] - mx);
                sc[j] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            sycl::group_barrier(it.get_group());
            // the output: a lane a feature
            if (l < D) {
                float acc = 0.0f;
                for (int64_t j = 0; j < n; ++j) acc += sc[j] * v[j * kvs + kh * D + l];
                out[i * Hq * D + hh * D + l] = acc / sum;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += a * y[i]; });
    return 0;
    NS_CATCH
}

int ns_image_hi21_to_half(ns_gpu* g, const float* x, void* out, int64_t n, float s) {
    NS_TRY
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) (x[i] * s); });
    return 0;
    NS_CATCH
}

int ns_image_hi21_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    const half* p = (const half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = (float) p[i]; });
    return 0;
    NS_CATCH
}

int ns_image_hi21_rep_add(ns_gpu* g, void* out, const void* x, int64_t N, int64_t C, int64_t Ci, int64_t rep) {
    NS_TRY
    if (rep <= 0 || Ci * rep != C) return ns_fail("ns_image_hi21_rep_add: C = Ci x rep");
    half* o = (half*) out;
    const half* s = (const half*) x;
    g->q.parallel_for(sycl::range<2>((size_t) N, (size_t) C), [=](sycl::id<2> id) {
        const size_t p = id[0], c = id[1];
        o[p * C + c] = (half) ((float) o[p * C + c] + (float) s[p * Ci + c / rep]);
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_up_shuffle(ns_gpu* g, const void* conv, const void* x, int64_t H, int64_t W, int64_t C, int64_t Ci, int64_t rep, void* out) {
    NS_TRY
    if (rep <= 0 || Ci * rep != 4 * C) return ns_fail("ns_image_hi21_up_shuffle: 4 C = Ci x rep");
    const half* cv = (const half*) conv;
    const half* s = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<3>((size_t) (2 * H), (size_t) (2 * W), (size_t) C), [=](sycl::id<3> id) {
        const int64_t y = id[0], xx = id[1], c = id[2];
        const int64_t h = y / 2, w = xx / 2, r = (y % 2) * 2 + (xx % 2);
        const int64_t j = r * C + c;               // the channel of the 4C before the rearrangement
        const size_t p = (size_t) (h * W + w);
        o[((size_t) y * 2 * W + xx) * C + c] = (half) ((float) cv[p * 4 * C + j] + (float) s[p * Ci + j / rep]);
    });
    return 0;
    NS_CATCH
}

int64_t ns_image_hi21_gn_scratch(int64_t G) {
    return G * 256 * 2 + G * 2;
}

int ns_image_hi21_group_norm(ns_gpu* g, const void* x, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias, float eps,
                             int silu, float* scratch, void* out) {
    NS_TRY
    if (N <= 0 || C <= 0) return 0;
    if (G <= 0 || C % G != 0) return ns_fail("ns_image_hi21_group_norm: C a multiple of G");
    constexpr int64_t kParts = 256;               // partial sums a group
    const half* xi = (const half*) x;
    half* o = (half*) out;
    const int64_t cg = C / G;
    float* part = scratch;
    float* stat = scratch + G * kParts * 2;
    const int64_t per = (N + kParts - 1) / kParts;
    const float cnt = (float) N * (float) cg;
    // two passes, as PyTorch's: the mean, then the mean square about it (no cancellation); partial sums a part of
    // the group's pixels, then a group's parts summed
    for (int pass = 0; pass < 2; ++pass) {
        g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (G * kParts), 64), sycl::range<2>(1, 64)), [=](sycl::nd_item<2> it) {
            const int64_t gp = it.get_global_id(0), gr = gp / kParts, pt = gp % kParts;
            const int64_t lane = it.get_local_id(1);
            const float mean = pass == 0 ? 0.0f : stat[gr * 2];
            float s = 0.0f;
            const int64_t n0 = pt * per, n1 = sycl::min(N, n0 + per);
            for (int64_t n = n0; n < n1; ++n)
                for (int64_t j = lane; j < cg; j += 64) {
                    const float v = (float) xi[n * C + gr * cg + j] - mean;
                    s += pass == 0 ? v : v * v;
                }
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            if (lane == 0) part[gp] = s;
        });
        g->q.parallel_for(sycl::range<1>((size_t) G), [=](sycl::id<1> gr) {
            float s = 0.0f;
            for (int64_t p = 0; p < kParts; ++p) s += part[gr * kParts + p];
            if (pass == 0) stat[gr * 2] = s / cnt;
            else stat[gr * 2 + 1] = sycl::rsqrt(s / cnt + eps);
        });
    }
    g->q.parallel_for(sycl::range<2>((size_t) N, (size_t) C), [=](sycl::id<2> id) {
        const size_t n = id[0], ch = id[1], i = n * C + ch;
        const int64_t gr = ch / cg;
        float v = ((float) xi[i] - stat[gr * 2]) * stat[gr * 2 + 1] * weight[ch] + bias[ch];
        if (silu) v = v / (1.0f + sycl::exp(-v));
        o[i] = (half) v;
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_transpose(ns_gpu* g, const void* x, int64_t R, int64_t C, void* out) {
    NS_TRY
    const half* s = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<2>((size_t) C, (size_t) R), [=](sycl::id<2> id) {
        o[id[0] * R + id[1]] = s[id[1] * C + id[0]];
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_softmax_rows(ns_gpu* g, const float* x, int64_t M, int64_t N, float scale, void* out) {
    NS_TRY
    constexpr int WG = 256;
    half* o = (half*) out;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_local_id(0);
        const float* row = x + r * N;
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t i = l; i < N; i += WG) m = sycl::fmax(m, row[i] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t i = l; i < N; i += WG) s += sycl::exp(row[i] * scale - m);
        const float inv = 1.0f / sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        for (int64_t i = l; i < N; i += WG) o[r * N + i] = (half) (sycl::exp(row[i] * scale - m) * inv);
    });
    return 0;
    NS_CATCH
}

int ns_image_hi21_to_rgb8(ns_gpu* g, const void* x, int64_t n, uint8_t* rgb) {
    NS_TRY
    const half* s = (const half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) (n * 3)), [=](sycl::id<1> i) {
        const float v = sycl::clamp((float) s[i] * 0.5f + 0.5f, 0.0f, 1.0f);
        rgb[i] = (uint8_t) sycl::clamp(sycl::floor(v * 255.0f), 0.0f, 255.0f);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
