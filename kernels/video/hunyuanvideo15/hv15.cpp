// hv15.cpp: the HunyuanVideo-1.5 engine's own kernels - what the shared diffusion kernels (nsd) do not have. The text
// encoders' are HunyuanImage 2.1's (kernels/image/hunyuanimage21/hi21.cpp, the same code: the image library and this
// video library each carry their engines' kernels); the DiT's modulation expansion; the 3-D causal VAE's replicate
// padding, channel RMS norm, upsample rearrangements and frame-causal attention softmax. Every loop is bounded by its
// arguments; each kernel is checked against the reference stage it serves (docs/ports/hunyuanvideo15.md).
#include "ns.h"
#include "ns_internal.hpp"
#include "hv15.h"

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

// ---- the text encoders' (hi21.cpp's) ------------------------------------------------------------------------------

int ns_video_hv15_q4_0_half(ns_gpu* g, const void* src, int64_t n, void* out) {
    NS_TRY
    if (n % 32 != 0) return ns_fail("ns_video_hv15_q4_0_half: a whole number of 32-value blocks");
    if (n <= 0) return 0;
    const uint8_t* b = (const uint8_t*) src;
    half* o = (half*) out;
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

int ns_video_hv15_ln_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                         const float* scale, const float* shift, void* out, int out_dt) {
    NS_TRY
    constexpr int WG = 256, PER = 32;
    if (C > (int64_t) WG * PER || C <= 0) return ns_fail("ns_video_hv15_ln_mod: rows of 1 to 8192");
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

int ns_video_hv15_act(ns_gpu* g, void* x, int dt, int64_t n, int kind) {
    NS_TRY
    if (kind < 0 || kind > 2) return ns_fail("ns_video_hv15_act: kind 0 (silu), 1 (gelu tanh) or 2 (gelu erf)");
    if (n <= 0) return 0;
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

int ns_video_hv15_geglu(ns_gpu* g, const void* x, int dt, int64_t M, int64_t C, void* out) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) M, (size_t) C), [=](sycl::id<2> id) {
        const size_t r = id[0], c = id[1];
        st(out, dt, r * C + c, gelu_tanh(ld(x, dt, r * 2 * C + c)) * ld(x, dt, r * 2 * C + C + c));
    });
    return 0;
    NS_CATCH
}

int ns_video_hv15_attn_small(ns_gpu* g, const float* q, int64_t qs, const float* k, const float* v, int64_t kvs, int64_t L, int64_t Hq,
                             int64_t Hkv, int64_t D, float scale, const float* bias, int causal, float* out) {
    NS_TRY
    constexpr int WG = 256, MAXL = 4096;
    if (L <= 0) return 0;
    if (L > MAXL || D > WG || Hkv <= 0 || Hq % Hkv != 0) return ns_fail("ns_video_hv15_attn_small: L <= 4096, D <= 256, Hq a multiple of Hkv");
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(MAXL), h);
        sycl::local_accessor<float, 1> qv(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (Hq * L * WG)), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0), hh = gi / L, i = gi % L, l = it.get_local_id(0);
            const int64_t kh = hh / (Hq / Hkv);
            if (l < D) qv[l] = q[i * qs + hh * D + l];
            sycl::group_barrier(it.get_group());
            const int64_t n = causal ? i + 1 : L;
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

int ns_video_hv15_axpy(ns_gpu* g, float* x, const float* y, int64_t n, float a) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += a * y[i]; });
    return 0;
    NS_CATCH
}

int ns_video_hv15_to_half(ns_gpu* g, const float* x, void* out, int64_t n, float s) {
    NS_TRY
    if (n <= 0) return 0;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) (x[i] * s); });
    return 0;
    NS_CATCH
}

int ns_video_hv15_to_float(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    const half* p = (const half*) x;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = (float) p[i]; });
    return 0;
    NS_CATCH
}

// ---- the DiT's ----------------------------------------------------------------------------------------------------

int ns_video_hv15_q8_0_float(ns_gpu* g, const void* src, int64_t n, float* out) {
    NS_TRY
    if (n % 32 != 0) return ns_fail("ns_video_hv15_q8_0_float: a whole number of 32-value blocks");
    if (n <= 0) return 0;
    const uint8_t* b = (const uint8_t*) src;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t blk = id[0] / 32, j = id[0] % 32;
        const uint8_t* x = b + blk * 34;
        half d;
        std::memcpy(&d, x, 2);
        out[id[0]] = (float) d * (float) (int8_t) x[2 + j];
    });
    return 0;
    NS_CATCH
}

// ---- the VAE's ----------------------------------------------------------------------------------------------------

int ns_video_hv15_pad_causal(ns_gpu* g, const void* x, const void* prev, int64_t T, int64_t H, int64_t W, int64_t C, void* out) {
    NS_TRY
    if (T <= 0 || H <= 0 || W <= 0 || C <= 0) return ns_fail("ns_video_hv15_pad_causal: an empty volume");
    const half* s = (const half*) x;
    const half* pv = (const half*) prev;
    half* o = (half*) out;
    const int64_t To = T + 2, Ho = H + 2, Wo = W + 2;
    g->q.parallel_for(sycl::range<2>((size_t) (To * Ho * Wo), (size_t) C), [=](sycl::id<2> id) {
        const int64_t p = id[0], c = id[1];
        const int64_t t = p / (Ho * Wo), y = (p / Wo) % Ho, xx = p % Wo;
        if (pv && t < 2) {
            // the previous chunk's last two padded frames, as they were
            o[p * C + c] = pv[p * C + c];
            return;
        }
        // replicate: the source index clamped into the volume
        const int64_t ts = t < 2 ? 0 : t - 2;
        const int64_t ys = y < 1 ? 0 : (y - 1 >= H ? H - 1 : y - 1), xs = xx < 1 ? 0 : (xx - 1 >= W ? W - 1 : xx - 1);
        o[p * C + c] = s[((ts * H + ys) * W + xs) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_video_hv15_rms_channels(ns_gpu* g, const void* x, int64_t N, int64_t C, const float* gamma, int silu, void* out) {
    NS_TRY
    constexpr int WG = 64, PER = 16;              // a voxel's up to 1,024 channels in registers
    if (C <= 0 || C > (int64_t) WG * PER) return ns_fail("ns_video_hv15_rms_channels: 1 to 1024 channels");
    if (N <= 0) return 0;
    const half* s = (const half*) x;
    half* o = (half*) out;
    const float root = sycl::sqrt((float) C);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) N * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        float v[PER];
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            v[j] = i < C ? (float) s[r * C + i] : 0.0f;
            ss += v[j] * v[j];
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float k = root / sycl::fmax(sycl::sqrt(ss), 1e-12f);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t i = (int64_t) j * WG + l;
            if (i < C) {
                float y = v[j] * k * gamma[i];
                if (silu) y = y / (1.0f + sycl::exp(-y));
                o[r * C + i] = (half) y;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_video_hv15_rep_add(ns_gpu* g, void* out, const void* x, int64_t N, int64_t C, int64_t Ci, int64_t rep) {
    NS_TRY
    if (rep <= 0 || Ci * rep != C) return ns_fail("ns_video_hv15_rep_add: C = Ci x rep");
    if (N <= 0) return 0;
    half* o = (half*) out;
    const half* s = (const half*) x;
    g->q.parallel_for(sycl::range<2>((size_t) N, (size_t) C), [=](sycl::id<2> id) {
        const size_t p = id[0], c = id[1];
        o[p * C + c] = (half) ((float) o[p * C + c] + (float) s[p * Ci + c / rep]);
    });
    return 0;
    NS_CATCH
}

int ns_video_hv15_add(ns_gpu* g, void* x, const void* y, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    half* a = (half*) x;
    const half* b = (const half*) y;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { a[i] = (half) ((float) a[i] + (float) b[i]); });
    return 0;
    NS_CATCH
}

int ns_video_hv15_upsample(ns_gpu* g, const void* conv, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t Ci, int temporal,
                           void* out) {
    return ns_video_hv15_upsample_ex(g, conv, x, T, H, W, C, Ci, temporal, 1, out);
}

int ns_video_hv15_upsample_ex(ns_gpu* g, const void* conv, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t Ci, int temporal,
                              int first, void* out) {
    NS_TRY
    if (T <= 0 || H <= 0 || W <= 0 || C <= 0 || Ci <= 0) return ns_fail("ns_video_hv15_upsample: an empty volume");
    const int64_t factor = temporal ? 8 : 4;
    const int64_t Cc = factor * C;
    if ((factor * C) % Ci != 0 || Ci % (temporal ? 8 : 4) != 0) return ns_fail("ns_video_hv15_upsample: channels that do not split");
    const int64_t rep = factor * C / Ci;          // the shortcut's repeats
    if (temporal && rep % 2 != 0) return ns_fail("ns_video_hv15_upsample: odd repeats on the first frame");
    const half* cv = (const half*) conv;
    const half* s = (const half*) x;
    half* o = (half*) out;
    const int64_t To = temporal ? (first ? 2 * T - 1 : 2 * T) : T, Ho = 2 * H, Wo = 2 * W;
    // a chunk after the first: its frame 0 is an ordinary frame (frame t of the output reads input frame t / 2)
    const int64_t lead = first ? 1 : 0;
    g->q.parallel_for(sycl::range<2>((size_t) (To * Ho * Wo), (size_t) C), [=](sycl::id<2> id) {
        const int64_t p = id[0], c = id[1];
        const int64_t to = p / (Ho * Wo), y = (p / Wo) % Ho, xx = p % Wo;
        const int64_t h = y / 2, w = xx / 2, r23 = (y % 2) * 2 + (xx % 2);
        float a, b;
        if (temporal && first && to == 0) {
            // "(r2 r3 c)": c of 2C, c < C kept; the shortcut's c of Ci / 4, repeated rep / 2 times
            const size_t q = (size_t) (h * W + w);
            a = (float) cv[q * Cc + r23 * 2 * C + c];
            b = (float) s[q * Ci + r23 * (Ci / 4) + c / (rep / 2)];
        } else if (temporal) {
            const int64_t t = lead + (to - lead) / 2, r1 = (to - lead) % 2;
            const size_t q = (size_t) ((t * H + h) * W + w);
            const int64_t r = r1 * 4 + r23;
            a = (float) cv[q * Cc + r * C + c];
            b = (float) s[q * Ci + r * (Ci / 8) + c / rep];
        } else {
            // the shortcut repeated first, then rearranged: channel r23 C + c of the repeated input
            const size_t q = (size_t) ((to * H + h) * W + w);
            a = (float) cv[q * Cc + r23 * C + c];
            b = (float) s[q * Ci + (r23 * C + c) / rep];
        }
        o[p * C + c] = (half) (a + b);
    });
    return 0;
    NS_CATCH
}

int ns_video_hv15_downsample(ns_gpu* g, const void* conv, const void* x, int64_t H, int64_t W, int64_t Cc, int64_t Ci, int64_t Cout, void* out) {
    NS_TRY
    if (H <= 0 || W <= 0 || H % 2 || W % 2 || Cc <= 0 || Ci <= 0 || Cout <= 0) return ns_fail("ns_video_hv15_downsample: even sides, channels");
    if (Cout % (4 * Cc) != 0 || (4 * Ci) % Cout != 0) return ns_fail("ns_video_hv15_downsample: channels that do not split");
    const half* cv = (const half*) conv;
    const half* s = (const half*) x;
    half* o = (half*) out;
    const int64_t Ho = H / 2, Wo = W / 2, gsz = 4 * Ci / Cout;
    g->q.parallel_for(sycl::range<2>((size_t) (Ho * Wo), (size_t) Cout), [=](sycl::id<2> id) {
        const int64_t p = id[0], oc = id[1];
        const int64_t y = p / Wo, xx = p % Wo;
        // the convolution: "(r2 r3 c)", repeated when Cout = 8 Cc (the first frame's channels twice)
        const int64_t j = oc % (4 * Cc), r23 = j / Cc, c = j % Cc;
        float a = (float) cv[((2 * y + r23 / 2) * W + 2 * xx + r23 % 2) * Cc + c];
        // the shortcut: x's "(r2 r3 c)" (4 Ci channels), each output the mean of its gsz
        float b = 0.0f;
        for (int64_t k = oc * gsz; k < (oc + 1) * gsz; ++k) {
            const int64_t q = k / Ci, ci = k % Ci;
            b += (float) s[((2 * y + q / 2) * W + 2 * xx + q % 2) * Ci + ci];
        }
        o[p * Cout + oc] = (half) (a + b / (float) gsz);
    });
    return 0;
    NS_CATCH
}

int ns_video_hv15_group_mean_add(ns_gpu* g, void* out, const void* x, int64_t N, int64_t C, int64_t r) {
    NS_TRY
    if (N <= 0 || C <= 0 || r <= 0) return 0;
    half* o = (half*) out;
    const half* s = (const half*) x;
    g->q.parallel_for(sycl::range<2>((size_t) N, (size_t) C), [=](sycl::id<2> id) {
        const int64_t n = id[0], c = id[1];
        float m = 0.0f;
        for (int64_t j = 0; j < r; ++j) m += (float) s[(n * C + c) * r + j];
        o[n * C + c] = (half) ((float) o[n * C + c] + m / (float) r);
    });
    return 0;
    NS_CATCH
}

int ns_video_hv15_transpose(ns_gpu* g, const void* x, int64_t R, int64_t C, void* out) {
    NS_TRY
    if (R <= 0 || C <= 0) return 0;
    const half* s = (const half*) x;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<2>((size_t) C, (size_t) R), [=](sycl::id<2> id) { o[id[0] * R + id[1]] = s[id[1] * C + id[0]]; });
    return 0;
    NS_CATCH
}

int ns_video_hv15_softmax_causal(ns_gpu* g, const float* x, int64_t M, int64_t Nk, int64_t row0, int64_t hw, float scale, void* out) {
    NS_TRY
    constexpr int WG = 256;
    if (hw <= 0 || row0 < 0) return ns_fail("ns_video_hv15_softmax_causal: a frame of pixels");
    if (M <= 0 || Nk <= 0) return 0;
    half* o = (half*) out;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = it.get_group(0);
        const int64_t l = (int64_t) it.get_local_id(0);
        const int64_t lim = ((row0 + (int64_t) r) / hw + 1) * hw;
        const int64_t n = lim < Nk ? lim : Nk;
        const float* row = x + r * Nk;
        float m = -std::numeric_limits<float>::infinity();
        for (int64_t i = l; i < n; i += WG) m = sycl::fmax(m, row[i] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t i = l; i < n; i += WG) s += sycl::exp(row[i] * scale - m);
        const float inv = 1.0f / sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        for (int64_t i = l; i < Nk; i += WG) o[r * Nk + i] = i < n ? (half) (sycl::exp(row[i] * scale - m) * inv) : (half) 0.0f;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
