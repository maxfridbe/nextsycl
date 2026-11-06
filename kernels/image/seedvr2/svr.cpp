// svr.cpp: the SeedVR2 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the VAE's per-frame
// group norm and upsampling shuffle, the frames' resize and colour fix, the DiT's modulated norm, its qk norm with
// interleaved rotary pairs and the windows' gathers. bfloat16 buffers (the reference's dtype), float32 arithmetic.
#include "ns.h"
#include "ns_internal.hpp"
#include "svr.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <map>
#include <mutex>
#include <string>

namespace {
using bf = sycl::ext::oneapi::bfloat16;
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
inline float silu(float v) { return v / (1.0f + sycl::exp(-v)); }
// 8 bfloat16 as one 16-byte load / store
using v8 = sycl::vec<uint16_t, 8>;
inline float bf2f(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }
inline uint16_t f2bf(float f) { return sycl::bit_cast<uint16_t>((bf) f); }

// a float32 scratch buffer per GPU, grown to the largest asked (the group norm's partial sums)
std::mutex mu;
struct Scratch {
    float* p = nullptr;
    size_t cap = 0;
};
std::map<ns_gpu*, Scratch> scratch;

float* grow(ns_gpu* g, size_t n) {
    std::lock_guard<std::mutex> l(mu);
    Scratch& s = scratch[g];
    if (n > s.cap) {
        g->q.wait();                               // nothing queued may still use the old one
        if (s.p) sycl::free(s.p, g->q);
        s.p = sycl::malloc_device<float>(n, g->q);
        s.cap = s.p ? n : 0;
    }
    return s.p;
}

// PyTorch's antialiased bicubic filter (a = -0.5, as PIL)
inline float aa_cubic(float x) {
    constexpr float a = -0.5f;
    x = sycl::fabs(x);
    if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
    return 0.0f;
}
}  // namespace

extern "C" {

// ---- the VAE ------------------------------------------------------------------------------------------------------

int ns_image_seedvr2_group_norm(ns_gpu* g, const void* x, int64_t F, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias,
                                float eps, int silu_, void* out) {
    NS_TRY
    if (F <= 0 || N <= 0 || C <= 0) return 0;
    if (G <= 0 || C % G) return ns_fail("ns_image_seedvr2_group_norm: the channels do not split into the groups");
    if (C % 8 == 0) {
        // 16-byte loads of 8 channels: the voxels split over P parts per frame, each part's 8 channels summed in
        // registers; then the groups' statistics; then the normalisation, 8 channels a work-item
        constexpr int64_t P = 1024;
        const int64_t C8 = C / 8;
        float* part = grow(g, sz(F * P * C * 2 + F * G * 2));
        if (!part) return ns_fail("ns_image_seedvr2_group_norm: no room for the partial sums");
        float* stat = part + F * P * C * 2;
        const v8* xv = (const v8*) x;
        g->q.parallel_for(sycl::range<3>(sz(F), sz(P), sz(C8)), [=](sycl::id<3> id) {
            const int64_t f = id[0], p = id[1], c8 = id[2];
            float s[8] = {0, 0, 0, 0, 0, 0, 0, 0}, s2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const v8* base = xv + (size_t) f * N * C8 + c8;
            for (int64_t n = p; n < N; n += P) {
                const v8 v = base[(size_t) n * C8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float a = bf2f(v[i]);
                    s[i] += a;
                    s2[i] += a * a;
                }
            }
            float* o = part + ((size_t) (f * P + p) * C + c8 * 8) * 2;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[2 * i] = s[i];
                o[2 * i + 1] = s2[i];
            }
        });
        const int64_t cg = C / G;
        // a work-group of 256 a (frame, group): the parts' sums split over it, then reduced together
        constexpr size_t RG = 256;
        g->q.parallel_for(sycl::nd_range<1>(sz(F * G) * RG, RG), [=](sycl::nd_item<1> it) {
            const int64_t fg = (int64_t) it.get_group(0), f = fg / G, gr = fg % G;
            const int64_t l = (int64_t) it.get_local_id(0);
            float a = 0.0f, a2 = 0.0f;
            for (int64_t p = l; p < P; p += RG)
                for (int64_t c = gr * cg; c < (gr + 1) * cg; ++c) {
                    a += part[((size_t) (f * P + p) * C + c) * 2];
                    a2 += part[((size_t) (f * P + p) * C + c) * 2 + 1];
                }
            const double sa = sycl::reduce_over_group(it.get_group(), (double) a, sycl::plus<double>());
            const double sa2 = sycl::reduce_over_group(it.get_group(), (double) a2, sycl::plus<double>());
            if (l == 0) {
                const double n = (double) N * (double) cg, mean = sa / n;
                const double var = sycl::fmax(sa2 / n - mean * mean, 0.0);
                stat[(f * G + gr) * 2] = (float) mean;
                stat[(f * G + gr) * 2 + 1] = (float) (1.0 / sycl::sqrt(var + (double) eps));
            }
        });
        v8* ov = (v8*) out;
        g->q.parallel_for(sycl::range<1>(sz(F * N * C8)), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int64_t c0 = (int64_t) (i % (size_t) C8) * 8, f = (int64_t) (i / ((size_t) N * C8));
            const v8 v = xv[i];
            v8 r;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int64_t c = c0 + k, gr = c / cg;
                const float y = (bf2f(v[k]) - stat[(f * G + gr) * 2]) * stat[(f * G + gr) * 2 + 1] * weight[c] + bias[c];
                r[k] = f2bf(silu_ ? silu(y) : y);
            }
            ov[i] = r;
        });
        return 0;
    }
    constexpr int64_t P = 256;                     // partial sums a (frame, channel): voxels part, part + P, ...
    float* part = grow(g, sz(F * P * C * 2 + F * G * 2));
    if (!part) return ns_fail("ns_image_seedvr2_group_norm: no room for the partial sums");
    float* stat = part + F * P * C * 2;
    const bf* xs = (const bf*) x;
    // neighbouring work-items read neighbouring channels of a voxel
    g->q.parallel_for(sycl::range<3>(sz(F), sz(P), sz(C)), [=](sycl::id<3> id) {
        const int64_t f = id[0], p = id[1], c = id[2];
        const bf* base = xs + (size_t) f * N * C;
        float s = 0.0f, s2 = 0.0f;
        for (int64_t n = p; n < N; n += P) {
            const float v = (float) base[(size_t) n * C + c];
            s += v;
            s2 += v * v;
        }
        part[((size_t) (f * P + p) * C + c) * 2] = s;
        part[((size_t) (f * P + p) * C + c) * 2 + 1] = s2;
    });
    const int64_t cg = C / G;
    g->q.parallel_for(sycl::range<2>(sz(F), sz(G)), [=](sycl::id<2> id) {
        const int64_t f = id[0], gr = id[1];
        double s = 0.0, s2 = 0.0;
        for (int64_t p = 0; p < P; ++p)
            for (int64_t c = gr * cg; c < (gr + 1) * cg; ++c) {
                s += part[((size_t) (f * P + p) * C + c) * 2];
                s2 += part[((size_t) (f * P + p) * C + c) * 2 + 1];
            }
        const double n = (double) N * (double) cg, mean = s / n;
        const double var = sycl::fmax(s2 / n - mean * mean, 0.0);
        stat[(f * G + gr) * 2] = (float) mean;
        stat[(f * G + gr) * 2 + 1] = (float) (1.0 / sycl::sqrt(var + (double) eps));
    });
    bf* o = (bf*) out;
    g->q.parallel_for(sycl::range<1>(sz(F * N * C)), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int64_t c = (int64_t) (i % (size_t) C), f = (int64_t) (i / ((size_t) N * C));
        const int64_t gr = c / cg;
        float v = ((float) xs[i] - stat[(f * G + gr) * 2]) * stat[(f * G + gr) * 2 + 1] * weight[c] + bias[c];
        o[i] = (bf) (silu_ ? silu(v) : v);
    });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_shuffle_up(ns_gpu* g, const void* x, int64_t T, int64_t H, int64_t W, int64_t C, int64_t sx, int64_t sy, int64_t sz_,
                                int drop_second, void* out) {
    NS_TRY
    const int64_t To = T * sz_ - (drop_second ? 1 : 0), Ho = H * sx, Wo = W * sy;
    if (To <= 0 || Ho <= 0 || Wo <= 0) return 0;
    const bf* xs = (const bf*) x;
    bf* o = (bf*) out;
    const int64_t Ci = sx * sy * sz_ * C;
    if (C % 8 == 0) {
        const v8* xv = (const v8*) x;
        v8* ov = (v8*) out;
        const int64_t C8 = C / 8, Ci8 = Ci / 8;
        g->q.parallel_for(sycl::range<3>(sz(To * Ho), sz(Wo), sz(C8)), [=](sycl::id<3> id) {
            const int64_t to = (int64_t) id[0] / Ho, yo = (int64_t) id[0] % Ho, xo = id[1], c8 = id[2];
            const int64_t tf = drop_second && to >= 1 ? to + 1 : to;
            const int64_t t = tf / sz_, zi = tf % sz_, y = yo / sx, xi = yo % sx, xx = xo / sy, yi = xo % sy;
            ov[((size_t) ((to * Ho + yo) * Wo + xo)) * C8 + c8] = xv[((size_t) ((t * H + y) * W + xx)) * Ci8 + (size_t) ((xi * sy + yi) * sz_ + zi) * C8 + c8];
        });
        return 0;
    }
    g->q.parallel_for(sycl::range<3>(sz(To * Ho), sz(Wo), sz(C)), [=](sycl::id<3> id) {
        const int64_t to = (int64_t) id[0] / Ho, yo = (int64_t) id[0] % Ho, xo = id[1], c = id[2];
        const int64_t tf = drop_second && to >= 1 ? to + 1 : to;   // the output frame of the whole shuffle
        const int64_t t = tf / sz_, zi = tf % sz_, y = yo / sx, xi = yo % sx, xx = xo / sy, yi = xo % sy;
        // channel ((xi sy + yi) sz + zi) C + c of input voxel (t, y, xx)
        const size_t src = ((size_t) ((t * H + y) * W + xx)) * Ci + (size_t) (((xi * sy + yi) * sz_ + zi) * C + c);
        o[((size_t) ((to * Ho + yo) * Wo + xo)) * C + c] = xs[src];
    });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_add(ns_gpu* g, void* x, const void* y, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    if (n % 8 == 0) {
        v8* xv = (v8*) x;
        const v8* yv = (const v8*) y;
        g->q.parallel_for(sycl::range<1>(sz(n / 8)), [=](sycl::id<1> i) {
            const v8 a = xv[i], b = yv[i];
            v8 r;
#pragma unroll
            for (int k = 0; k < 8; ++k) r[k] = f2bf(bf2f(a[k]) + bf2f(b[k]));
            xv[i] = r;
        });
        return 0;
    }
    bf* xs = (bf*) x;
    const bf* ys = (const bf*) y;
    g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { xs[i] = (bf) ((float) xs[i] + (float) ys[i]); });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_softmax_rows(ns_gpu* g, const float* s, int64_t R, int64_t L, float scale, void* out) {
    NS_TRY
    if (R <= 0 || L <= 0) return 0;
    constexpr size_t WG = 256;
    bf* o = (bf*) out;
    g->q.parallel_for(sycl::nd_range<1>(sz(R) * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = (int64_t) it.get_group(0);
        const int64_t l = (int64_t) it.get_local_id(0);
        const float* row = s + (size_t) r * L;
        float m = -INFINITY;
        for (int64_t j = l; j < L; j += WG) m = sycl::fmax(m, row[j] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float sum = 0.0f;
        for (int64_t j = l; j < L; j += WG) sum += sycl::exp(row[j] * scale - m);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.0f / sum;
        for (int64_t j = l; j < L; j += WG) o[(size_t) r * L + j] = (bf) (sycl::exp(row[j] * scale - m) * inv);
    });
    return 0;
    NS_CATCH
}

// ---- the frames ---------------------------------------------------------------------------------------------------

int ns_image_seedvr2_resize(ns_gpu* g, const uint8_t* rgb, int64_t N, int64_t Hi, int64_t Wi, int64_t Hr, int64_t Wr, int64_t oy, int64_t ox,
                            int64_t Ho, int64_t Wo, void* out) {
    NS_TRY
    if (N <= 0 || Ho <= 0 || Wo <= 0) return 0;
    if (oy < 0 || ox < 0 || oy + Ho > Hr || ox + Wo > Wr) return ns_fail("ns_image_seedvr2_resize: the crop is outside the resize");
    // the horizontal pass first (into float32 [N, Hi, Wo, 3]), then the vertical one
    float* mid = grow(g, sz(N * Hi * Wo * 3));
    if (!mid) return ns_fail("ns_image_seedvr2_resize: no room for the horizontal pass");
    // one axis's taps for output index i of `out_n` from `in_n`: PyTorch's _compute_index_ranges_weights for an
    // antialiased filter of 4 taps (support 2, widened by the scale when shrinking)
    auto taps = [](int64_t i, int64_t in_n, int64_t out_n, int64_t& lo, int64_t& n, float* w) {
        const float scale = (float) in_n / (float) out_n;
        const float support = scale >= 1.0f ? 2.0f * scale : 2.0f;
        const float inv = scale >= 1.0f ? 1.0f / scale : 1.0f;
        const float centre = scale * ((float) i + 0.5f);
        lo = sycl::max((int64_t) (centre - support + 0.5f), (int64_t) 0);
        const int64_t hi = sycl::min((int64_t) (centre + support + 0.5f), in_n);
        n = sycl::min(hi - lo, (int64_t) 16);
        float total = 0.0f;
        for (int64_t j = 0; j < n; ++j) total += (w[j] = aa_cubic(((float) (j + lo) - centre + 0.5f) * inv));
        if (total != 0.0f)
            for (int64_t j = 0; j < n; ++j) w[j] /= total;
    };
    g->q.parallel_for(sycl::range<3>(sz(N * Hi), sz(Wo), 3), [=](sycl::id<3> id) {
        const int64_t row = id[0], x = id[1], c = id[2];
        float w[16];
        int64_t lo, n;
        taps(x + ox, Wi, Wr, lo, n, w);
        float v = 0.0f;
        for (int64_t j = 0; j < n; ++j) v += w[j] * ((float) rgb[((size_t) row * Wi + lo + j) * 3 + c] / 255.0f);
        mid[((size_t) row * Wo + x) * 3 + c] = v;
    });
    bf* o = (bf*) out;
    g->q.parallel_for(sycl::range<3>(sz(N * Ho), sz(Wo), 3), [=](sycl::id<3> id) {
        const int64_t f = (int64_t) id[0] / Ho, y = (int64_t) id[0] % Ho, x = id[1], c = id[2];
        float w[16];
        int64_t lo, n;
        taps(y + oy, Hi, Hr, lo, n, w);
        float v = 0.0f;
        for (int64_t j = 0; j < n; ++j) v += w[j] * mid[(((size_t) f * Hi + lo + j) * Wo + x) * 3 + c];
        o[(((size_t) f * Ho + y) * Wo + x) * 3 + c] = (bf) (sycl::clamp(v, 0.0f, 1.0f) * 2.0f - 1.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_colorfix(ns_gpu* g, const void* content, const void* style, int64_t N, int64_t H, int64_t W, float* tmp, void* out) {
    NS_TRY
    if (N <= 0 || H <= 0 || W <= 0) return 0;
    const bf* cs = (const bf*) content;
    const bf* ss = (const bf*) style;
    const size_t n = (size_t) (N * H * W * 3);
    float* a = tmp;
    float* b = tmp + n;
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a[i] = (float) ss[i] - (float) cs[i]; });
    const int64_t cap = sycl::max((int64_t) 1, sycl::min(H, W) / 8);
    for (int level = 0; level < 5; ++level) {
        const int64_t r = sycl::min((int64_t) 1 << level, cap);
        float* src = level % 2 ? b : a;
        float* dst = level % 2 ? a : b;
        g->q.parallel_for(sycl::range<3>(sz(N * H), sz(W), 3), [=](sycl::id<3> id) {
            const int64_t f = (int64_t) id[0] / H, y = (int64_t) id[0] % H, x = id[1], c = id[2];
            const float k[3] = {0.25f, 0.5f, 0.25f};   // the 3 x 3 kernel is the outer product of these
            float v = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                const int64_t yy = sycl::clamp(y + dy * r, (int64_t) 0, H - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int64_t xx = sycl::clamp(x + dx * r, (int64_t) 0, W - 1);
                    v += k[dy + 1] * k[dx + 1] * src[(((size_t) f * H + yy) * W + xx) * 3 + c];
                }
            }
            dst[(((size_t) f * H + y) * W + x) * 3 + c] = v;
        });
    }
    // five passes: the last wrote b
    bf* o = (bf*) out;
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { o[i] = (bf) sycl::clamp((float) cs[i] + b[i], -1.0f, 1.0f); });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_to_rgb8(ns_gpu* g, const void* x, int64_t n, uint8_t* out) {
    NS_TRY
    const bf* xs = (const bf*) x;
    if (n > 0)
        g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) {
            out[i] = (uint8_t) sycl::rint(sycl::clamp((float) xs[i], -1.0f, 1.0f) * 127.5f + 127.5f);
        });
    return 0;
    NS_CATCH
}

// ---- the DiT ------------------------------------------------------------------------------------------------------

int ns_image_seedvr2_to_bf16(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    bf* o = (bf*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { o[i] = (bf) x[i]; });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_to_f32(ns_gpu* g, const void* x, float* out, int64_t n) {
    NS_TRY
    const bf* xs = (const bf*) x;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { out[i] = (float) xs[i]; });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_norm_mod(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* weight, float eps, const float* scale,
                              const float* shift, void* out) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    constexpr size_t WG = 256;
    bf* o = (bf*) out;
    g->q.parallel_for(sycl::nd_range<1>(sz(M) * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = (int64_t) it.get_group(0), l = (int64_t) it.get_local_id(0);
        const float* row = x + (size_t) r * C;
        float s2 = 0.0f;
        for (int64_t c = l; c < C; c += WG) s2 += row[c] * row[c];
        s2 = sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>());
        const float inv = 1.0f / sycl::sqrt(s2 / (float) C + eps);
        for (int64_t c = l; c < C; c += WG) {
            float v = row[c] * inv;
            if (weight) v *= weight[c];
            o[(size_t) r * C + c] = (bf) (v * scale[c] + shift[c]);
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_gate_add(ns_gpu* g, float* x, const void* y, int64_t M, int64_t C, const float* gate) {
    NS_TRY
    const bf* ys = (const bf*) y;
    if (M > 0 && C > 0)
        g->q.parallel_for(sycl::range<1>(sz(M * C)), [=](sycl::id<1> i) { x[i] += (float) ys[i] * gate[i % (size_t) C]; });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_qk_norm_rope(ns_gpu* g, void* qkv, int64_t M, int64_t H, int64_t D, int64_t stride, const float* wq, const float* wk,
                                  float eps, const float* cs, int64_t P) {
    NS_TRY
    if (M <= 0 || H <= 0) return 0;
    if (D != 128) return ns_fail("ns_image_seedvr2_qk_norm_rope: heads of 128 features only");
    if (2 * P > D) return ns_fail("ns_image_seedvr2_qk_norm_rope: more rotary pairs than a head has");
    bf* x = (bf*) qkv;
    // a work-group of D a (row, head, q or k); the pair partners are neighbouring lanes of one sub-group
    g->q.parallel_for(sycl::nd_range<1>(sz(M * H * 2) * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t grp = (int64_t) it.get_group(0);
        const int64_t r = grp / (H * 2), h = (grp / 2) % H, which = grp % 2;
        const int64_t d = (int64_t) it.get_local_id(0);
        bf* head = x + (size_t) r * stride + (size_t) (which * H + h) * 128;
        const float v = (float) head[d];
        const float s2 = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
        const float w = which ? wk[d] : wq[d];
        const float n = (float) (bf) (v / sycl::sqrt(s2 / 128.0f + eps) * w);   // the norm's output is bfloat16
        const float partner = sycl::permute_group_by_xor(it.get_sub_group(), n, 1);
        float o = n;
        const int64_t j = d / 2;
        if (cs && j < P) {
            const float c = cs[((size_t) r * P + j) * 2], s = cs[((size_t) r * P + j) * 2 + 1];
            o = d % 2 == 0 ? n * c - partner * s : n * c + partner * s;
        }
        head[d] = (bf) o;
    });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_gather(ns_gpu* g, const void* a, int64_t na, int64_t lda, const void* b, int64_t ldb, const int32_t* idx, int64_t rows,
                            int64_t cols, void* out) {
    NS_TRY
    if (rows <= 0 || cols <= 0) return 0;
    const bf* as = (const bf*) a;
    const bf* bs = (const bf*) b;
    bf* o = (bf*) out;
    g->q.parallel_for(sycl::range<2>(sz(rows), sz(cols)), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        const int64_t s = idx[r];
        o[(size_t) r * cols + c] = s < na ? as[(size_t) s * lda + c] : bs[(size_t) (s - na) * ldb + c];
    });
    return 0;
    NS_CATCH
}

int ns_image_seedvr2_text_mean(ns_gpu* g, const void* att, const int32_t* base, int64_t n, int64_t L, int64_t C, void* out) {
    NS_TRY
    if (n <= 0 || L <= 0 || C <= 0) return 0;
    const bf* as = (const bf*) att;
    bf* o = (bf*) out;
    g->q.parallel_for(sycl::range<2>(sz(L), sz(C)), [=](sycl::id<2> id) {
        const int64_t l = id[0], c = id[1];
        float s = 0.0f;
        for (int64_t w = 0; w < n; ++w) s += (float) as[((size_t) base[w] + l) * C + c];
        o[(size_t) l * C + c] = (bf) (s / (float) n);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
