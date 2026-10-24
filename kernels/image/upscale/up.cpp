// up.cpp: the upscalers' own kernels (up.h): what Real-ESRGAN's and waifu2x's ncnn graphs need beside the shared
// convolutions - the tile's pre- and post-processing as the two ncnn programs do them (mirrored or clamped borders, the
// +0.5 rounding), zero padding and dilation (a transposed convolution as a convolution), the activations, PixelShuffle,
// ncnn's Interp (nearest, bilinear, bicubic), Concat's channel copies, Crop, the squeeze-and-excitation blocks' mean,
// dense layers and channel scale. Channels-last half, float32 arithmetic, on the GPU's queue. After nextcuda's up.cu.
#include "ns.h"
#include "ns_internal.hpp"
#include "up.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;

inline float h2f(half h) { return (float) h; }
inline half f2h(float f) { return (half) f; }

inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }

inline int64_t mirror(int64_t i, int64_t n) {
    i = i < 0 ? -i : i;
    const int64_t d = i - (n - 1);
    return (n - 1) - (d < 0 ? -d : d);
}

inline int64_t clampi(int64_t i, int64_t n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

inline uint8_t to_u8(float v) {
    v = sycl::floor(v * 255.0f + 0.5f);
    return (uint8_t) (v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
}

inline float act(float v, int kind, float a, float slope) {
    switch (kind) {
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: return v > 0.0f ? v : v * a;
        case 3: return v > 0.0f ? v : v * slope;
        case 4: return 1.0f / (1.0f + sycl::exp(-v));
        default: return v;
    }
}

inline void cubic(float x, float* w) {
    const float A = -0.75f;
    w[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    w[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    w[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    w[3] = 1.0f - w[0] - w[1] - w[2];
}

// floor(a / b) for b > 0
inline int64_t floordiv(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}
}  // namespace

extern "C" {

int ns_image_up_pre(ns_gpu* g, const uint8_t* src, int64_t H, int64_t W, int64_t ch, int64_t x0, int64_t y0, int64_t tw, int64_t th,
                    int border, void* out) {
    NS_TRY
    half* o = (half*) out;
    const int64_t n = tw * th * 3;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % 3, x = (id / 3) % tw, y = id / 3 / tw;
        int64_t sx = x0 + x, sy = y0 + y;
        if (border == 0) {
            sx = mirror(sx, W);
            sy = mirror(sy, H);
        } else {
            sx = clampi(sx, W);
            sy = clampi(sy, H);
        }
        o[id] = f2h((float) src[(sy * W + sx) * ch + c] * (1.0f / 255.0f));
    });
    return 0;
    NS_CATCH
}

int ns_image_up_pre_alpha(ns_gpu* g, const uint8_t* src, int64_t H, int64_t W, int64_t x0, int64_t y0, int64_t tw, int64_t th, void* out) {
    NS_TRY
    half* o = (half*) out;
    const int64_t n = tw * th;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t x = id % tw, y = id / tw;
        const int64_t sx = clampi(x0 + x, W), sy = clampi(y0 + y, H);
        o[id] = f2h((float) src[(sy * W + sx) * 4 + 3] * (1.0f / 255.0f));
    });
    return 0;
    NS_CATCH
}

int ns_image_up_post(ns_gpu* g, const void* srcv, int64_t SW, int64_t sx, int64_t sy, const void* alphav, uint8_t* dst, int64_t DW,
                     int64_t dch, int64_t dx, int64_t dy, int64_t w, int64_t h) {
    NS_TRY
    const half* src = (const half*) srcv;
    const half* alpha = (const half*) alphav;
    const int64_t n = w * h * dch;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % dch, x = (id / dch) % w, y = id / dch / w;
        const float v = c < 3 ? h2f(src[((sy + y) * SW + sx + x) * 3 + c]) : (alpha ? h2f(alpha[y * w + x]) : 1.0f);
        dst[((dy + y) * DW + dx + x) * dch + c] = to_u8(v);
    });
    return 0;
    NS_CATCH
}

int ns_image_up_pad(ns_gpu* g, const void* xv, int64_t H, int64_t W, int64_t C, int64_t t, int64_t l, int64_t b, int64_t r, void* out) {
    NS_TRY
    const half* x = (const half*) xv;
    half* o = (half*) out;
    const int64_t Wo = W + l + r, n = (H + t + b) * Wo * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % C, ox = (id / C) % Wo, oy = id / C / Wo;
        const int64_t ix = ox - l, iy = oy - t;
        o[id] = (ix >= 0 && ix < W && iy >= 0 && iy < H) ? x[(iy * W + ix) * C + c] : f2h(0.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_up_dilate(ns_gpu* g, const void* xv, int64_t H, int64_t W, int64_t C, int64_t s, void* out) {
    NS_TRY
    const half* x = (const half*) xv;
    half* o = (half*) out;
    const int64_t Wo = (W - 1) * s + 1, n = ((H - 1) * s + 1) * Wo * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % C, ox = (id / C) % Wo, oy = id / C / Wo;
        o[id] = (ox % s == 0 && oy % s == 0) ? x[((oy / s) * W + ox / s) * C + c] : f2h(0.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_up_act(ns_gpu* g, void* xv, int64_t N, int64_t C, int kind, float a, const float* slopes, int64_t n_slopes) {
    NS_TRY
    half* x = (half*) xv;
    const int64_t n = N * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const float slope = slopes ? slopes[n_slopes == 1 ? 0 : id % C] : 0.0f;
        x[id] = f2h(act(h2f(x[id]), kind, a, slope));
    });
    return 0;
    NS_CATCH
}

int ns_image_up_pixel_shuffle(ns_gpu* g, const void* xv, int64_t H, int64_t W, int64_t C, int64_t r, void* out) {
    NS_TRY
    const half* x = (const half*) xv;
    half* o = (half*) out;
    const int64_t Wo = W * r, n = H * r * Wo * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % C, ox = (id / C) % Wo, oy = id / C / Wo;
        const int64_t ix = ox / r, iy = oy / r, dx = ox % r, dy = oy % r;
        o[id] = x[(iy * W + ix) * (C * r * r) + c * r * r + dy * r + dx];
    });
    return 0;
    NS_CATCH
}

int ns_image_up_resize(ns_gpu* g, const void* xv, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, int kind, void* out) {
    NS_TRY
    const half* x = (const half*) xv;
    half* o = (half*) out;
    const int64_t n = Ho * Wo * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % C, ox = (id / C) % Wo, oy = id / C / Wo;
        // nearest is integer; the others' source positions are exact rationals, (2 o + 1) W / (2 Wo) - 1/2 =
        // ((2 o + 1) W - Wo) / (2 Wo): integer numerators, one division for the fraction - no float division that a
        // fast-math build rounds (220 / 880 came out under 0.25 on CUDA and nearest took the pixel before), and no
        // double (the Arc cards have no native fp64)
        if (kind == 1) {
            const int64_t ix = sycl::min(ox * W / Wo, W - 1), iy = sycl::min(oy * H / Ho, H - 1);
            o[id] = x[(iy * W + ix) * C + c];
            return;
        }
        const int64_t nx = (2 * ox + 1) * W - Wo, ny = (2 * oy + 1) * H - Ho, dx2 = 2 * Wo, dy2 = 2 * Ho;
        int64_t sx = floordiv(nx, dx2), sy = floordiv(ny, dy2);
        float fx = (float) (nx - sx * dx2) / (float) dx2, fy = (float) (ny - sy * dy2) / (float) dy2;
        if (kind == 2) {
            if (sx < 0) { sx = 0; fx = 0.0f; }
            if (sx >= W - 1) { sx = W > 1 ? W - 2 : 0; fx = W > 1 ? 1.0f : 0.0f; }
            if (sy < 0) { sy = 0; fy = 0.0f; }
            if (sy >= H - 1) { sy = H > 1 ? H - 2 : 0; fy = H > 1 ? 1.0f : 0.0f; }
            const int64_t sx1 = sycl::min(sx + 1, W - 1), sy1 = sycl::min(sy + 1, H - 1);
            const float a = h2f(x[(sy * W + sx) * C + c]), b = h2f(x[(sy * W + sx1) * C + c]);
            const float d = h2f(x[(sy1 * W + sx) * C + c]), e = h2f(x[(sy1 * W + sx1) * C + c]);
            o[id] = f2h((a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy);
            return;
        }
        float wx[4], wy[4];
        cubic(fx, wx);
        cubic(fy, wy);
        float v = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const int64_t yy = clampi(sy - 1 + j, H);
            float row = 0.0f;
            for (int i = 0; i < 4; ++i) row += wx[i] * h2f(x[(yy * W + clampi(sx - 1 + i, W)) * C + c]);
            v += wy[j] * row;
        }
        o[id] = f2h(v);
    });
    return 0;
    NS_CATCH
}

int ns_image_up_add(ns_gpu* g, const void* av, const void* bv, int64_t n, int op, float ca, float cb, void* out) {
    NS_TRY
    const half* a = (const half*) av;
    const half* b = (const half*) bv;
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const size_t id = idx[0];
        const float x = h2f(a[id]), y = h2f(b[id]);
        o[id] = f2h(op == 1 ? x * y : (op == 2 ? sycl::fmax(x, y) : ca * x + cb * y));
    });
    return 0;
    NS_CATCH
}

int ns_image_up_copy_channels(ns_gpu* g, const void* srcv, int64_t P, int64_t Cs, void* dstv, int64_t Cd, int64_t off) {
    NS_TRY
    const half* src = (const half*) srcv;
    half* dst = (half*) dstv;
    const int64_t n = P * Cs;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % Cs, p = id / Cs;
        dst[p * Cd + off + c] = src[id];
    });
    return 0;
    NS_CATCH
}

int ns_image_up_crop(ns_gpu* g, const void* xv, int64_t W, int64_t C, int64_t x0, int64_t y0, int64_t w, int64_t h, void* out) {
    NS_TRY
    const half* x = (const half*) xv;
    half* o = (half*) out;
    const int64_t n = w * h * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const int64_t id = (int64_t) idx[0];
        const int64_t c = id % C, ox = (id / C) % w, oy = id / C / w;
        o[id] = x[((y0 + oy) * W + x0 + ox) * C + c];
    });
    return 0;
    NS_CATCH
}

// a work-group of 256 a channel: the mean over every pixel
int ns_image_up_mean(ns_gpu* g, const void* xv, int64_t P, int64_t C, float* out) {
    NS_TRY
    const half* x = (const half*) xv;
    if (C > 0) g->q.parallel_for(sycl::nd_range<1>(sz(C) * 256, 256), [=](sycl::nd_item<1> it) {
        const int64_t c = it.get_group(0);
        float acc = 0.0f;
        for (int64_t p = it.get_local_id(0); p < P; p += 256) acc += h2f(x[p * C + c]);
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        if (it.get_local_id(0) == 0) out[c] = acc / (float) P;
    });
    return 0;
    NS_CATCH
}

int ns_image_up_dense(ns_gpu* g, const float* x, int64_t I, const float* w, const float* b, int64_t O, int kind, float a, float* out) {
    NS_TRY
    if (O > 0) g->q.parallel_for(sycl::range<1>(sz(O)), [=](sycl::id<1> idx) {
        const int64_t o = (int64_t) idx[0];
        float v = b ? b[o] : 0.0f;
        for (int64_t i = 0; i < I; ++i) v += w[o * I + i] * x[i];
        out[o] = act(v, kind, a, 0.0f);
    });
    return 0;
    NS_CATCH
}

int ns_image_up_scale(ns_gpu* g, void* xv, int64_t P, int64_t C, const float* s) {
    NS_TRY
    half* x = (half*) xv;
    const int64_t n = P * C;
    if (n > 0) g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> idx) {
        const size_t id = idx[0];
        x[id] = f2h(h2f(x[id]) * s[id % C]);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
