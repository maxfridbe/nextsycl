// vision.cpp: picture operations the foundation's vision networks share (vision.h) - BiRefNet's convolutions as
// im2col + GEMM (its deformable ones too: torchvision's modulated sampling), Swin's windows and window attention, the
// resizes and rearrangements of its decoder. float32 throughout; every loop bounded by its arguments.
#include "ns.h"
#include "ns_internal.hpp"
#include "vision.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
constexpr int64_t kMaxRows = int64_t(1) << 31;

inline float bilinear_tv(const float* x, int64_t H, int64_t W, int64_t C, int64_t c, float y, float xx) {
    // torchvision's deform_conv2d bilinear_interpolate: zero beyond one pixel outside, missing corners zero
    if (y <= -1.f || (float) H <= y || xx <= -1.f || (float) W <= xx) return 0.f;
    const int64_t y0 = (int64_t) sycl::floor(y), x0 = (int64_t) sycl::floor(xx);
    const int64_t y1 = y0 + 1, x1 = x0 + 1;
    const float ly = y - (float) y0, lx = xx - (float) x0, hy = 1.f - ly, hx = 1.f - lx;
    const float v1 = (y0 >= 0 && x0 >= 0) ? x[(y0 * W + x0) * C + c] : 0.f;
    const float v2 = (y0 >= 0 && x1 <= W - 1) ? x[(y0 * W + x1) * C + c] : 0.f;
    const float v3 = (y1 <= H - 1 && x0 >= 0) ? x[(y1 * W + x0) * C + c] : 0.f;
    const float v4 = (y1 <= H - 1 && x1 <= W - 1) ? x[(y1 * W + x1) * C + c] : 0.f;
    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

inline int64_t region(int64_t t, int64_t P, int64_t ws, int64_t shift) {
    return t < P - ws ? 0 : (t < P - shift ? 1 : 2);
}
}  // namespace

extern "C" {

int ns_vision_im2col(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, int64_t kh, int64_t kw, int64_t s, int64_t p, int64_t d,
                     int64_t Ho, int64_t Wo, int64_t row0, int64_t rows, float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    if (row0 < 0 || row0 + rows > Ho * Wo || C <= 0 || kh <= 0 || kw <= 0 || s <= 0 || d <= 0) return ns_fail("ns_vision_im2col: bad shape");
    const int64_t K = kh * kw * C, n = rows * K;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t r = i / K, col = i % K;
        const int64_t c = col % C, tap = col / C, kx = tap % kw, ky = tap / kw;
        const int64_t o = row0 + r, oy = o / Wo, ox = o % Wo;
        const int64_t iy = oy * s - p + ky * d, ix = ox * s - p + kx * d;
        out[i] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[(iy * W + ix) * C + c] : 0.f;
    });
    return 0;
    NS_CATCH
}

int ns_vision_deform_im2col(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, int64_t k, int64_t p, const float* om,
                            int64_t row0, int64_t rows, float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    if (row0 < 0 || row0 + rows > H * W || C <= 0 || k <= 0) return ns_fail("ns_vision_deform_im2col: bad shape");
    const int64_t K = k * k * C, n = rows * K, oms = 3 * k * k;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t r = i / K, col = i % K;
        const int64_t c = col % C, tap = col / C, kx = tap % k, ky = tap / k;
        const int64_t o = row0 + r, oy = o / W, ox = o % W;
        const float* m = om + o * oms;
        const float dy = m[2 * tap], dx = m[2 * tap + 1];
        const float mask = 2.f / (1.f + sycl::exp(-m[2 * k * k + tap]));
        const float y = (float) (oy - p + ky) + dy, xx = (float) (ox - p + kx) + dx;
        out[i] = mask * bilinear_tv(x, H, W, C, c, y, xx);
    });
    return 0;
    NS_CATCH
}

int ns_vision_resize_ac(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t out_stride) {
    NS_TRY
    if (Ho <= 0 || Wo <= 0 || C <= 0) return 0;
    if (out_stride < C) return ns_fail("ns_vision_resize_ac: a stride under the channels");
    const float sy = Ho > 1 ? (float) (Hi - 1) / (float) (Ho - 1) : 0.f;
    const float sx = Wo > 1 ? (float) (Wi - 1) / (float) (Wo - 1) : 0.f;
    g->q.parallel_for(sycl::range<1>((size_t) (Ho * Wo * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t c = i % C, px = i / C, ox = px % Wo, oy = px / Wo;
        // PyTorch's upsample_bilinear2d with align_corners: the source index, its floor, the lambda
        const float fy = sy * (float) oy, fx = sx * (float) ox;
        const int64_t y0 = (int64_t) fy, x0 = (int64_t) fx;
        const int64_t y1 = y0 + (y0 < Hi - 1 ? 1 : 0), x1 = x0 + (x0 < Wi - 1 ? 1 : 0);
        const float ly = fy - (float) y0, lx = fx - (float) x0;
        const float v = (1.f - ly) * ((1.f - lx) * x[(y0 * Wi + x0) * C + c] + lx * x[(y0 * Wi + x1) * C + c]) +
                        ly * ((1.f - lx) * x[(y1 * Wi + x0) * C + c] + lx * x[(y1 * Wi + x1) * C + c]);
        out[px * out_stride + c] = v;
    });
    return 0;
    NS_CATCH
}

int ns_vision_tiles(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, int64_t G, float* out) {
    NS_TRY
    if (G <= 0 || H % G || W % G) return ns_fail("ns_vision_tiles: the picture does not split into G x G tiles");
    const int64_t h = H / G, w = W / G, Co = C * G * G;
    g->q.parallel_for(sycl::range<1>((size_t) (h * w * Co)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t ch = i % Co, px = i / Co, xx = px % w, y = px / w;
        const int64_t gx = ch % G, gy = (ch / G) % G, c = ch / (G * G);
        out[i] = x[((gy * h + y) * W + gx * w + xx) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_vision_window_gather(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, float* out) {
    NS_TRY
    if (ws <= 0 || shift < 0 || shift >= ws) return ns_fail("ns_vision_window_gather: bad window");
    const int64_t Hp = (H + ws - 1) / ws * ws, Wp = (W + ws - 1) / ws * ws, nww = Wp / ws, N = ws * ws;
    g->q.parallel_for(sycl::range<1>((size_t) (Hp * Wp * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t c = i % C, row = i / C, win = row / N, l = row % N;
        const int64_t ty = (win / nww) * ws + l / ws, tx = (win % nww) * ws + l % ws;
        const int64_t sy = (ty + shift) % Hp, sx = (tx + shift) % Wp;
        out[i] = (sy < H && sx < W) ? x[(sy * W + sx) * C + c] : 0.f;
    });
    return 0;
    NS_CATCH
}

int ns_vision_window_scatter_add(ns_gpu* g, const float* win, int64_t H, int64_t W, int64_t C, int64_t ws, int64_t shift, float* x) {
    NS_TRY
    if (ws <= 0 || shift < 0 || shift >= ws) return ns_fail("ns_vision_window_scatter_add: bad window");
    const int64_t Hp = (H + ws - 1) / ws * ws, Wp = (W + ws - 1) / ws * ws, nww = Wp / ws, N = ws * ws;
    g->q.parallel_for(sycl::range<1>((size_t) (Hp * Wp * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t c = i % C, row = i / C, w = row / N, l = row % N;
        const int64_t ty = (w / nww) * ws + l / ws, tx = (w % nww) * ws + l % ws;
        const int64_t sy = (ty + shift) % Hp, sx = (tx + shift) % Wp;
        if (sy < H && sx < W) x[(sy * W + sx) * C + c] += win[i];
    });
    return 0;
    NS_CATCH
}

int ns_vision_window_attention(ns_gpu* g, const float* qkv, int64_t nwh, int64_t nww, int64_t ws, int64_t heads, const float* bias, int64_t shift,
                               float scale, float* out) {
    NS_TRY
    constexpr int HD = 32;
    const int64_t N = ws * ws, C = heads * HD, nW = nwh * nww;
    if (N > 256 || N <= 0 || heads <= 0) return ns_fail("ns_vision_window_attention: windows of 1..256 positions");
    const int64_t WG = (N + 31) / 32 * 32, Hp = nwh * ws, Wp = nww * ws;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> kv(sycl::range<1>((size_t) (2 * N * HD)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (nW * heads * WG)), sycl::range<1>((size_t) WG)), [=](sycl::nd_item<1> it) {
            const int64_t grp = (int64_t) it.get_group(0), win = grp / heads, hh = grp % heads;
            const int64_t l = (int64_t) it.get_local_id(0);
            const float* base = qkv + win * N * 3 * C;
            for (int64_t j = l; j < N * HD; j += WG) {
                const int64_t r = j / HD, d = j % HD;
                kv[j] = base[r * 3 * C + C + hh * HD + d];
                kv[N * HD + j] = base[r * 3 * C + 2 * C + hh * HD + d];
            }
            sycl::group_barrier(it.get_group());
            if (l >= N) return;
            float q[HD], acc[HD];
            for (int d = 0; d < HD; ++d) {
                q[d] = base[l * 3 * C + hh * HD + d] * scale;
                acc[d] = 0.f;
            }
            const int64_t wy = win / nww, wx = win % nww;
            const int64_t ri = shift > 0 ? 3 * region(wy * ws + l / ws, Hp, ws, shift) + region(wx * ws + l % ws, Wp, ws, shift) : 0;
            const float* b = bias + (hh * N + l) * N;
            float m = -INFINITY, den = 0.f;
            for (int64_t j = 0; j < N; ++j) {
                float s = 0.f;
                for (int d = 0; d < HD; ++d) s += q[d] * kv[j * HD + d];
                s += b[j];
                if (shift > 0) {
                    const int64_t rj = 3 * region(wy * ws + j / ws, Hp, ws, shift) + region(wx * ws + j % ws, Wp, ws, shift);
                    if (rj != ri) s += -100.f;
                }
                const float mn = sycl::fmax(m, s);
                const float corr = sycl::exp(m - mn), e = sycl::exp(s - mn);
                den = den * corr + e;
                for (int d = 0; d < HD; ++d) acc[d] = acc[d] * corr + e * kv[N * HD + j * HD + d];
                m = mn;
            }
            float* o = out + (win * N + l) * C + hh * HD;
            for (int d = 0; d < HD; ++d) o[d] = acc[d] / den;
        });
    });
    return 0;
    NS_CATCH
}

int ns_vision_patch_merge(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out) {
    NS_TRY
    const int64_t H2 = (H + 1) / 2, W2 = (W + 1) / 2;
    g->q.parallel_for(sycl::range<1>((size_t) (H2 * W2 * 4 * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t ch = i % (4 * C), px = i / (4 * C), x2 = px % W2, y2 = px / W2;
        const int64_t part = ch / C, c = ch % C;
        const int64_t y = 2 * y2 + (part & 1), xx = 2 * x2 + (part >> 1);
        out[i] = (y < H && xx < W) ? x[(y * W + xx) * C + c] : 0.f;
    });
    return 0;
    NS_CATCH
}

int ns_vision_relu(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] = sycl::fmax(x[i], 0.f); });
    return 0;
    NS_CATCH
}

int ns_vision_mul_sigmoid(ns_gpu* g, float* x, int64_t M, int64_t C, const float* logit) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        x[i] *= 1.f / (1.f + sycl::exp(-logit[i / C]));
    });
    return 0;
    NS_CATCH
}

int ns_vision_copy_cols(ns_gpu* g, const float* src, int64_t M, int64_t C, int64_t ss, float* dst, int64_t ds) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    if (ss < C || ds < C) return ns_fail("ns_vision_copy_cols: a stride under the columns");
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / C, c = i % C;
        dst[r * ds + c] = src[r * ss + c];
    });
    return 0;
    NS_CATCH
}

int ns_vision_col_mean(ns_gpu* g, const float* x, int64_t M, int64_t C, float* out) {
    NS_TRY
    if (C <= 0) return 0;
    if (M <= 0 || M > kMaxRows) return ns_fail("ns_vision_col_mean: bad rows");
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) C * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t c = (int64_t) it.get_group(0);
        float s = 0.f;
        for (int64_t r = (int64_t) it.get_local_id(0); r < M; r += WG) s += x[r * C + c];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (it.get_local_id(0) == 0) out[c] = s / (float) M;
    });
    return 0;
    NS_CATCH
}

int ns_vision_broadcast_row(ns_gpu* g, const float* v, int64_t M, int64_t C, float* dst, int64_t ds) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    if (ds < C) return ns_fail("ns_vision_broadcast_row: a stride under the columns");
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        dst[(i / C) * ds + i % C] = v[i % C];
    });
    return 0;
    NS_CATCH
}

int ns_vision_add(ns_gpu* g, float* x, const float* y, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] += y[i]; });
    return 0;
    NS_CATCH
}

}  // extern "C"
