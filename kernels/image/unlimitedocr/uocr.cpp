// uocr.cpp: the Unlimited-OCR engine's kernels (uocr.h) - the vision tower's rearrangements (patches, a 3 x 3
// convolution's columns, SAM's windows), SAM's attention with its decomposed relative positions, CLIP's quick GELU,
// and the decoder's row copies and its experts' gathers and scatters. The linears, norms and the decoder's attention
// are the shared diffusion kernels' (nsd.h).
#include "ns.h"
#include "ns_internal.hpp"
#include "uocr.h"

#include <sycl/sycl.hpp>
#include <cmath>

using half = sycl::half;

extern "C" {

int ns_image_uocr_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    NS_TRY
    half* o = (half*) out;
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { o[i] = (half) x[i]; });
    return 0;
    NS_CATCH
}

int ns_image_uocr_quick_gelu(ns_gpu* g, float* x, int64_t n) {
    NS_TRY
    if (n > 0)
        g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
            const float v = x[i];
            x[i] = v / (1.0f + sycl::exp(-1.702f * v));
        });
    return 0;
    NS_CATCH
}

int ns_image_uocr_patchify(ns_gpu* g, const float* img, int64_t B, int64_t H, int64_t W, int64_t P, float* out) {
    NS_TRY
    if (H % P || W % P) return ns_fail("ns_image_uocr_patchify: sides multiples of the patch");
    const int64_t gh = H / P, gw = W / P, k = 3 * P * P, n = B * gh * gw * k;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0], e = i % k, p = i / k;
        const int64_t b = p / (gh * gw), py = (p / gw) % gh, px = p % gw;
        const int64_t c = e / (P * P), y = (e / P) % P, x = e % P;
        out[i] = img[((b * 3 + c) * H + py * P + y) * W + px * P + x];
    });
    return 0;
    NS_CATCH
}

int ns_image_uocr_im2col3(ns_gpu* g, const float* x, int64_t B, int64_t H, int64_t W, int64_t C, int64_t s, float* out) {
    NS_TRY
    const int64_t ho = (H - 1) / s + 1, wo = (W - 1) / s + 1, k = C * 9, n = B * ho * wo * k;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0], e = i % k, p = i / k;
        const int64_t b = p / (ho * wo), oy = (p / wo) % ho, ox = p % wo;
        const int64_t c = e / 9, ky = (e / 3) % 3, kx = e % 3;
        const int64_t y = oy * s + ky - 1, xx = ox * s + kx - 1;
        out[i] = (y < 0 || y >= H || xx < 0 || xx >= W) ? 0.0f : x[((b * H + y) * W + xx) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_image_uocr_window(ns_gpu* g, const float* x, int64_t B, int64_t H, int64_t W, int64_t C, int64_t w, int back, float* out) {
    NS_TRY
    const int64_t hp = (H + w - 1) / w * w, wp = (W + w - 1) / w * w, nh = hp / w, nw = wp / w;
    // a work-item a (window element, channel) of the windows' layout; back: only those inside the picture
    const int64_t n = B * hp * wp * C;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % C, r = i / C;
        const int64_t wx = r % w, wy = (r / w) % w, win = r / (w * w);
        const int64_t bx = win % nw, by = (win / nw) % nh, b = win / (nw * nh);
        const int64_t y = by * w + wy, xx = bx * w + wx;
        const bool in = y < H && xx < W;
        if (back == 0) out[i] = in ? x[((b * H + y) * W + xx) * C + c] : 0.0f;
        else if (in) out[((b * H + y) * W + xx) * C + c] = x[i];
    });
    return 0;
    NS_CATCH
}

int ns_image_uocr_sam_attn(ns_gpu* g, const float* qkv, int64_t B, int64_t Gh, int64_t Gw, int64_t heads, int64_t D, const float* rh,
                           const float* rw, float* out) {
    NS_TRY
    const int64_t N = Gh * Gw;
    if (N > 4096 || D != 64 || Gh > 64 || Gw > 64) return ns_fail("ns_image_uocr_sam_attn: at most 64 x 64 tokens, heads of 64");
    constexpr int WG = 256, DD = 64;
    const int64_t C = heads * D, row = 3 * C;
    const float scale = 1.0f / sycl::sqrt((float) D);
    // a work-group a (grid, head, query): the scores in local memory, their softmax, then the values' sum (64 dims x 4
    // key groups)
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>((size_t) N), h);
        sycl::local_accessor<float, 1> qv(sycl::range<1>(DD), h);
        sycl::local_accessor<float, 1> rel(sycl::range<1>(128), h);
        sycl::local_accessor<float, 1> red(sycl::range<1>(WG), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (B * heads * N) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int t = (int) it.get_local_id(0);
            const int64_t gi = it.get_group(0);
            const int64_t qi = gi % N, hd = (gi / N) % heads, b = gi / (N * heads);
            const int64_t qy = qi / Gw, qx = qi % Gw;
            const float* base = qkv + b * N * row;
            if (t < DD) qv[t] = base[qi * row + hd * D + t];
            sycl::group_barrier(it.get_group());
            // the query's relative terms: along the height (Gh of them), then the width (Gw)
            if (t < Gh + Gw) {
                const bool hgt = t < Gh;
                const int64_t kk = hgt ? t : t - Gh;
                const float* tab = hgt ? rh + (qy - kk + Gh - 1) * D : rw + (qx - kk + Gw - 1) * D;
                float s = 0.0f;
                for (int d = 0; d < DD; ++d) s += qv[d] * tab[d];
                rel[t] = s;
            }
            sycl::group_barrier(it.get_group());
            float mx = -INFINITY;
            for (int64_t j = t; j < N; j += WG) {
                const float* kr = base + j * row + C + hd * D;
                float s = 0.0f;
                for (int d = 0; d < DD; ++d) s += qv[d] * kr[d];
                s = s * scale + rel[j / Gw] + rel[Gh + j % Gw];
                sc[j] = s;
                mx = sycl::fmax(mx, s);
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t j = t; j < N; j += WG) {
                const float e = sycl::exp(sc[j] - mx);
                sc[j] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const int d = t % DD, part = t / DD;
            float acc = 0.0f;
            for (int64_t j = part; j < N; j += WG / DD) acc += sc[j] * base[j * row + 2 * C + hd * D + d];
            red[t] = acc;
            sycl::group_barrier(it.get_group());
            if (t < DD) {
                float v = 0.0f;
                for (int p = 0; p < WG / DD; ++p) v += red[p * DD + t];
                out[(b * N + qi) * C + hd * D + t] = v / sum;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_image_uocr_q5_0_half(ns_gpu* g, const void* src, int64_t n, void* out) {
    NS_TRY
    if (n % 32) return ns_fail("ns_image_uocr_q5_0_half: whole blocks of 32");
    const uint8_t* s = (const uint8_t*) src;
    half* o = (half*) out;
    // ggml's dequantize_row_q5_0: element j < 16 the low nibble of qs[j] and bit j of qh, j >= 16 the high nibble of
    // qs[j - 16] and bit j
    if (n > 0)
        g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            const int64_t i = id[0], b = i / 32, j = i % 32;
            const uint8_t* blk = s + b * 22;
            const float d = (float) *reinterpret_cast<const half*>(blk);
            const uint32_t qh = (uint32_t) blk[2] | ((uint32_t) blk[3] << 8) | ((uint32_t) blk[4] << 16) | ((uint32_t) blk[5] << 24);
            const uint8_t q = blk[6 + (j % 16)];
            const int lo = j < 16 ? (q & 15) : (q >> 4);
            const int hi = (int) ((qh >> j) & 1) << 4;
            o[i] = (half) ((float) ((lo | hi) - 16) * d);
        });
    return 0;
    NS_CATCH
}

int ns_image_uocr_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols) {
    NS_TRY
    if (rows > 0 && cols > 0)
        g->q.parallel_for(sycl::range<2>((size_t) rows, (size_t) cols), [=](sycl::id<2> id) { dst[id[0] * ds + id[1]] = src[id[0] * ss + id[1]]; });
    return 0;
    NS_CATCH
}

int ns_image_uocr_gather(ns_gpu* g, const float* src, int64_t C, const int32_t* idx, int64_t n, float* out) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) C), [=](sycl::id<2> id) { out[id[0] * C + id[1]] = src[(int64_t) idx[id[0]] * C + id[1]]; });
    return 0;
    NS_CATCH
}

int ns_image_uocr_scatter_add(ns_gpu* g, float* dst, int64_t C, const float* src, const int32_t* idx, const float* w, int64_t n) {
    NS_TRY
    if (n > 0)
        g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) C), [=](sycl::id<2> id) {
            dst[(int64_t) idx[id[0]] * C + id[1]] += w[id[0]] * src[id[0] * C + id[1]];
        });
    return 0;
    NS_CATCH
}

}  // extern "C"
