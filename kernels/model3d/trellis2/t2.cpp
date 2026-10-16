// t2.cpp: the TRELLIS.2 engine's own kernels - what the shared diffusion kernels (nsd) do not have: the modulated
// layer norm, TRELLIS's per-head RMS norm with its interleaved 3-D rotary embedding, the gated residual adds, the
// sparse convolution's 27-tap gather, the channel-to-space subdivision's gathers, the dense decoder's 3-D pixel
// shuffle, Pixal3D's bilinear feature sampling. Every loop is bounded by its arguments; each kernel is checked
// against the reference stage it serves (reference/trellis2/ref.py, `nextsycl model3d check`).
#include "ns.h"
#include "ns_internal.hpp"
#include "t2.h"

#include <sycl/sycl.hpp>
#include <algorithm>

namespace {
using half = sycl::half;

inline float ld(const void* p, int dt, size_t i) {
    return dt == 0 ? ((const float*) p)[i] : (float) ((const half*) p)[i];
}
inline void st(void* p, int dt, size_t i, float v) {
    if (dt == 0) ((float*) p)[i] = v;
    else ((half*) p)[i] = (half) v;
}
bool dt_ok(int dt) { return dt == 0 || dt == 1; }
}  // namespace

extern "C" {

int ns_model3d_trellis2_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                               const float* scale, const float* shift, void* out, int out_dt) {
    NS_TRY
    constexpr int WG = 256, PER = 32;            // a row of up to 8,192 values held in registers
    if (C > (int64_t) WG * PER || C <= 0) return ns_fail("ns_model3d_trellis2_ln_mod: rows of 1..8192");
    if (!dt_ok(x_dt) || !dt_ok(out_dt)) return ns_fail("ns_model3d_trellis2_ln_mod: float32 or half");
    if (M <= 0) return 0;
    // launches of at most 2^22 rows (a range of rows x 256 work-items stays within an int)
    constexpr int64_t kRows = int64_t(1) << 22;
    for (int64_t r0 = 0; r0 < M; r0 += kRows) {
    const int64_t rows = std::min(kRows, M - r0);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const size_t r = r0 + it.get_group(0);
        const int l = (int) it.get_local_id(0);
        float v[PER];
        float s = 0.f;
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * WG;
            v[i] = c < C ? ld(x, x_dt, r * C + c) : 0.f;
            s += v[i];
        }
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float q = 0.f;
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * WG;
            if (c < C) q += (v[i] - mean) * (v[i] - mean);
        }
        const float var = sycl::reduce_over_group(it.get_group(), q, sycl::plus<float>()) / (float) C;
        const float rs = sycl::rsqrt(var + eps);
        for (int i = 0; i < PER; ++i) {
            const int64_t c = l + (int64_t) i * WG;
            if (c >= C) break;
            float y = (v[i] - mean) * rs;
            if (weight) y = y * weight[c] + (bias ? bias[c] : 0.f);
            if (scale) y = y * (1.f + scale[c]);
            if (shift) y += shift[c];
            st(out, out_dt, r * C + c, y);
        }
    });
    }
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_qk_norm_rope(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* gamma, const float* cs) {
    NS_TRY
    if (D != 128) return ns_fail("ns_model3d_trellis2_qk_norm_rope: heads of 128");
    if (stride < H * D) return ns_fail("ns_model3d_trellis2_qk_norm_rope: a stride under a row");
    if (M <= 0 || H <= 0) return 0;
    half* xh = (half*) x;
    // a sub-group of 32 a head: lane l holds features 4l .. 4l + 3 (pairs 2l and 2l + 1)
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * H * 32)), sycl::range<1>(32)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t grp = (int64_t) it.get_group(0), m = grp / H, h = grp % H;
        const int l = (int) it.get_local_id(0);
        half* p = xh + m * stride + h * D + 4 * l;
        float v[4];
        float ss = 0.f;
        for (int i = 0; i < 4; ++i) {
            v[i] = (float) p[i];
            ss += v[i] * v[i];
        }
        ss = sycl::reduce_over_group(it.get_sub_group(), ss, sycl::plus<float>());
        const float k = sycl::sqrt((float) D) / sycl::fmax(sycl::sqrt(ss), 1e-12f);
        for (int i = 0; i < 4; ++i) v[i] = v[i] * k * gamma[h * D + 4 * l + i];
        if (cs) {
            for (int pr = 0; pr < 2; ++pr) {
                const float* c = cs + (m * (D / 2) + 2 * l + pr) * 2;
                const float a = v[2 * pr], b = v[2 * pr + 1];
                v[2 * pr] = a * c[0] - b * c[1];
                v[2 * pr + 1] = a * c[1] + b * c[0];
            }
        }
        for (int i = 0; i < 4; ++i) p[i] = (half) v[i];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* h, int h_dt, const float* gate) {
    NS_TRY
    if (!dt_ok(h_dt)) return ns_fail("ns_model3d_trellis2_gate_add: float32 or half");
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float v = ld(h, h_dt, i);
        x[i] += gate ? v * gate[i % (size_t) C] : v;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_add_row(ns_gpu* g, float* x, int64_t M, int64_t C, const float* row) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) { x[id[0]] += row[id[0] % (size_t) C]; });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_convert(ns_gpu* g, const void* src, int src_dt, int64_t n, void* dst, int dst_dt) {
    NS_TRY
    if (!dt_ok(src_dt) || !dt_ok(dst_dt)) return ns_fail("ns_model3d_trellis2_convert: float32 or half");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { st(dst, dst_dt, i[0], ld(src, src_dt, i[0])); });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_silu(ns_gpu* g, void* x, int dt, int64_t n) {
    NS_TRY
    if (!dt_ok(dt)) return ns_fail("ns_model3d_trellis2_silu: float32 or half");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = ld(x, dt, i[0]);
        st(x, dt, i[0], v / (1.f + sycl::exp(-v)));
    });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_add(ns_gpu* g, void* x, int x_dt, const void* y, int y_dt, int64_t n) {
    NS_TRY
    if (!dt_ok(x_dt) || !dt_ok(y_dt)) return ns_fail("ns_model3d_trellis2_add: float32 or half");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { st(x, x_dt, i[0], ld(x, x_dt, i[0]) + ld(y, y_dt, i[0])); });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_gather27(ns_gpu* g, const void* feats, int64_t C, const int32_t* nmap, int64_t row0, int64_t rows, void* out) {
    NS_TRY
    if (rows <= 0 || C <= 0) return 0;
    if (C % 8) return ns_fail("ns_model3d_trellis2_gather27: channels a multiple of 8");
    const sycl::vec<uint32_t, 4>* f = (const sycl::vec<uint32_t, 4>*) feats;   // 8 halves a load
    sycl::vec<uint32_t, 4>* o = (sycl::vec<uint32_t, 4>*) out;
    const int64_t C8 = C / 8, n = rows * 27 * C8;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t c = i % C8, t = (i / C8) % 27, r = i / (27 * C8);
        const int32_t s = nmap[(row0 + r) * 27 + t];
        o[i] = s >= 0 ? f[(int64_t) s * C8 + c] : sycl::vec<uint32_t, 4>(0u);
    });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_gather_rows(ns_gpu* g, const void* src, int dt, int64_t C, const int32_t* idx, int64_t n, void* out) {
    NS_TRY
    if (!dt_ok(dt)) return ns_fail("ns_model3d_trellis2_gather_rows: float32 or half");
    if (n <= 0 || C <= 0) return 0;
    const size_t e = dt == 0 ? 4 : 2;
    if ((C * (int64_t) e) % 16 == 0) {
        const int64_t V = C * (int64_t) e / 16;
        const sycl::vec<uint32_t, 4>* s = (const sycl::vec<uint32_t, 4>*) src;
        sycl::vec<uint32_t, 4>* o = (sycl::vec<uint32_t, 4>*) out;
        g->q.parallel_for(sycl::range<1>((size_t) (n * V)), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0];
            o[i] = s[(int64_t) idx[i / V] * V + i % V];
        });
    } else {
        g->q.parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0];
            st(out, dt, (size_t) i, ld(src, dt, (size_t) ((int64_t) idx[i / C] * C + i % C)));
        });
    }
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_c2s_skip(ns_gpu* g, const float* x, int64_t Cin, const int32_t* parent, const int32_t* sub, int64_t n, int64_t Cout,
                                 int64_t r, float* out) {
    NS_TRY
    if (n <= 0 || Cout <= 0) return 0;
    if (r <= 0 || 8 * Cout != Cin * r) return ns_fail("ns_model3d_trellis2_c2s_skip: 8 Cout must be Cin r");
    g->q.parallel_for(sycl::range<1>((size_t) (n * Cout)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], j = i / Cout, c = i % Cout;
        out[i] = x[(int64_t) parent[j] * Cin + ((int64_t) sub[j] * Cout + c) / r];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_pixel_shuffle3d(ns_gpu* g, const void* x, int64_t D, int64_t C, void* out) {
    NS_TRY
    if (D <= 0 || C <= 0) return 0;
    const half* xi = (const half*) x;
    half* o = (half*) out;
    const int64_t D2 = 2 * D;
    g->q.parallel_for(sycl::range<1>((size_t) (D2 * D2 * D2 * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0];
        const int64_t c = i % C, v = i / C;
        const int64_t k2 = v % D2, j2 = (v / D2) % D2, i2 = v / (D2 * D2);
        const int64_t s = (i2 % 2) * 4 + (j2 % 2) * 2 + (k2 % 2);
        o[i] = xi[(((i2 / 2) * D + j2 / 2) * D + k2 / 2) * 8 * C + c * 8 + s];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_trellis2_grid_sample2d(ns_gpu* g, const float* feat, int64_t Hf, int64_t Wf, int64_t C, const float* uv, int64_t n, float* out,
                                      int64_t stride) {
    NS_TRY
    if (n <= 0 || C <= 0) return 0;
    if (stride < C) return ns_fail("ns_model3d_trellis2_grid_sample2d: a stride under the channels");
    g->q.parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], j = i / C, c = i % C;
        // grid_sample, align_corners=False: pixel = ((u + 1) W - 1) / 2; padding_mode border: clipped to the map
        const float fx = sycl::clamp(((uv[2 * j] + 1.f) * (float) Wf - 1.f) * 0.5f, 0.f, (float) (Wf - 1));
        const float fy = sycl::clamp(((uv[2 * j + 1] + 1.f) * (float) Hf - 1.f) * 0.5f, 0.f, (float) (Hf - 1));
        const float x0f = sycl::floor(fx), y0f = sycl::floor(fy);
        const int64_t x0 = (int64_t) x0f, y0 = (int64_t) y0f;
        const float lx = fx - x0f, ly = fy - y0f;
        float acc = 0.f;
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const int64_t yy = y0 + dy, xx = x0 + dx;
                if (yy < 0 || yy >= Hf || xx < 0 || xx >= Wf) continue;
                const float w = (dy ? ly : 1.f - ly) * (dx ? lx : 1.f - lx);
                acc += w * feat[(yy * Wf + xx) * C + c];
            }
        }
        out[j * stride + c] = acc;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
