// hi3d.cpp: the Hi3D engine's own kernels - what the shared diffusion kernels (nsd) do not have: the modulated layer
// norm, the self-attention's rotary embedding then per-head RMS norm, the gated residual adds, the sparse convolution's
// 27-tap gather and the row gathers, the paired downsample's mean, PointGroupNorm (its sums and its normalisation with
// Swish), PTv3's window attention, the children's neighbour maps. Every loop is bounded by its arguments; each kernel is
// checked against the reference stage it serves (reference/hi3d/ref.py, `nextsycl model3d check`). The layer norm,
// gate add, convert, gathers are the trellis2 engine's (kernels/model3d/trellis2/t2.cpp), copied.
#include "ns.h"
#include "ns_internal.hpp"
#include "hi3d.h"

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

int ns_model3d_hi3d_ln_mod(ns_gpu* g, const void* x, int x_dt, int64_t M, int64_t C, float eps, const float* weight, const float* bias,
                           const float* scale, const float* shift, void* out, int out_dt) {
    NS_TRY
    constexpr int WG = 256, PER = 32;            // a row of up to 8,192 values held in registers
    if (C > (int64_t) WG * PER || C <= 0) return ns_fail("ns_model3d_hi3d_ln_mod: rows of 1..8192");
    if (!dt_ok(x_dt) || !dt_ok(out_dt)) return ns_fail("ns_model3d_hi3d_ln_mod: float32 or half");
    if (M <= 0) return 0;
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

int ns_model3d_hi3d_rope_rms(ns_gpu* g, void* x, int64_t M, int64_t H, int64_t D, int64_t stride, const float* cs, const float* gamma) {
    NS_TRY
    constexpr int SG = 32, MAXP = 4;   // a sub-group a head, up to 4 feature pairs a lane: heads of up to 256
    if (D % 2 || D <= 0 || D / 2 > SG * MAXP) return ns_fail("ns_model3d_hi3d_rope_rms: even heads of up to 256");
    if (stride < H * D) return ns_fail("ns_model3d_hi3d_rope_rms: a stride under a row");
    if (M <= 0 || H <= 0) return 0;
    half* xh = (half*) x;
    const int64_t P = D / 2;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (M * H * SG)), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int64_t grp = (int64_t) it.get_group(0), m = grp / H, h = grp % H;
        const int l = (int) it.get_local_id(0);
        half* p = xh + m * stride + h * D;
        float a[MAXP], b[MAXP];
        float ss = 0.f;
        for (int i = 0; i < MAXP; ++i) {
            const int64_t pr = l + (int64_t) i * SG;
            a[i] = b[i] = 0.f;
            if (pr >= P) continue;
            const float x0 = (float) p[2 * pr], x1 = (float) p[2 * pr + 1];
            const float* c = cs + (m * P + pr) * 2;
            a[i] = x0 * c[0] - x1 * c[1];
            b[i] = x0 * c[1] + x1 * c[0];
            ss += a[i] * a[i] + b[i] * b[i];
        }
        ss = sycl::reduce_over_group(it.get_sub_group(), ss, sycl::plus<float>());
        const float k = sycl::sqrt((float) D) / sycl::fmax(sycl::sqrt(ss), 1e-12f);
        for (int i = 0; i < MAXP; ++i) {
            const int64_t pr = l + (int64_t) i * SG;
            if (pr >= P) continue;
            p[2 * pr] = (half) (a[i] * k * gamma[h * D + 2 * pr]);
            p[2 * pr + 1] = (half) (b[i] * k * gamma[h * D + 2 * pr + 1]);
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_gate_add(ns_gpu* g, float* x, int64_t M, int64_t C, const void* h, int h_dt, const float* gate) {
    NS_TRY
    if (!dt_ok(h_dt)) return ns_fail("ns_model3d_hi3d_gate_add: float32 or half");
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float v = ld(h, h_dt, i);
        x[i] += gate ? v * gate[i % (size_t) C] : v;
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_add_row(ns_gpu* g, float* x, int64_t M, int64_t C, const float* row) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) { x[id[0]] += row[id[0] % (size_t) C]; });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_convert(ns_gpu* g, const void* src, int src_dt, int64_t n, void* dst, int dst_dt) {
    NS_TRY
    if (!dt_ok(src_dt) || !dt_ok(dst_dt)) return ns_fail("ns_model3d_hi3d_convert: float32 or half");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { st(dst, dst_dt, i[0], ld(src, src_dt, i[0])); });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_act(ns_gpu* g, void* x, int dt, int64_t n, int act) {
    NS_TRY
    if (!dt_ok(dt)) return ns_fail("ns_model3d_hi3d_act: float32 or half");
    if (act < 0 || act > 2) return ns_fail("ns_model3d_hi3d_act: silu, quick_gelu or tanh");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = ld(x, dt, i[0]);
        float y;
        if (act == 0) y = v / (1.f + sycl::exp(-v));
        else if (act == 1) y = v / (1.f + sycl::exp(-1.702f * v));
        else y = sycl::tanh(v);
        st(x, dt, i[0], y);
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_add(ns_gpu* g, void* x, int x_dt, const void* y, int y_dt, int64_t n) {
    NS_TRY
    if (!dt_ok(x_dt) || !dt_ok(y_dt)) return ns_fail("ns_model3d_hi3d_add: float32 or half");
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { st(x, x_dt, i[0], ld(x, x_dt, i[0]) + ld(y, y_dt, i[0])); });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_put_cols(ns_gpu* g, const void* src, int src_dt, int64_t n, int64_t C, void* dst, int dst_dt, int64_t ldd, int64_t off) {
    NS_TRY
    if (!dt_ok(src_dt) || !dt_ok(dst_dt)) return ns_fail("ns_model3d_hi3d_put_cols: float32 or half");
    if (off < 0 || off + C > ldd) return ns_fail("ns_model3d_hi3d_put_cols: columns past the row");
    if (n <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], r = i / C, c = i % C;
        st(dst, dst_dt, (size_t) (r * ldd + off + c), ld(src, src_dt, (size_t) i));
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_gather27(ns_gpu* g, const void* feats, int64_t C, const int32_t* nmap, int64_t row0, int64_t rows, void* out) {
    NS_TRY
    if (rows <= 0 || C <= 0) return 0;
    if (C % 8) return ns_fail("ns_model3d_hi3d_gather27: channels a multiple of 8");
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

int ns_model3d_hi3d_gather_rows(ns_gpu* g, const void* src, int dt, int64_t C, const int32_t* idx, int64_t n, void* out) {
    NS_TRY
    if (!dt_ok(dt)) return ns_fail("ns_model3d_hi3d_gather_rows: float32 or half");
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

int ns_model3d_hi3d_seg_mean(ns_gpu* g, const void* x, int x_dt, int64_t C, const int32_t* ptr, const int32_t* idx, int64_t P, float* out) {
    NS_TRY
    if (!dt_ok(x_dt)) return ns_fail("ns_model3d_hi3d_seg_mean: float32 or half");
    if (P <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (P * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], p = i / C, c = i % C;
        const int32_t a = ptr[p], b = ptr[p + 1];
        float s = 0.f;
        for (int32_t j = a; j < b; ++j) s += ld(x, x_dt, (size_t) ((int64_t) idx[j] * C + c));
        out[i] = s / (float) (b - a + 1);   // the zero row counted: scatter_reduce(mean, include_self=True)
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_gn_sums(ns_gpu* g, const float* x, int64_t N, int64_t C, int64_t G, int64_t rows, float* part) {
    NS_TRY
    constexpr int WG = 256, MAXG = 32;
    if (G <= 0 || G > MAXG || C % G) return ns_fail("ns_model3d_hi3d_gn_sums: up to 32 groups dividing the channels");
    if (rows <= 0) return ns_fail("ns_model3d_hi3d_gn_sums: blocks of at least one row");
    if (N <= 0) return 0;
    const int64_t blocks = (N + rows - 1) / rows, CP = C / G;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) blocks * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t b = (int64_t) it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const int64_t r0 = b * rows, r1 = sycl::min(N, r0 + rows);
        float s[MAXG], q[MAXG];
        for (int k = 0; k < MAXG; ++k) s[k] = q[k] = 0.f;
        const int64_t n = (r1 - r0) * C;
        for (int64_t i = l; i < n; i += WG) {
            const float v = x[r0 * C + i];
            const int gi = (int) ((i % C) / CP);
            for (int k = 0; k < MAXG; ++k) {
                if (k == gi) {
                    s[k] += v;
                    q[k] += v * v;
                }
            }
        }
        for (int k = 0; k < G; ++k) {
            const float ts = sycl::reduce_over_group(it.get_group(), s[k], sycl::plus<float>());
            const float tq = sycl::reduce_over_group(it.get_group(), q[k], sycl::plus<float>());
            if (l == 0) {
                part[(b * G + k) * 2] = ts;
                part[(b * G + k) * 2 + 1] = tq;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_gn_swish(ns_gpu* g, const float* x, int64_t N, int64_t C, int64_t G, const float* mean, const float* inv, const float* w,
                             const float* b, void* out, int out_dt) {
    NS_TRY
    if (!dt_ok(out_dt)) return ns_fail("ns_model3d_hi3d_gn_swish: float32 or half");
    if (G <= 0 || C % G) return ns_fail("ns_model3d_hi3d_gn_swish: groups dividing the channels");
    if (N <= 0) return 0;
    const int64_t CP = C / G;
    g->q.parallel_for(sycl::range<1>((size_t) (N * C)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], c = i % C, k = c / CP;
        const float y = (x[i] - mean[k]) * inv[k] * w[c] + b[c];
        st(out, out_dt, (size_t) i, y / (1.f + sycl::exp(-y)));
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_win_attn(ns_gpu* g, const void* qkv, int64_t W, int64_t S, int64_t H, int64_t D, float scale, void* out) {
    NS_TRY
    constexpr int MS = 64, MD = 64;
    if (S <= 0 || S > MS || D <= 0 || D > MD) return ns_fail("ns_model3d_hi3d_win_attn: windows of up to 64 rows, heads of up to 64");
    if (W <= 0 || H <= 0) return 0;
    const half* x = (const half*) qkv;
    half* o = (half*) out;
    const int64_t C = H * D;
    // a work-group a (window, head), a work-item a query row; the window's keys and values in local memory
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> kv(sycl::range<1>(2 * MS * MD), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (W * H * MS)), sycl::range<1>(MS)), [=](sycl::nd_item<1> it) {
            const int64_t grp = (int64_t) it.get_group(0), w = grp / H, hd = grp % H;
            const int i = (int) it.get_local_id(0);
            for (int j = i; j < S * D; j += MS) {
                const int64_t r = w * S + j / D, d = j % D;
                kv[j] = (float) x[r * 3 * C + C + hd * D + d];
                kv[MS * MD + j] = (float) x[r * 3 * C + 2 * C + hd * D + d];
            }
            sycl::group_barrier(it.get_group());
            if (i >= S) return;
            const int64_t r = w * S + i;
            float q[MD];
            for (int d = 0; d < D; ++d) q[d] = (float) x[r * 3 * C + hd * D + d] * scale;
            float sc[MS];
            float mx = -3.0e38f;
            for (int j = 0; j < S; ++j) {
                float a = 0.f;
                for (int d = 0; d < D; ++d) a += q[d] * kv[j * D + d];
                sc[j] = a;
                mx = sycl::fmax(mx, a);
            }
            float sum = 0.f;
            for (int j = 0; j < S; ++j) {
                sc[j] = sycl::exp(sc[j] - mx);
                sum += sc[j];
            }
            for (int d = 0; d < D; ++d) {
                float a = 0.f;
                for (int j = 0; j < S; ++j) a += sc[j] * kv[MS * MD + j * D + d];
                o[r * C + hd * D + d] = (half) (a / sum);
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_child_nmap(ns_gpu* g, const int32_t* pnmap, const int32_t* kids, const int32_t* parent, const int32_t* sub, int64_t n, int32_t* out) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (n * 27)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], j = i / 27;
        const int t = (int) (i % 27);
        const int s = sub[j];
        // the child's offset along each axis (-1, 0, 1) from its own sub-cell: the parent's tap and the sub-cell there
        const int tx = (s >> 2) + t / 9 - 1, ty = ((s >> 1) & 1) + (t / 3) % 3 - 1, tz = (s & 1) + t % 3 - 1;
        const int px = tx < 0 ? -1 : tx / 2, py = ty < 0 ? -1 : ty / 2, pz = tz < 0 ? -1 : tz / 2;
        const int q = pnmap[(int64_t) parent[j] * 27 + (px + 1) * 9 + (py + 1) * 3 + (pz + 1)];
        out[i] = q < 0 ? -1 : kids[(int64_t) q * 8 + ((tx & 1) << 2 | (ty & 1) << 1 | (tz & 1))];
    });
    return 0;
    NS_CATCH
}

int ns_model3d_hi3d_copy_heads(ns_gpu* g, const void* src, int64_t ss, int64_t sd, int64_t M, int64_t H, void* dst, int64_t ds, int64_t dd, float scale) {
    NS_TRY
    if (sd <= 0 || dd <= 0 || ss < H * sd || ds < H * dd) return ns_fail("ns_model3d_hi3d_copy_heads: rows narrower than their heads");
    if (M <= 0 || H <= 0) return 0;
    const half* s = (const half*) src;
    half* d = (half*) dst;
    g->q.parallel_for(sycl::range<1>((size_t) (M * H * dd)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id[0], f = i % dd, h = (i / dd) % H, m = i / (dd * H);
        d[m * ds + h * dd + f] = f < sd ? (half) ((float) s[m * ss + h * sd + f] * scale) : (half) 0.f;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
