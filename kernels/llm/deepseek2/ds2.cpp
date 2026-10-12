// ds2.cpp: the deepseek2 engine's own kernels (ds2.h) - YaRN's RoPE as ggml computes it, the MLA latent cache's rows,
// the experts' SwiGLU rows, IQ1_S expanded, and the host's view of tickets.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds2.h"

#include <sycl/sycl.hpp>
#include <cmath>

// ggml's IQ1_S grid (iq1s_grid_gpu: 2048 entries of eight values in {-1, 0, 1}, each + 1 in a nibble)
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

namespace {

// ggml's rope_yarn_ramp and the angle for pair i0 at the extrapolated angle theta_extrap (ggml-cpu ops.cpp)
inline float yarn_ramp(float low, float high, int i0) {
    const float y = ((float) (i0 / 2) - low) / sycl::fmax(0.001f, high - low);
    return 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y));
}
// cos and sin of pair i0 (of n_dims) at position p: ggml's rope cache (the angle by repeated products, as it does)
inline void yarn_cs(float p, int i0, int n_dims, float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1,
                    float& c, float& s) {
    const float theta_scale = sycl::pow(freq_base, -2.0f / (float) n_dims);
    float theta_extrap = p;
    for (int k = 0; k < i0 / 2; ++k) theta_extrap *= theta_scale;
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = yarn_ramp(corr0, corr1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
    }
    c = sycl::cos(theta) * mscale;
    s = sycl::sin(theta) * mscale;
}

constexpr int QK = 256;

// IQ1_S: d, then qs [32] (the grid indices' low bytes), then qh [8] (a sub-block's 4 x 3 high index bits, its
// 3-bit scale, its delta's sign)
struct IQ1SBlock { sycl::half d; uint8_t qs[QK / 8]; uint16_t qh[QK / 32]; };
static_assert(sizeof(IQ1SBlock) == 50, "block_iq1_s is 50 bytes");

// the 8 values of group l of sub-block ib of block b (ggml's dequantize_row_iq1_s)
inline void iq1s_group(const IQ1SBlock* b, int ib, int l, float (&y)[8]) {
    const float d = (float) b->d;
    const uint16_t qh = b->qh[ib];
    const float dl = d * (float) (2 * ((qh >> 12) & 7) + 1);
    const float delta = qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
    // the GPU form of the grid: value j + 1 in a nibble - values 0..3 in the bytes' low nibbles, 4..7 in the high ones
    const uint32_t e = iq1s_grid_gpu[b->qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = dl * ((float) ((int) ((e >> (8 * (j & 3) + 4 * (j >> 2))) & 0xf) - 1) + delta);
}

}  // namespace

extern "C" {

int ns_ds2_rope(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t row, int64_t head, int64_t off, int64_t n_dims, int64_t pos0,
                float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1, float* out, int64_t out_row,
                int64_t out_head) {
    NS_TRY
    if (n_dims % 2 != 0) return ns_fail("ns_ds2_rope: an even rotated width");
    const int64_t pairs = n_dims / 2;
    g->q.parallel_for(sycl::range<3>(T, H, pairs), [=](sycl::id<3> id) {
        const int64_t t = id[0], h = id[1], i = id[2];
        const int i0 = (int) (2 * i);
        float c, s;
        yarn_cs((float) (pos0 + t), i0, (int) n_dims, freq_base, freq_scale, ext_factor, mscale, corr0, corr1, c, s);
        float* v = x + t * row + h * head + off + i0;
        const float x0 = v[0], x1 = v[1];
        const float y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
        v[0] = y0;
        v[1] = y1;
        if (out) {
            float* o = out + t * out_row + h * out_head + i0;
            o[0] = y0;
            o[1] = y1;
        }
    });
    return 0;
    NS_CATCH
}

int ns_ds2_kv_store(ns_gpu* g, const float* kv, const float* w, float eps, uint16_t* cache, int64_t T, int64_t L, int64_t R, int64_t pos0,
                    float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1) {
    NS_TRY
    if (R % 2 != 0) return ns_fail("ns_ds2_kv_store: an even rotated width");
    constexpr int WG = 256;
    sycl::half* c16 = reinterpret_cast<sycl::half*>(cache);
    g->q.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t t = it.get_group(0);
        const int lid = (int) it.get_local_id(0);
        const float* xr = kv + t * (L + R);
        float ss = 0.f;
        for (int64_t i = lid; i < L; i += WG) ss += xr[i] * xr[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float inv = 1.f / sycl::sqrt(ss / (float) L + eps);
        sycl::half* o = c16 + (pos0 + t) * (L + R);
        for (int64_t i = lid; i < L; i += WG) o[i] = (sycl::half) (xr[i] * inv * w[i]);
        for (int64_t i = lid; i < R / 2; i += WG) {
            const int i0 = (int) (2 * i);
            float c, s;
            yarn_cs((float) (pos0 + t), i0, (int) R, freq_base, freq_scale, ext_factor, mscale, corr0, corr1, c, s);
            const float x0 = xr[L + i0], x1 = xr[L + i0 + 1];
            o[L + i0] = (sycl::half) (x0 * c - x1 * s);
            o[L + i0 + 1] = (sycl::half) (x0 * s + x1 * c);
        }
    });
    return 0;
    NS_CATCH
}

int ns_ds2_mla_decode(ns_gpu* g, const float* q, const uint16_t* kb16, const uint16_t* vb16, const uint16_t* cache16, float* out, int64_t T,
                      int64_t H, int64_t pos0, float scale) {
    NS_TRY
    constexpr int L = 512, R = 64, NOPE = 128, V = 128, KL = NOPE + R, CR = L + R;
    constexpr int SG = 16, NSG = 16, WG = SG * NSG, DL = L / SG, DR = R / SG;
    const sycl::half* kb = reinterpret_cast<const sycl::half*>(kb16);
    const sycl::half* vb = reinterpret_cast<const sycl::half*>(vb16);
    const sycl::half* c = reinterpret_cast<const sycl::half*>(cache16);
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qa(sycl::range<1>(L + R + NOPE), h), part(sycl::range<1>(NSG * L), h), red(sycl::range<1>(2 * NSG), h),
            u(sycl::range<1>(L), h);
        h.parallel_for(sycl::nd_range<1>(T * H * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int64_t t = it.get_group(0) / H, hd = it.get_group(0) % H;
            const int lid = (int) it.get_local_id(0), lane = (int) sg.get_local_id()[0], sgi = lid / SG;
            const float* qr = q + (t * H + hd) * KL;
            // q_nope and q_pe into local memory (q_pe after the absorbed part: qa [L .. L + R))
            if (lid < NOPE) qa[L + R + lid] = qr[lid];
            if (lid < R) qa[L + lid] = qr[NOPE + lid];
            sycl::group_barrier(it.get_group());
            // q~ = k_b[h] q_nope: a sub-group a row (its 128 halves read together, 8 a lane), reduced
            using h8 = sycl::vec<sycl::half, 8>;
            for (int r = sgi; r < L; r += NSG) {
                const h8 w = reinterpret_cast<const h8*>(kb + ((size_t) hd * L + r) * NOPE)[lane];
                float a = 0.f;
#pragma unroll
                for (int f = 0; f < 8; ++f) a += (float) w[f] * qa[L + R + lane * 8 + f];
                a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
                if (lane == 0) qa[r] = a;
            }
            sycl::group_barrier(it.get_group());
            float qv[DL], qp[DR];
#pragma unroll
            for (int f = 0; f < DL; ++f) qv[f] = qa[lane * DL + f];
#pragma unroll
            for (int f = 0; f < DR; ++f) qp[f] = qa[L + lane * DR + f];
            // the sub-group's cells sgi, sgi + NSG, ...: an online softmax each
            const int64_t n = pos0 + t + 1;
            float m = -INFINITY, sum = 0.f, acc[DL] = {};
            for (int64_t i = sgi; i < n; i += NSG) {
                const sycl::half* row = c + i * CR;
                float x[DL];
                float d = 0.f;
                const h8* rv = reinterpret_cast<const h8*>(row + lane * DL);
#pragma unroll
                for (int k = 0; k < DL / 8; ++k) {
                    const h8 w = rv[k];
#pragma unroll
                    for (int f = 0; f < 8; ++f) {
                        x[k * 8 + f] = (float) w[f];
                        d += qv[k * 8 + f] * x[k * 8 + f];
                    }
                }
#pragma unroll
                for (int f = 0; f < DR; ++f) d += qp[f] * (float) row[L + lane * DR + f];
                d = sycl::reduce_over_group(sg, d, sycl::plus<float>()) * scale;
                const float mn = sycl::fmax(m, d);
                const float a = sycl::exp(m - mn), b = sycl::exp(d - mn);
                sum = sum * a + b;
#pragma unroll
                for (int f = 0; f < DL; ++f) acc[f] = acc[f] * a + b * x[f];
                m = mn;
            }
            // the sub-groups combined
            if (lane == 0) {
                red[sgi] = m;
                red[NSG + sgi] = sum;
            }
            sycl::group_barrier(it.get_group());
            float mx = -INFINITY;
            for (int k = 0; k < NSG; ++k) mx = sycl::fmax(mx, red[k]);
            const float sc = m == -INFINITY ? 0.f : sycl::exp(m - mx);
#pragma unroll
            for (int f = 0; f < DL; ++f) part[sgi * L + lane * DL + f] = acc[f] * sc;
            sycl::group_barrier(it.get_group());
            float tot = 0.f;
            for (int k = 0; k < NSG; ++k) tot += red[k] == -INFINITY ? 0.f : red[NSG + k] * sycl::exp(red[k] - mx);
            for (int r = lid; r < L; r += WG) {
                float v = 0.f;
                for (int k = 0; k < NSG; ++k) v += part[k * L + r];
                u[r] = v / tot;
            }
            sycl::group_barrier(it.get_group());
            // out = v_b[h] u: a sub-group a row (its 512 halves, 32 a lane in four 16-byte loads)
            for (int r = sgi; r < V; r += NSG) {
                const h8* vr = reinterpret_cast<const h8*>(vb + ((size_t) hd * V + r) * L);
                float a = 0.f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const h8 w = vr[k * SG + lane];
#pragma unroll
                    for (int f = 0; f < 8; ++f) a += (float) w[f] * u[(k * SG + lane) * 8 + f];
                }
                a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
                if (lane == 0) out[(t * H + hd) * V + r] = a;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_ds2_swiglu_rows(ns_gpu* g, const float* gu, float* out, int64_t rows, int64_t f) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(rows, f), [=](sycl::id<2> id) {
        const int64_t r = id[0], i = id[1];
        const float a = gu[r * 2 * f + i], u = gu[r * 2 * f + f + i];
        out[r * f + i] = a / (1.f + sycl::exp(-a)) * u;
    });
    return 0;
    NS_CATCH
}

int ns_ds2_dequant_supported(int type) { return type == 19; }

int ns_ds2_dequant(ns_gpu* g, int type, const void* src, int64_t n, float* dst) {
    NS_TRY
    if (type != 19 || n % QK != 0) return ns_fail("ns_ds2_dequant: IQ1_S, whole blocks");
    const IQ1SBlock* x = static_cast<const IQ1SBlock*>(src);
    // a work-item a group of 8 values: block n / 256, sub-block, group
    g->q.parallel_for(sycl::range<1>(n / 8), [=](sycl::id<1> id) {
        const int64_t i = id[0], b = i / 32;
        const int ib = (int) ((i / 4) % 8), l = (int) (i % 4);
        float y[8];
        iq1s_group(x + b, ib, l, y);
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[i * 8 + j] = y[j];
    });
    return 0;
    NS_CATCH
}

int ns_ds2_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst) {
    NS_TRY
    if (type != 19 || n % QK != 0) return ns_fail("ns_ds2_dequant_f16: IQ1_S, whole blocks");
    const IQ1SBlock* x = static_cast<const IQ1SBlock*>(src);
    using h8 = sycl::vec<sycl::half, 8>;
    h8* o = reinterpret_cast<h8*>(dst);
    g->q.parallel_for(sycl::range<1>(n / 8), [=](sycl::id<1> id) {
        const int64_t i = id[0], b = i / 32;
        const int ib = (int) ((i / 4) % 8), l = (int) (i % 4);
        float y[8];
        iq1s_group(x + b, ib, l, y);
        h8 r;
#pragma unroll
        for (int j = 0; j < 8; ++j) r[j] = (sycl::half) y[j];
        o[i] = r;
    });
    return 0;
    NS_CATCH
}

int ns_ds2_kcopy(ns_gpu* g, void* dst, const void* src, size_t n, int64_t after, int64_t* ticket) {
    NS_TRY
    if (n % 16 != 0) return ns_fail("ns_ds2_kcopy: a multiple of 16 bytes");
    using v4 = sycl::uint4;
    const v4* s = static_cast<const v4*>(src);
    v4* d = static_cast<v4*>(dst);
    std::vector<sycl::event> deps;
    if (after >= 0) deps.push_back(g->ev[after % g->ev.size()]);
    const size_t m = n / 16;
    sycl::event e = g->q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(m), [=](sycl::id<1> i) { d[i] = s[i]; });
    });
    const int64_t t = g->next_ev++;
    g->ev[t % g->ev.size()] = e;
    *ticket = t;
    return 0;
    NS_CATCH
}

int ns_ds2_fill(ns_gpu* g, float* x, float v, int64_t n) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] = v; });
    return 0;
    NS_CATCH
}

int ns_ds2_ticket_wait(ns_gpu* g, int64_t t) {
    NS_TRY
    g->ev[t % g->ev.size()].wait();
    return 0;
    NS_CATCH
}

int ns_ds2_ticket_done(ns_gpu* g, int64_t t) {
    try {
        const auto st = g->ev[t % g->ev.size()].get_info<sycl::info::event::command_execution_status>();
        return st == sycl::info::event_command_status::complete ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

}  // extern "C"
