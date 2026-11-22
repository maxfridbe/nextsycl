// mv.cpp: the inkling2 silo's decode products (silo.h). Activations quantized as ggml's Q8_0; the weights read in
// their GGUF blocks - Q8_0 (a half scale, 32 int8: read as 16 halfwords, the blocks being 2-byte aligned) and NVFP4 (four
// UE4M3 scales, 32 bytes of E2M1 codes: a 64-value block 4-byte aligned) - an int8 dot (dp4a) a group of 32 values against
// the activations' block. A sub-group takes R rows, its lanes the groups of 32 in turn (the activations' quants loaded
// once for the R rows); the lanes' sums are added in a fixed order, so a row's value is the same whatever rows share a
// pass. The routed experts: gate | up and SiTU in one launch (the act quantized for the down products in the
// work-group), the down products of a row's K experts and their weighted sum in another.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstring>
#include <type_traits>

namespace {
using half = sycl::half;
constexpr int SG = 16;
constexpr int MAXT = 16;

inline size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// ---- the weight formats ----
inline float ue4m3(uint32_t x) {
    x &= 0xFF;
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (int) ((x >> 3) & 0xF), m = (int) (x & 0x7);
    return e == 0 ? sycl::ldexp((float) m, -10) : sycl::ldexp((float) (8 + m), e - 11);
}

// four E2M1 codes (a byte each, 0..15) to their doubled values as signed bytes: 0 1 2 3 4 6 8 12, the top bit the sign
inline int fp4x4(uint32_t c) {
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (c >> (8 * i)) & 15u;
        int v = (int) ((0xC8643210u >> ((b & 7u) << 2)) & 15u);
        v = (b & 8u) ? -v : v;
        r |= ((uint32_t) v & 0xFFu) << (8 * i);
    }
    return (int) r;
}

inline int dp4a(int a, int b, int c) { return dpct::dp4a(a, b, c); }

// the activations' quants of row t, group g (32 int8 as 8 words) and scale
struct XQ {
    const int8_t* q;
    const float* d;
    int64_t n;
};

// group g (32 values) of a row of type TY: its words w[8] and the float factors: Q8_0 one scale (f[0]); NVFP4 two (the
// two sub-blocks of 16: words 0-3 and 4-7)
template <int TY>
inline void wgroup(const uint8_t* row, int64_t g, int* w, float* f) {
    if constexpr (TY == 8) {
        const uint8_t* p = row + g * 34;
        uint16_t h;
        std::memcpy(&h, p, 2);
        f[0] = (float) sycl::bit_cast<half>(h);
        const uint16_t* qs = (const uint16_t*) (p + 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = (int) ((uint32_t) qs[2 * i] | ((uint32_t) qs[2 * i + 1] << 16));
    } else {
        const uint8_t* p = row + (g >> 1) * 36;
        const int hf = (int) (g & 1);
        const uint32_t sc = *(const uint32_t*) p;
        const uint32_t* c = (const uint32_t*) (p + 4 + hf * 16);
        const uint32_t c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
        w[0] = fp4x4(c0 & 0x0F0F0F0Fu);
        w[1] = fp4x4(c1 & 0x0F0F0F0Fu);
        w[2] = fp4x4((c0 >> 4) & 0x0F0F0F0Fu);
        w[3] = fp4x4((c1 >> 4) & 0x0F0F0F0Fu);
        w[4] = fp4x4(c2 & 0x0F0F0F0Fu);
        w[5] = fp4x4(c3 & 0x0F0F0F0Fu);
        w[6] = fp4x4((c2 >> 4) & 0x0F0F0F0Fu);
        w[7] = fp4x4((c3 >> 4) & 0x0F0F0F0Fu);
        f[0] = ue4m3(sc >> (16 * hf));
        f[1] = ue4m3(sc >> (16 * hf + 8));
    }
}

// a weight group against an activation group: the float sum
template <int TY>
inline float gdot(const int* w, const float* f, const int* x, float xd) {
    if constexpr (TY == 8) {
        int s = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) s = dp4a(w[i], x[i], s);
        return (float) s * (f[0] * xd);
    } else {
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) s0 = dp4a(w[i], x[i], s0);
#pragma unroll
        for (int i = 4; i < 8; ++i) s1 = dp4a(w[i], x[i], s1);
        return ((float) s0 * f[0] + (float) s1 * f[1]) * xd;
    }
}

inline void xgroup(const int8_t* q, int* x) {
    const sycl::int4* p = (const sycl::int4*) q;
    const sycl::int4 a = p[0], b = p[1];
    x[0] = a.x(); x[1] = a.y(); x[2] = a.z(); x[3] = a.w();
    x[4] = b.x(); x[5] = b.y(); x[6] = b.z(); x[7] = b.w();
}

inline size_t row_bytes(int ty, int64_t n_in) {
    return ty == 8 ? (size_t) (n_in / 32) * 34 : (size_t) (n_in / 64) * 36;
}

// ---- products: R rows a sub-group, NSG sub-groups a work-group ----
constexpr int R = 4, NSG = 4;

struct Job {
    const uint8_t* w;
    const int8_t* xq;
    const float* xd;
    float* y;
    int64_t rows, n_in;
    int64_t blk0;   // its first row block in the launch
};
struct Jobs {
    Job j[8];
    int n;
};

template <int TY, int TT>
void mv_launch(sycl::queue& q, Jobs jobs, int64_t blocks) {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) blocks * SG), sycl::range<1>(NSG * SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t blk = it.get_group(0) * NSG + sg.get_group_linear_id();
        int ji = 0;
#pragma unroll
        for (int i = 1; i < 8; ++i)
            if (i < jobs.n && blk >= jobs.j[i].blk0) ji = i;
        const Job jb = jobs.j[ji];
        const int64_t r0 = (blk - jb.blk0) * R;
        if (r0 >= jb.rows) return;
        const int64_t ng = jb.n_in / 32;
        const size_t rb = row_bytes(TY, jb.n_in);
        float acc[R][TT];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < TT; ++t) acc[r][t] = 0.0f;
        for (int64_t gi = lane; gi < ng; gi += SG) {
            int w[R][8];
            float f[R][2];
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (r0 + r < jb.rows) wgroup<TY>(jb.w + (r0 + r) * rb, gi, w[r], f[r]);
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                int x[8];
                xgroup(jb.xq + t * jb.n_in + gi * 32, x);
                const float xd = jb.xd[t * ng + gi];
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r][t] += gdot<TY>(w[r], f[r], x, xd);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const float v = sycl::reduce_over_group(sg, acc[r][t], sycl::plus<float>());
                if (lane == 0 && r0 + r < jb.rows) jb.y[t * jb.rows + r0 + r] = v;
            }
    });
}

template <int TY>
void mv_t(sycl::queue& q, const Jobs& jobs, int64_t blocks, int64_t T) {
    switch (T) {
        case 1: mv_launch<TY, 1>(q, jobs, blocks); break;
        case 2: mv_launch<TY, 2>(q, jobs, blocks); break;
        case 3: mv_launch<TY, 3>(q, jobs, blocks); break;
        case 4: mv_launch<TY, 4>(q, jobs, blocks); break;
        case 5: mv_launch<TY, 5>(q, jobs, blocks); break;
        case 6: mv_launch<TY, 6>(q, jobs, blocks); break;
        case 7: mv_launch<TY, 7>(q, jobs, blocks); break;
        case 8: mv_launch<TY, 8>(q, jobs, blocks); break;
        default: mv_launch<TY, 16>(q, jobs, blocks); break;
    }
}

// the activations' quants and scales of xq (T rows of n)
inline const int8_t* xq_q(const void* xq) { return (const int8_t*) xq; }
inline const float* xq_d(const void* xq, int64_t T, int64_t n) { return (const float*) ((const uint8_t*) xq + ns_ink2s_xq_scales_at(T, n)); }

// ggml's Q8_0 rounding of a block: d = amax / 127 (kept as half), q = round(x / d)
inline void quant_block(sycl::sub_group sg, int lane, const float* x, int8_t* q, float* d) {
    // 16 lanes, 2 values each
    const float a = x[lane * 2], b = x[lane * 2 + 1];
    const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(a), sycl::fabs(b)), sycl::maximum<float>());
    const float dd = amax / 127.0f;
    const float id = dd != 0.0f ? 1.0f / dd : 0.0f;
    q[lane * 2] = (int8_t) sycl::round(a * id);
    q[lane * 2 + 1] = (int8_t) sycl::round(b * id);
    if (lane == 0) *d = (float) half(dd);
}

}  // namespace

extern "C" {

size_t ns_ink2s_xq_scales_at(int64_t T, int64_t n) {
    return align256((size_t) (T * n));
}
size_t ns_ink2s_xq_bytes(int64_t T, int64_t n) {
    return ns_ink2s_xq_scales_at(T, n) + align256((size_t) (T * (n / 32)) * 4);
}

int ns_ink2s_xq(ns_gpu* g, const float* x, void* xq, int64_t T, int64_t n) {
    NS_TRY
    if (n % 32) return ns_fail("ns_ink2s_xq: rows of whole blocks of 32");
    const int64_t nb = T * (n / 32);
    int8_t* q = (int8_t*) xq;
    float* d = (float*) ((uint8_t*) xq + ns_ink2s_xq_scales_at(T, n));
    if (nb > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((nb + 3) / 4) * 4 * SG), sycl::range<1>(4 * SG)),
                                  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int64_t b = it.get_group(0) * 4 + sg.get_group_linear_id();
        if (b >= nb) return;
        quant_block(sg, (int) sg.get_local_linear_id(), x + b * 32, q + b * 32, d + b);
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_rms_xq(ns_gpu* g, const float* x, const float* w, float* y, void* xq, int64_t T, int64_t n, float eps) {
    NS_TRY
    if (n % 32 || n > 4096) return ns_fail("ns_ink2s_rms_xq: rows of whole blocks of 32, at most 4096");
    constexpr int W = 256;
    int8_t* q = (int8_t*) xq;
    float* d = (float*) ((uint8_t*) xq + ns_ink2s_xq_scales_at(T, n));
    if (T > 0) g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> row(sycl::range<1>(4096), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t t = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const float* xr = x + t * n;
            float ss = 0.0f;
            for (int64_t i = tid; i < n; i += W) ss += xr[i] * xr[i];
            const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) n + eps);
            for (int64_t i = tid; i < n; i += W) {
                const float v = xr[i] * inv * w[i];
                row[i] = v;
                if (y) y[t * n + i] = v;
            }
            sycl::group_barrier(it.get_group());
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            float* rp = row.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int64_t b = sgi; b < n / 32; b += W / SG) quant_block(sg, lane, rp + b * 32, q + t * n + b * 32, d + t * (n / 32) + b);
        });
    });
    return 0;
    NS_CATCH
}

int ns_ink2s_mv_supported(int type, int64_t n_in) {
    return (type == 8 && n_in % 32 == 0) || (type == 40 && n_in % 64 == 0);
}

int ns_ink2s_mv_many(ns_gpu* g, int n, const int* ty, const void* const* w, const void* const* xq, float* const* y, const int64_t* rows,
                     const int64_t* n_in, int64_t T) {
    NS_TRY
    if (n < 1 || n > 8 || T < 1 || T > MAXT) return ns_fail("ns_ink2s_mv_many: 1..8 products, 1..16 rows");
    // jobs of one type a launch (the launches in order)
    for (int i = 0; i < n;) {
        Jobs js{};
        int64_t blocks = 0;
        const int t0 = ty[i];
        for (; i < n && ty[i] == t0 && js.n < 8; ++i) {
            if (!ns_ink2s_mv_supported(ty[i], n_in[i])) return ns_fail("ns_ink2s_mv_many: a type or width it does not take");
            Job& j = js.j[js.n++];
            j.w = (const uint8_t*) w[i];
            j.xq = xq_q(xq[i]);
            j.xd = xq_d(xq[i], T, n_in[i]);
            j.y = y[i];
            j.rows = rows[i];
            j.n_in = n_in[i];
            j.blk0 = blocks;
            blocks += (rows[i] + R - 1) / R;
        }
        const int64_t wgs = (blocks + NSG - 1) / NSG;
        if (t0 == 8) mv_t<8>(g->q, js, wgs * NSG, T);
        else mv_t<40>(g->q, js, wgs * NSG, T);
    }
    return 0;
    NS_CATCH
}

int ns_ink2s_mv(ns_gpu* g, int type, const void* w, const void* xq, float* y, int64_t n_in, int64_t n_out, int64_t T) {
    return ns_ink2s_mv_many(g, 1, &type, &w, &xq, &y, &n_out, &n_in, T);
}

int ns_ink2s_moe_gu(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const void* xq, void* act_q, int64_t T,
                    int64_t K, int64_t n_in, int64_t F, float beta, float lb) {
    NS_TRY
    if (!ns_ink2s_mv_supported(type, n_in) || F % 32) return ns_fail("ns_ink2s_moe_gu: a type or shape it does not take");
    // a work-group a (pair, 32 act values): 8 sub-groups of 4 values, each 4 gate and 4 up rows
    constexpr int NS8 = 8, RV = 4;
    const int64_t P = T * K, nblk = F / 32, ng = n_in / 32;
    const size_t rb = row_bytes(type, n_in);
    const int8_t* xqq = xq_q(xq);
    const float* xqd = xq_d(xq, T, n_in);
    int8_t* aq = (int8_t*) act_q;
    float* ad = (float*) ((uint8_t*) act_q + ns_ink2s_xq_scales_at(P, F));
    const uint8_t* wb = (const uint8_t*) w;
    auto run = [&](auto tyc) {
        constexpr int TY = decltype(tyc)::value;
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> av(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (P * nblk) * NS8 * SG), sycl::range<1>(NS8 * SG)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
                const int64_t p = it.get_group(0) / nblk, ab = it.get_group(0) % nblk;
                const int64_t t = p / K, e = ids[p];
                const uint8_t* we = wb + (size_t) e * stride;
                const int64_t i0 = ab * 32 + sgi * RV;
                float acc[2 * RV];
#pragma unroll
                for (int r = 0; r < 2 * RV; ++r) acc[r] = 0.0f;
                for (int64_t gi = lane; gi < ng; gi += SG) {
                    int x[8];
                    xgroup(xqq + t * n_in + gi * 32, x);
                    const float xd = xqd[t * ng + gi];
#pragma unroll
                    for (int r = 0; r < 2 * RV; ++r) {
                        const int64_t row = r < RV ? i0 + r : F + i0 + (r - RV);
                        int wv[8];
                        float f[2];
                        wgroup<TY>(we + row * rb, gi, wv, f);
                        acc[r] += gdot<TY>(wv, f, x, xd);
                    }
                }
                const float s = scale ? scale[e] : 1.0f;
#pragma unroll
                for (int r = 0; r < RV; ++r) {
                    const float gt = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>()) * s;
                    const float up = sycl::reduce_over_group(sg, acc[RV + r], sycl::plus<float>()) * s;
                    const float a = beta * sycl::tanh(gt / beta) * (1.0f / (1.0f + sycl::exp(-gt)));
                    const float u = lb > 0.0f ? lb * sycl::tanh(up / lb) : up;
                    if (lane == 0) av[sgi * RV + r] = a * u;
                }
                sycl::group_barrier(it.get_group());
                if (sgi == 0) {
                    float* ap = av.get_multi_ptr<sycl::access::decorated::no>().get();
                    quant_block(sg, lane, ap, aq + p * F + ab * 32, ad + p * nblk + ab);
                }
            });
        });
    };
    if (type == 8) run(std::integral_constant<int, 8>{});
    else run(std::integral_constant<int, 40>{});
    return 0;
    NS_CATCH
}

int ns_ink2s_moe_down(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const float* wts, const void* act_q,
                      float* y, int64_t T, int64_t K, int64_t F, int64_t n_out) {
    NS_TRY
    if (!ns_ink2s_mv_supported(type, F) || K > 16 || K < 1) return ns_fail("ns_ink2s_moe_down: a type or shape it does not take");
    // a work-group a (row, 4 output rows): a sub-group an expert, its 4 rows; the K sums weighted in order
    constexpr int RD = 4;
    const int64_t P = T * K, ng = F / 32, nrb = (n_out + RD - 1) / RD;
    const size_t rb = row_bytes(type, F);
    const int8_t* aq = xq_q(act_q);
    const float* ad = xq_d(act_q, P, F);
    const uint8_t* wb = (const uint8_t*) w;
    auto run = [&](auto tyc) {
        constexpr int TY = decltype(tyc)::value;
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> part(sycl::range<1>(16 * RD), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * nrb) * K * SG), sycl::range<1>((size_t) K * SG)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                const int lane = (int) sg.get_local_linear_id(), k = (int) sg.get_group_linear_id();
                const int64_t t = it.get_group(0) / nrb, r0 = (it.get_group(0) % nrb) * RD;
                const int64_t p = t * K + k, e = ids[p];
                const uint8_t* we = wb + (size_t) e * stride;
                float acc[RD];
#pragma unroll
                for (int r = 0; r < RD; ++r) acc[r] = 0.0f;
                for (int64_t gi = lane; gi < ng; gi += SG) {
                    int x[8];
                    xgroup(aq + p * F + gi * 32, x);
                    const float xd = ad[p * ng + gi];
#pragma unroll
                    for (int r = 0; r < RD; ++r) {
                        if (r0 + r >= n_out) break;
                        int wv[8];
                        float f[2];
                        wgroup<TY>(we + (r0 + r) * rb, gi, wv, f);
                        acc[r] += gdot<TY>(wv, f, x, xd);
                    }
                }
                const float s = (scale ? scale[e] : 1.0f) * wts[p];
#pragma unroll
                for (int r = 0; r < RD; ++r) {
                    const float v = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
                    if (lane == 0) part[k * RD + r] = v * s;
                }
                sycl::group_barrier(it.get_group());
                if (k == 0 && lane < RD && r0 + lane < n_out) {
                    float v = 0.0f;
                    for (int kk = 0; kk < K; ++kk) v += part[kk * RD + lane];
                    y[t * n_out + r0 + lane] = v;
                }
            });
        });
    };
    if (type == 8) run(std::integral_constant<int, 8>{});
    else run(std::integral_constant<int, 40>{});
    return 0;
    NS_CATCH
}

}  // extern "C"
