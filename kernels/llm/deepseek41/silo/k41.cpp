// k41.cpp: DeepSeek-V4.1's own steps (ds41.h): the model's low-precision key simulations, the mHC collapse by the
// previous half's mix, the Engram's rows and gate, the indexer at ratio 1 and 2 with its candidate blocks.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdlib>
#include <cmath>
#include <cstdint>

namespace {

// 2^ceil(log2 x), x > 0 (kernel.py's fast_round_scale)
inline float pow2_ceil(float x) {
    int e;
    const float m = sycl::frexp(x, &e);
    if (m == 0.5f) e -= 1;
    return sycl::ldexp(1.0f, e);
}

// the nearest E4M3FN value (ties to even), |x| <= 448
inline float e4m3_round(float x) {
    const float a = sycl::fabs(x);
    if (a == 0.f) return x;
    int e;
    sycl::frexp(a, &e);   // a = m 2^e, m in [0.5, 1): the binade's exponent is e - 1
    int be = e - 1;
    if (be < -6) be = -6;   // the subnormals' step
    const float step = sycl::ldexp(1.0f, be - 3);
    float q = sycl::rint(a / step) * step;
    q = sycl::fmin(q, 448.f);
    return x < 0 ? -q : q;
}

// the nearest E2M1 value (ties to the even code), |y| <= 6
inline float e2m1_round(float y) {
    const float a = sycl::fabs(y);
    float q;
    if (a <= 0.25f) q = 0.f;
    else if (a < 0.75f) q = 0.5f;
    else if (a <= 1.25f) q = 1.f;
    else if (a < 1.75f) q = 1.5f;
    else if (a <= 2.5f) q = 2.f;
    else if (a < 3.5f) q = 3.f;
    else if (a <= 5.f) q = 4.f;
    else q = 6.f;
    return y < 0 ? -q : q;
}

// E4M3 byte -> float (exp 0: subnormal m 2^-9; 0x7f NaN - absent from the tables here)
inline float e4m3_f(uint8_t b) {
    const int a = b & 0x7f, ex = a >> 3, m = a & 7;
    const float v = ex == 0 ? (float) m * 0.001953125f : sycl::ldexp(1.0f + (float) m * 0.125f, ex - 7);
    return (b & 0x80) ? -v : v;
}
inline float e8m0_f(uint8_t b) {
    return sycl::bit_cast<float>(b == 0 ? 0x00400000u : ((uint32_t) b << 23));
}
inline float bf16_round(float x) {
    uint32_t u = sycl::bit_cast<uint32_t>(x);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return sycl::bit_cast<float>(u);
}

// quant_sim with a lane an element and a sub-group a block (block 16 or 32): the same arithmetic as a work-item a block,
// without its serial chain (a decode pass's blocks are few: the serial form took ~19 us a call)
template <int B>
void quant_sim_sg(sycl::queue& q, float* x, int64_t nb, int kind) {
    constexpr int WG = 128;
    const int64_t n = nb * B;
    q.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(B)]] {
        const auto sg = it.get_sub_group();
        const int64_t i = it.get_global_id(0);
        const bool in = i < n;
        const float v = in ? x[i] : 0.f;
        const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
        if (!in) return;
        if (kind == 0) {
            const float s = pow2_ceil(sycl::fmax(amax, 1e-4f) / 448.f);
            x[i] = e4m3_round(sycl::fmin(sycl::fmax(v / s, -448.f), 448.f)) * s;
        } else {
            const float s = kind == 1 ? pow2_ceil(sycl::fmax(amax, 6.f * 1.1754944e-38f) / 6.f) : e4m3_round(sycl::fmax(amax, 6.f * 0.001953125f) / 6.f);
            x[i] = e2m1_round(sycl::fmin(sycl::fmax(v / s, -6.f), 6.f)) * s;
        }
    });
}


// The indexer's scores on XMX (deepseek4's ns_ds4_idx_score_dec, copied for 32 heads and a ratio): a work-group a row
// and 8 TB x 8 keys, the row's queries packed in local memory as B (fp16, VNNI [D/2][32][2]), the keys straight from
// their fp16 cache as A tiles (8 keys x 16 dims); relu, the head weights and the sum over the heads in registers. The
// scalar form's work grew with the context x the rows: a 32K prompt's indexer layers its largest cost after the experts.
namespace jmi = sycl::ext::oneapi::experimental::matrix;
constexpr int IXD = 128, IXH = 32, IXM = 8, IXSG = 8, IXTB = 4;

template <typename T>
inline auto ix_lp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto ix_gp(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

void idx_score_xmx(sycl::queue& q, const float* qf, const float* w, const sycl::half* kc, float* score, int64_t T, int64_t n, int64_t ld, int64_t pos0,
                   int64_t ratio) {
    constexpr int64_t per_wg = IXSG * IXTB * IXM;
    const int64_t wgs = (ld + per_wg - 1) / per_wg;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> qp(sycl::range<1>(IXD * IXH), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * wgs) * IXSG * 16), sycl::range<1>(IXSG * 16)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int64_t gi = it.get_group(0), t = gi / wgs, wg0 = (gi % wgs) * per_wg;
            sycl::half* qs = qp.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int i = tid; i < IXD * IXH; i += IXSG * 16) {
                const int hh = i / IXD, k = i % IXD;
                qs[(k / 2) * (IXH * 2) + hh * 2 + (k & 1)] = (sycl::half) qf[(t * IXH + hh) * IXD + k];
            }
            sycl::group_barrier(it.get_group());
            const int64_t nv = sycl::min<int64_t>(n, (pos0 + t + 1) / ratio);
            float* out = score + t * ld;
            for (int tb = 0; tb < IXTB; ++tb) {
                const int64_t c0 = wg0 + (sgi * IXTB + tb) * IXM;
                if (c0 >= ld) break;
                if (c0 >= nv) {
                    if (lane < IXM && c0 + lane < ld) out[c0 + lane] = -INFINITY;
                    continue;
                }
                jmi::joint_matrix<sycl::sub_group, float, jmi::use::accumulator, IXM, 16> acc[IXH / 16];
#pragma unroll
                for (int nt = 0; nt < IXH / 16; ++nt) jmi::joint_matrix_fill(sg, acc[nt], 0.0f);
                // a tile past the cache's keys reads keys n - 8 .. (in bounds; the rows before c0 masked below)
                const int64_t cr = sycl::max<int64_t>(0, sycl::min<int64_t>(c0, n - IXM));
                const sycl::half* kb = kc + cr * IXD;
#pragma unroll
                for (int k = 0; k < IXD / 16; ++k) {
                    jmi::joint_matrix<sycl::sub_group, sycl::half, jmi::use::a, IXM, 16, jmi::layout::row_major> a;
                    jmi::joint_matrix_load(sg, a, ix_gp(kb + k * 16), (size_t) IXD);
#pragma unroll
                    for (int nt = 0; nt < IXH / 16; ++nt) {
                        jmi::joint_matrix<sycl::sub_group, sycl::half, jmi::use::b, 16, 16, jmi::layout::ext_intel_packed> b;
                        jmi::joint_matrix_load(sg, b, ix_lp(qs + (k * 16 / 2) * (IXH * 2) + nt * 16 * 2), (size_t) (IXH * 2));
                        jmi::joint_matrix_mad(sg, acc[nt], a, b, acc[nt]);
                    }
                }
                float sv[IXM] = {};
#pragma unroll
                for (int nt = 0; nt < IXH / 16; ++nt) {
                    const float wh = w[t * IXH + nt * 16 + lane];
                    int e = 0;
                    jmi::joint_matrix_apply(sg, acc[nt], [&](float& x) { sv[e] += sycl::fmax(x, 0.0f) * wh; ++e; });
                }
                const int64_t shift = c0 - cr;
#pragma unroll
                for (int e = 0; e < IXM; ++e) {
                    const float v = sycl::reduce_over_group(sg, sv[e], sycl::plus<float>());
                    const int64_t j = cr + e;
                    if (lane == 0 && e >= shift && j < ld) out[j] = j < nv ? v : -INFINITY;
                }
                if (lane == 0)
                    for (int64_t e = IXM - shift; e < IXM; ++e)
                        if (c0 + e < ld) out[c0 + e] = -INFINITY;
            }
        });
    });
}

// The same scores for the index sources after the candidate source, only over a row's candidate blocks (cand [T][K]
// block ids of 8 keys, ascending): score [T][K x 8] compact, a block's 8 keys one A tile; -inf for keys the row does not
// see. The masked full form scored every key of the context (1M at a 1M window) and kept these.
void idx_score_cand_xmx(sycl::queue& q, const float* qf, const float* w, const sycl::half* kc, const int32_t* cand, int64_t K, float* score, int64_t T,
                        int64_t n, int64_t pos0, int64_t ratio) {
    constexpr int64_t per_wg = IXSG * IXTB;   // blocks a work-group
    const int64_t wgs = (K + per_wg - 1) / per_wg, ld = K * IXM;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> qp(sycl::range<1>(IXD * IXH), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * wgs) * IXSG * 16), sycl::range<1>(IXSG * 16)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            const int64_t gi = it.get_group(0), t = gi / wgs, wb0 = (gi % wgs) * per_wg;
            sycl::half* qs = qp.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int i = tid; i < IXD * IXH; i += IXSG * 16) {
                const int hh = i / IXD, k = i % IXD;
                qs[(k / 2) * (IXH * 2) + hh * 2 + (k & 1)] = (sycl::half) qf[(t * IXH + hh) * IXD + k];
            }
            sycl::group_barrier(it.get_group());
            const int64_t nv = sycl::min<int64_t>(n, (pos0 + t + 1) / ratio);
            const int64_t nb = (n + IXM - 1) / IXM;
            float* out = score + t * ld;
            for (int tb = 0; tb < IXTB; ++tb) {
                const int64_t c = wb0 + sgi * IXTB + tb;
                if (c >= K) break;
                const int32_t b = cand[t * K + c];
                const int64_t j0 = (int64_t) b * IXM;
                if (b < 0 || b >= nb || j0 >= nv) {
                    if (lane < IXM) out[c * IXM + lane] = -INFINITY;
                    continue;
                }
                jmi::joint_matrix<sycl::sub_group, float, jmi::use::accumulator, IXM, 16> acc[IXH / 16];
#pragma unroll
                for (int nt = 0; nt < IXH / 16; ++nt) jmi::joint_matrix_fill(sg, acc[nt], 0.0f);
                // the last block may run past the cache's keys: its tile read from n - 8 (in bounds), shifted back below
                const int64_t cr = sycl::max<int64_t>(0, sycl::min<int64_t>(j0, n - IXM));
                const sycl::half* kb = kc + cr * IXD;
#pragma unroll
                for (int k = 0; k < IXD / 16; ++k) {
                    jmi::joint_matrix<sycl::sub_group, sycl::half, jmi::use::a, IXM, 16, jmi::layout::row_major> a;
                    jmi::joint_matrix_load(sg, a, ix_gp(kb + k * 16), (size_t) IXD);
#pragma unroll
                    for (int nt = 0; nt < IXH / 16; ++nt) {
                        jmi::joint_matrix<sycl::sub_group, sycl::half, jmi::use::b, 16, 16, jmi::layout::ext_intel_packed> bm;
                        jmi::joint_matrix_load(sg, bm, ix_lp(qs + (k * 16 / 2) * (IXH * 2) + nt * 16 * 2), (size_t) (IXH * 2));
                        jmi::joint_matrix_mad(sg, acc[nt], a, bm, acc[nt]);
                    }
                }
                float sv[IXM] = {};
#pragma unroll
                for (int nt = 0; nt < IXH / 16; ++nt) {
                    const float wh = w[t * IXH + nt * 16 + lane];
                    int e = 0;
                    jmi::joint_matrix_apply(sg, acc[nt], [&](float& x) { sv[e] += sycl::fmax(x, 0.0f) * wh; ++e; });
                }
                const int64_t shift = j0 - cr;
#pragma unroll
                for (int e = 0; e < IXM; ++e) {
                    const float v = sycl::reduce_over_group(sg, sv[e], sycl::plus<float>());
                    // tile row e is key cr + e, the block's key e - shift
                    const int64_t be = e - shift, j = cr + e;
                    if (lane == 0 && be >= 0 && be < IXM) out[c * IXM + be] = j < nv ? v : -INFINITY;
                }
                if (lane == 0)
                    for (int64_t be = IXM - shift; be < IXM; ++be) out[c * IXM + be] = -INFINITY;
            }
        });
    });
}

// deepseek4's YaRN rotation (ds41.cpp's, copied for the fused kernels here)
inline float kv_yarn_ramp(float lo, float hi, int i0) {
    const float y = ((float) (i0 / 2) - lo) / sycl::fmax(0.001f, hi - lo);
    return 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y));
}
inline void kv_yarn_cs(float p, int i0, int n_dims, float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1, float& c,
                       float& s) {
    const float theta_scale = sycl::pow(freq_base, -2.0f / (float) n_dims);
    float theta_extrap = p;
    for (int k = 0; k < i0 / 2; ++k) theta_extrap *= theta_scale;
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = kv_yarn_ramp(corr0, corr1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
    }
    c = sycl::cos(theta) * mscale;
    s = sycl::sin(theta) * mscale;
}
}  // namespace

extern "C" {

int ns_ds41_quant_sim(ns_gpu* g, float* x, int64_t rows, int64_t cols, int64_t block, int kind) {
    NS_TRY
    if (block <= 0 || block > 64 || cols % block != 0 || kind < 0 || kind > 2) return ns_fail("ns_ds41_quant_sim: a block of 1..64 dividing the row");
    const int64_t nb = rows * (cols / block);
    if (block == 16 || block == 32) {
        // whole blocks a work-group (128 a multiple of the block): no block spans two sub-groups
        if (block == 16) quant_sim_sg<16>(g->q, x, nb, kind);
        else quant_sim_sg<32>(g->q, x, nb, kind);
        return 0;
    }
    g->q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
        float* v = x + id[0] * block;
        float amax = 0.f;
        for (int64_t i = 0; i < block; ++i) amax = sycl::fmax(amax, sycl::fabs(v[i]));
        if (kind == 0) {
            const float s = pow2_ceil(sycl::fmax(amax, 1e-4f) / 448.f);
            for (int64_t i = 0; i < block; ++i) v[i] = e4m3_round(sycl::fmin(sycl::fmax(v[i] / s, -448.f), 448.f)) * s;
        } else {
            const float s = kind == 1 ? pow2_ceil(sycl::fmax(amax, 6.f * 1.1754944e-38f) / 6.f) : e4m3_round(sycl::fmax(amax, 6.f * 0.001953125f) / 6.f);
            for (int64_t i = 0; i < block; ++i) v[i] = e2m1_round(sycl::fmin(sycl::fmax(v[i] / s, -6.f), 6.f)) * s;
        }
    });
    return 0;
    NS_CATCH
}

int ns_ds41_hc_collapse(ns_gpu* g, const float* X, const float* pre, float* h, int64_t T, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, C), [=](sycl::id<2> id) {
        const int64_t t = id[0], k = id[1];
        float s = 0.f;
        for (int i = 0; i < 4; ++i) s += pre[t * 4 + i] * X[(t * 4 + i) * C + k];
        h[t * C + k] = s;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_engram_rows(ns_gpu* g, const uint8_t* raw, float* out, int64_t n, int64_t hd) {
    NS_TRY
    if (hd % 32 != 0) return ns_fail("ns_ds41_engram_rows: a row a multiple of 32");
    const int64_t rb = hd + hd / 32;
    g->q.parallel_for(sycl::range<2>(n, hd), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        const uint8_t* row = raw + r * rb;
        out[r * hd + c] = bf16_round(e4m3_f(row[c]) * e8m0_f(row[hd + c / 32]));
    });
    return 0;
    NS_CATCH
}

int ns_ds41_engram_gate(ns_gpu* g, float* X, const float* kv, const float* qk, int64_t T, int64_t C, float eps) {
    NS_TRY
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(T * 4 * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t ts = it.get_group(0), t = ts / 4, s = ts % 4;
        const int lid = it.get_local_id(0);
        float* x = X + (t * 4 + s) * C;
        const float* key = kv + t * 5 * C + s * C;
        const float* val = kv + t * 5 * C + 4 * C;
        const float* w = qk + s * C;
        float dot = 0.f, xx = 0.f, kk = 0.f;
        for (int64_t c = lid; c < C; c += WG) {
            dot += x[c] * w[c] * key[c];
            xx += x[c] * x[c];
            kk += key[c] * key[c];
        }
        dot = sycl::reduce_over_group(it.get_group(), dot, sycl::plus<float>());
        xx = sycl::reduce_over_group(it.get_group(), xx, sycl::plus<float>());
        kk = sycl::reduce_over_group(it.get_group(), kk, sycl::plus<float>());
        const float d = dot * sycl::rsqrt(xx / (float) C + eps) * sycl::rsqrt(kk / (float) C + eps) * sycl::rsqrt((float) C);
        const float r = sycl::sqrt(sycl::fmax(sycl::fabs(d), 1e-6f));
        const float gate = 1.f / (1.f + sycl::exp(-(d < 0 ? -r : r)));
        sycl::group_barrier(it.get_group());
        for (int64_t c = lid; c < C; c += WG) x[c] += gate * val[c];
    });
    return 0;
    NS_CATCH
}

int ns_ds41_idx_score(ns_gpu* g, const float* q, const float* w, const uint16_t* k, float* score, int64_t T, int64_t H, int64_t D, int64_t n,
                      int64_t ld, int64_t pos0, int64_t ratio) {
    NS_TRY
    constexpr int WG = 128, HM = 32, CH = 16;
    if (H > HM || D % CH != 0 || ratio < 1) return ns_fail("ns_ds41_idx_score: at most 32 heads, dims a multiple of 16");
    // NS_DS41_IDX_XMX=0: the scalar form
    static const bool xmx = [] { const char* v = std::getenv("NS_DS41_IDX_XMX"); return !v || std::atoi(v) != 0; }();
    if (xmx && H == IXH && D == IXD && n >= IXM) {
        idx_score_xmx(g->q, q, w, reinterpret_cast<const sycl::half*>(k), score, T, n, ld, pos0, ratio);
        return 0;
    }
    const sycl::half* kh = reinterpret_cast<const sycl::half*>(k);
    const int64_t per = (ld + WG - 1) / WG;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(HM * 128 > H * D ? (size_t) (H * D) : (size_t) (H * D)), h);
        sycl::local_accessor<float, 1> ws(sycl::range<1>(HM), h);
        h.parallel_for(sycl::nd_range<1>(T * per * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t gi = it.get_group(0), t = gi / per, j = (gi % per) * WG + it.get_local_id(0);
            const int lid = it.get_local_id(0);
            for (int64_t i = lid; i < H * D; i += WG) qs[i] = q[t * H * D + i];
            if (lid < H) ws[lid] = w[t * H + lid];
            sycl::group_barrier(it.get_group());
            if (j >= ld) return;
            const int64_t nv = sycl::min<int64_t>(n, (pos0 + t + 1) / ratio);
            if (j >= nv) {
                score[t * ld + j] = -INFINITY;
                return;
            }
            float acc[HM];
            for (int hh = 0; hh < HM; ++hh) acc[hh] = 0.f;
            const sycl::half* kr = kh + j * D;
            for (int64_t c0 = 0; c0 < D; c0 += CH) {
                float kv[CH];
                for (int c = 0; c < CH; ++c) kv[c] = (float) kr[c0 + c];
                for (int hh = 0; hh < HM; ++hh) {
                    if (hh >= H) break;
                    float s = 0.f;
                    for (int c = 0; c < CH; ++c) s += qs[hh * D + c0 + c] * kv[c];
                    acc[hh] += s;
                }
            }
            float s = 0.f;
            for (int hh = 0; hh < HM; ++hh)
                if (hh < H) s += sycl::fmax(acc[hh], 0.f) * ws[hh];
            score[t * ld + j] = s;
        });
    });
    return 0;
    NS_CATCH
}

int ns_ds41_block_max(ns_gpu* g, const float* score, float* bmax, int64_t T, int64_t n, int64_t ld, int64_t bs, int64_t pos0, int64_t ratio) {
    NS_TRY
    const int64_t nb = (n + bs - 1) / bs;
    g->q.parallel_for(sycl::range<2>(T, nb), [=](sycl::id<2> id) {
        const int64_t t = id[0], b = id[1];
        const int64_t lens = sycl::min<int64_t>(n, (pos0 + t + 1) / ratio);
        float m = -INFINITY;
        for (int64_t j = b * bs; j < sycl::min<int64_t>(n, (b + 1) * bs); ++j) m = sycl::fmax(m, score[t * ld + j]);
        // the block of the row's newest position: in, whatever it scores (it is only partly filled)
        if (lens > 0 && b == (lens - 1) / bs) m = INFINITY;
        bmax[t * nb + b] = m;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_cand_mask(ns_gpu* g, float* score, const int32_t* cand, uint8_t* mark, int64_t T, int64_t n, int64_t ld, int64_t K, int64_t bs) {
    NS_TRY
    const int64_t nb = (n + bs - 1) / bs;
    auto& q = g->q;
    q.memset(mark, 0, (size_t) (T * nb));
    q.parallel_for(sycl::range<2>(T, K), [=](sycl::id<2> id) {
        const int32_t b = cand[id[0] * K + id[1]];
        if (b >= 0 && b < nb) mark[id[0] * nb + b] = 1;
    });
    q.parallel_for(sycl::range<2>(T, n), [=](sycl::id<2> id) {
        const int64_t t = id[0], j = id[1];
        if (!mark[t * nb + j / bs]) score[t * ld + j] = -INFINITY;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_markov_step(ns_gpu* g, const float* logits, const float* bias, int64_t n, int32_t* tok) {
    NS_TRY
    constexpr int WG = 1024;
    g->q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const int lid = it.get_local_id(0);
        float best = -INFINITY;
        int32_t bi = 0;
        for (int64_t i = lid; i < n; i += WG) {
            const float v = logits[i] + bias[i];
            if (v > best) {
                best = v;
                bi = (int32_t) i;
            }
        }
        const float m = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
        // the lowest index of the best (the host's argmax keeps the first)
        const int32_t w = sycl::reduce_over_group(it.get_group(), best == m ? bi : INT32_MAX, sycl::minimum<int32_t>());
        if (lid == 0) *tok = w;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_gather_rows(ns_gpu* g, const float* table, const int32_t* idx, int64_t count, int64_t cols, float* out) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(count, cols), [=](sycl::id<2> id) { out[id[0] * cols + id[1]] = table[(int64_t) idx[id[0]] * cols + id[1]]; });
    return 0;
    NS_CATCH
}

int ns_ds41_idx_score_cand(ns_gpu* g, const float* q, const float* w, const uint16_t* k, const int32_t* cand, int64_t K, float* score, int64_t T, int64_t H,
                           int64_t D, int64_t n, int64_t pos0, int64_t ratio) {
    NS_TRY
    if (H != IXH || D != IXD || n < IXM || ratio < 1) return ns_fail("ns_ds41_idx_score_cand: 32 heads of 128, at least 8 keys");
    idx_score_cand_xmx(g->q, q, w, reinterpret_cast<const sycl::half*>(k), cand, K, score, T, n, pos0, ratio);
    return 0;
    NS_CATCH
}

int ns_ds41_cand_pos(ns_gpu* g, const int32_t* selc, const int32_t* cand, int32_t* sel, int64_t T, int64_t K, int64_t kp, int64_t bs) {
    NS_TRY
    g->q.parallel_for(sycl::range<2>(T, kp), [=](sycl::id<2> id) {
        const int64_t t = id[0], i = id[1];
        const int32_t c = selc[t * kp + i];
        sel[t * kp + i] = cand[t * K + c / bs] * (int32_t) bs + c % bs;
    });
    return 0;
    NS_CATCH
}

// A decode token's routing on the GPU (one row): the router's `used` best of `ne` experts (sqrt(softplus) + bias, ties to
// the lower index; the host's order), their weights (sqrt(softplus) / their sum x scale, summed in rank order), and the
// products' tables built from the layer's slot table (tab: an expert's VRAM payload or 0): gate | up entries (gtab:
// ptrs[2u], meta[2u][10], the lookup lists [33]) and down entries (dtab: ptrs[u], meta[u][10], lists [u][33]), an expert
// row j of dn in ascending id. info: ids[u] (ascending), weights[u] (bits), missing[u] (no VRAM slot: computed by the
// host after it reads this). An expert not in VRAM gets an entry of no columns on a resident payload (`dummy`).
int ns_ds41_route1(ns_gpu* g, const float* logits, const float* bias, int64_t ne, int64_t used, float scale, const uint64_t* tab, uint64_t dummy,
                   int64_t off_gate, int64_t off_up, int64_t off_down, uint8_t* gtab, uint8_t* dtab, int32_t* info, int32_t* ctab) {
    NS_TRY
    if (used < 1 || used > 16 || ne < used || ne > 1024) return ns_fail("ns_ds41_route1: 1..16 of at most 1024 experts");
    constexpr int WG = 1024;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint64_t, 1> best(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const int lid = (int) it.get_local_id(0);
            auto grp = it.get_group();
            float pr = 0.f;
            uint64_t key = 0;
            if (lid < ne) {
                const float z = logits[lid];
                pr = sycl::sqrt(z > 20.f ? z : sycl::log1p(sycl::exp(z)));
                const uint32_t u = sycl::bit_cast<uint32_t>(pr + bias[lid]);
                const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                key = ((uint64_t) k << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) lid);
            }
            for (int r = 0; r < used; ++r) {
                const uint64_t m = sycl::reduce_over_group(grp, key, sycl::maximum<uint64_t>());
                if (lid == 0) best[r] = m;
                if (key == m) key = 0;   // taken
                sycl::group_barrier(grp);
            }
            // the chosen experts' sqrt(softplus): each held by its own item
            sycl::group_barrier(grp);
            if (lid != 0) return;
            int32_t id[16];
            float prv[16];
            for (int r = 0; r < used; ++r) {
                id[r] = (int32_t) (0xFFFFFFFFu - (uint32_t) (best[r] & 0xFFFFFFFFu));
                const float z = logits[id[r]];
                prv[r] = sycl::sqrt(z > 20.f ? z : sycl::log1p(sycl::exp(z)));
            }
            float sum = 0.f;
            for (int r = 0; r < used; ++r) sum += prv[r];
            sum += 1e-20f;
            float wr[16];
            for (int r = 0; r < used; ++r) wr[r] = prv[r] / sum * scale;
            // ascending id (rows of dn, the host's expert order)
            int ord[16];
            for (int r = 0; r < used; ++r) ord[r] = r;
            for (int a = 1; a < used; ++a)
                for (int b = a; b > 0 && id[ord[b]] < id[ord[b - 1]]; --b) {
                    const int t2 = ord[b];
                    ord[b] = ord[b - 1];
                    ord[b - 1] = t2;
                }
            uint64_t* gp = reinterpret_cast<uint64_t*>(gtab);
            int32_t* gm = reinterpret_cast<int32_t*>(gtab + 2 * used * 8);
            int32_t* gl = gm + 2 * used * 10;
            uint64_t* dp = reinterpret_cast<uint64_t*>(dtab);
            int32_t* dm = reinterpret_cast<int32_t*>(dtab + used * 8);
            int32_t* dl = dm + used * 10;
            int nres = 0;
            for (int i = 0; i < 1 + 2 * 16; ++i) gl[i] = 0;
            for (int j = 0; j < used; ++j) {
                const int r = ord[j];
                const uint64_t base = tab[id[r]];
                const bool res = base != 0;
                const uint64_t b = res ? base : dummy;
                info[j] = id[r];
                info[used + j] = sycl::bit_cast<int32_t>(wr[r]);
                // the combine's table (moe_combine: t_ptr [2], ent [u], then the weights)
                ctab[2 + j] = j;
                ctab[2 + used + j] = sycl::bit_cast<int32_t>(wr[r]);
                info[2 * used + j] = res ? 0 : 1;
                for (int w = 0; w < 2; ++w) {
                    const int e = w * (int) used + j;
                    gp[e] = b + (uint64_t) (w == 0 ? off_gate : off_up);
                    int32_t* me = gm + e * 10;
                    me[0] = res ? 1 : 0;
                    me[1] = e;
                    for (int c = 2; c < 10; ++c) me[c] = 0;
                }
                dp[j] = b + (uint64_t) off_down;
                int32_t* me = dm + j * 10;
                me[0] = res ? 1 : 0;
                me[1] = j;
                me[2] = j;
                for (int c = 3; c < 10; ++c) me[c] = 0;
                int32_t* dlj = dl + j * 33;
                for (int c = 0; c < 33; ++c) dlj[c] = 0;
                if (res) {
                    dlj[0] = 1;
                    dlj[1] = j;
                    dlj[2] = 0;
                    ++nres;
                }
            }
            // gate | up: the token's column read by the resident experts' gate and up entries
            int cnt = 0;
            for (int w = 0; w < 2; ++w)
                for (int j = 0; j < used; ++j)
                    if (info[2 * used + j] == 0 && cnt < 16) {
                        gl[1 + 2 * cnt] = w * (int) used + j;
                        gl[2 + 2 * cnt] = 0;
                        ++cnt;
                    }
            gl[0] = cnt;
            ctab[0] = 0;
            ctab[1] = (int32_t) used;
            (void) nres;
        });
    });
    return 0;
    NS_CATCH
}

// dst[pairs[i].idx] = pairs[i].val, n pairs (i64 index, u64 value)
int ns_ds41_scatter_u64(ns_gpu* g, uint64_t* dst, const int64_t* pairs, int64_t n) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { dst[pairs[2 * i[0]]] = (uint64_t) pairs[2 * i[0] + 1]; });
    return 0;
    NS_CATCH
}

// A layer's window key, fused (decode's four launches a layer were four of its ~45): kv [T][D] rms-normed by w (the
// glm5next kernel's sums and order: 256 items a row), its last rd values rotated at its position (YaRN, ds41_rope's),
// the E4M3 simulation in blocks of 32 (quant_sim_sg's, a sub-group a block), then into the fp16 ring slot (pos0 + t) % R.
int ns_ds41_kv_post(ns_gpu* g, float* kv, const float* w, int64_t T, int64_t D, float eps, int64_t rd, int64_t pos0, float freq_base, float freq_scale,
                    float ext_factor, float mscale, float corr0, float corr1, uint16_t* ring, int64_t R) {
    NS_TRY
    constexpr int WG = 256;
    if (D % 32 != 0 || D > 2 * WG || rd % 2 != 0 || rd > D) return ns_fail("ns_ds41_kv_post: a row of at most 512 in blocks of 32");
    sycl::half* r16 = reinterpret_cast<sycl::half*>(ring);
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> y(sycl::range<1>(2 * WG), h);
        h.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t t = it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            float* xr = kv + t * D;
            float s = 0.f;
            for (int64_t c = lid; c < D; c += WG) s += xr[c] * xr[c];
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            const float inv = 1.f / sycl::sqrt(s / (float) D + eps);
            for (int64_t c = lid; c < D; c += WG) y[c] = xr[c] * inv * w[c];
            sycl::group_barrier(it.get_group());
            const int64_t off = D - rd;
            if (lid < rd / 2) {
                const int i0 = 2 * lid;
                float cs, sn;
                kv_yarn_cs((float) (pos0 + t), i0, (int) rd, freq_base, freq_scale, ext_factor, mscale, corr0, corr1, cs, sn);
                const float x0 = y[off + i0], x1 = y[off + i0 + 1];
                y[off + i0] = x0 * cs - x1 * sn;
                y[off + i0 + 1] = x0 * sn + x1 * cs;
            }
            sycl::group_barrier(it.get_group());
            const auto sg = it.get_sub_group();
            for (int64_t c0 = 0; c0 < D; c0 += WG) {
                const int64_t c = c0 + lid;
                const bool in = c < D;
                const float v = in ? y[c] : 0.f;
                const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
                if (in) {
                    const float sc = pow2_ceil(sycl::fmax(amax, 1e-4f) / 448.f);
                    const float q = e4m3_round(sycl::fmin(sycl::fmax(v / sc, -448.f), 448.f)) * sc;
                    xr[c] = q;
                    r16[((pos0 + t) % R) * D + c] = (sycl::half) q;
                }
            }
        });
    });
    return 0;
    NS_CATCH
}

// mHC before a half at decode widths, its input collapsed by the half before's mix: glm5next's ns_hc_pre_fused
// (copied) with h = sum_i pre_in_i X_i in its second launch instead of its own pre - that one's h and norm were
// replaced by ns_ds41_hc_collapse + ns_rms_norm after it (two of a layer's launches a half)
int ns_ds41_hc_pre_in(ns_gpu* g, const float* x, const float* pre_in, const float* fn, const float* scale, const float* base, const float* nw, float* h,
                    float* post, float* comb, float* pre, float* normed, float* part, int64_t T, int64_t C, float eps, float hc_eps, int iters) {
    NS_TRY
    constexpr int WG = 512, M = 24, NB = 32, WA = 256;
    const int64_t n = 4 * C;
    const float heps = hc_eps;
    // 1. the 24 dots and the sum of squares, NB blocks of columns a token (ns_hc_mix's first pass)
    const int64_t chunk = (n + NB - 1) / NB;
    // the 25 sums: across each sub-group by shuffles, then the sub-groups' through local memory - one barrier (25
    // work-group reductions in a row, a barrier each, took most of this kernel's ~20 us)
    constexpr int SGN = WA / 16;
    g->q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(SGN * (M + 1)), hd);
        hd.parallel_for(sycl::nd_range<1>(T * NB * WA, WA), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t grp = it.get_group(0), t = grp / NB, b = grp % NB;
            const float* xr = x + t * n;
            const int lid = it.get_local_id(0);
            const auto sg = it.get_sub_group();
            const int sgi = sg.get_group_id()[0], ln = sg.get_local_id()[0];
            const int64_t c0 = b * chunk, c1 = sycl::min(n, c0 + chunk);
            float ss = 0.f, acc[M];
            for (int o = 0; o < M; ++o) acc[o] = 0.f;
            for (int64_t c = c0 + lid; c < c1; c += WA) {
                const float v = xr[c];
                ss += v * v;
                for (int o = 0; o < M; ++o) acc[o] += fn[o * n + c] * v;
            }
            for (int o = 0; o < M; ++o) {
                const float r = sycl::reduce_over_group(sg, acc[o], sycl::plus<float>());
                if (ln == 0) red[sgi * (M + 1) + o] = r;
            }
            const float r = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
            if (ln == 0) red[sgi * (M + 1) + M] = r;
            sycl::group_barrier(it.get_group());
            if (lid <= M) {
                float v = 0.f;
                for (int k = 0; k < SGN; ++k) v += red[k * (M + 1) + lid];
                part[grp * (M + 1) + lid] = v;
            }
        });
    });
    // 2. per token: the mixes, pre / post / comb, h (by pre_in), rms_norm(h). The 4 x 4 Sinkhorn on the first sub-group, a lane
    // per entry c[d + 4 src] (one work-item doing it all took ~60 us: its divisions in a row): a row's sums over
    // dst are the xor-1/2 shuffles, a column's over src the xor-4/8 ones.
    g->q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> pw(sycl::range<1>(4), hd);
        hd.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t t = it.get_group(0);
            const int lid = it.get_local_id(0);
            const float* xr = x + t * n;
            if (lid < 16) {
                const auto sg = it.get_sub_group();
                const int ln = lid;   // entry d + 4 src
                auto mix = [&](int o) {
                    float v = 0.f;
                    for (int b = 0; b < NB; ++b) v += part[(t * NB + b) * (M + 1) + o];
                    return v;
                };
                float ss = 0.f;
                for (int b = 0; b < NB; ++b) ss += part[(t * NB + b) * (M + 1) + M];
                const float inv = 1.f / sycl::sqrt(ss / (float) n + eps);
                const float e = heps;
                if (ln < 4) {
                    pre[t * 4 + ln] = 1.f / (1.f + sycl::exp(-(mix(ln) * inv * scale[0] + base[ln]))) + e;
                    pw[ln] = pre_in[t * 4 + ln];
                } else if (ln < 8) {
                    post[t * 4 + ln - 4] = 2.f / (1.f + sycl::exp(-(mix(ln) * inv * scale[1] + base[ln])));
                }
                float c = mix(8 + ln) * inv * scale[2] + base[8 + ln];
                auto row = [&](float v, bool mx) {   // over the 4 dst of this src (lanes 4 src .. 4 src + 3)
                    for (int k = 1; k < 4; k <<= 1) {
                        const float o = sycl::permute_group_by_xor(sg, v, k);
                        v = mx ? sycl::fmax(v, o) : v + o;
                    }
                    return v;
                };
                auto col = [&](float v) {            // over the 4 src of this dst (lanes d, d + 4, d + 8, d + 12)
                    for (int k = 4; k < 16; k <<= 1) v += sycl::permute_group_by_xor(sg, v, k);
                    return v;
                };
                // softmax over dst for each src, then + eps
                const float mx = row(c, true);
                c = sycl::exp(c - mx);
                c = c / row(c, false) + e;
                c /= e + col(c);                     // norm_cols
                for (int k = 1; k < iters; ++k) {
                    c /= e + row(c, false);          // norm_rows
                    c /= e + col(c);                 // norm_cols
                }
                comb[t * 16 + ln] = c;
            }
            sycl::group_barrier(it.get_group());
            float s2 = 0.f;
            for (int64_t k = lid; k < C; k += WG) {
                float v = 0.f;
                for (int i = 0; i < 4; ++i) v += pw[i] * xr[i * C + k];
                h[t * C + k] = v;
                s2 += v * v;
            }
            s2 = sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>());
            const float inv2 = 1.f / sycl::sqrt(s2 / (float) C + eps);
            for (int64_t k = lid; k < C; k += WG) normed[t * C + k] = h[t * C + k] * inv2 * nw[k];
        });
    });
    return 0;
    NS_CATCH
}

// The attention's output rotated back and simulated, fused (two launches a layer): x [rows][D] (rows = T tokens x H
// heads, token t at pos0 + t), each head's last rd values rotated back at its position (ds41_rope's inverse, the pair's
// other value from the neighbouring lane), then the E4M3 simulation in blocks of 32 (quant_sim_sg's, a sub-group a block)
int ns_ds41_att_post(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t D, int64_t rd, int64_t pos0, float freq_base, float freq_scale, float ext_factor,
                     float mscale, float corr0, float corr1) {
    NS_TRY
    constexpr int WG = 128;
    if (D % 32 != 0 || rd % 2 != 0 || rd > D) return ns_fail("ns_ds41_att_post: rows in blocks of 32");
    const int64_t n = T * H * D;
    g->q.parallel_for(sycl::nd_range<1>((n + WG - 1) / WG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const auto sg = it.get_sub_group();
        const int64_t i = it.get_global_id(0);
        const bool in = i < n;
        float v = in ? x[i] : 0.f;
        const int64_t c = i % D, off = D - rd;
        const float o = sycl::permute_group_by_xor(sg, v, 1);
        if (in && c >= off) {
            const int i0 = (int) ((c - off) & ~int64_t(1));
            float cs, sn;
            kv_yarn_cs((float) (pos0 + i / (H * D)), i0, (int) rd, freq_base, freq_scale, ext_factor, mscale, corr0, corr1, cs, sn);
            sn = -sn;
            v = (c - off) % 2 == 0 ? v * cs - o * sn : o * sn + v * cs;
        }
        const float amax = sycl::reduce_over_group(sg, sycl::fabs(v), sycl::maximum<float>());
        if (!in) return;
        const float s = pow2_ceil(sycl::fmax(amax, 1e-4f) / 448.f);
        x[i] = e4m3_round(sycl::fmin(sycl::fmax(v / s, -448.f), 448.f)) * s;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
