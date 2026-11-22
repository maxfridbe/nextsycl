// ink.cpp: the inkling2 engine's own kernels (ink.h) - Inkling-2-Mini's weights (Q8_0, BF16, NVFP4 decoded from the
// GGUF block layouts), its MoE (sigmoid router, SiTU), the depth attention and loop controllers, the short
// convolutions, the clipped attention with its relative bias and sink. After llama.cpp's src/models/inkling2.cpp and
// inkling2-loop.cpp; the arithmetic is written from the formats and the graph, not taken from ggml.
#include "ns.h"
#include "ns_internal.hpp"
#include "ink.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstring>

namespace {
using half = sycl::half;
constexpr int SG = 16;

// ---- the block formats ----
// Q8_0: a half scale, then 32 int8 values. NVFP4: 64 values - four UE4M3 scales (one a 16), then 32 bytes of E2M1
// codes, sub-block s's value j in the low nibble of byte 8s + j and value j + 8 in its high nibble; a code is twice
// its E2M1 value (the table), the scale half its UE4M3 value. BF16: the top half of a float32.
constexpr int VB_Q8 = 32, BB_Q8 = 34, VB_FP4 = 64, BB_FP4 = 36;

inline float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 0x7;
    // (1 + m / 8) 2^(e - 7) / 2 = (8 + m) 2^(e - 11); subnormal m 2^-9 / 2
    return e == 0 ? sycl::ldexp((float) m, -10) : sycl::ldexp((float) (8 + m), e - 11);
}

inline float e2m1x2(int c) {
    // 0, 0.5, 1, 1.5, 2, 3, 4, 6 doubled, then the negatives
    constexpr float t[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return t[c & 15];
}

inline float bf16(uint16_t v) {
    return sycl::bit_cast<float>((uint32_t) v << 16);
}

inline float f16(const uint8_t* p) {
    uint16_t b;
    std::memcpy(&b, p, 2);
    return (float) sycl::bit_cast<half>(b);
}

// values and bytes of a block of a type (BF16 / F32: 32 values a "block")
inline int vb(int type) {
    return type == 40 ? VB_FP4 : 32;
}
inline int bb(int type) {
    switch (type) {
        case 0: return 128;
        case 30: return 64;
        case 8: return BB_Q8;
        case 40: return BB_FP4;
        default: return 0;
    }
}

// block b of a row decoded into v (vb(T) values)
template <int T>
inline void decode(const uint8_t* row, int64_t b, float* v) {
    if constexpr (T == 8) {
        const uint8_t* p = row + b * BB_Q8;
        const float d = f16(p);
        const int8_t* q = (const int8_t*) (p + 2);
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = d * (float) q[i];
    } else if constexpr (T == 40) {
        const uint8_t* p = row + b * BB_FP4;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3(p[s]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint8_t c = p[4 + s * 8 + j];
                v[s * 16 + j] = e2m1x2(c & 15) * d;
                v[s * 16 + j + 8] = e2m1x2(c >> 4) * d;
            }
        }
    } else if constexpr (T == 30) {
        const uint16_t* p = (const uint16_t*) row + b * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = bf16(p[i]);
    } else {
        const float* p = (const float*) row + b * 32;
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = p[i];
    }
}

constexpr int MAXT = 16;
constexpr int MV_SG = 4;   // rows (sub-groups) a work-group

// y [ncols? pairs] rows: a sub-group a row; its lanes take the row's blocks in turn, each block against the T inputs
template <int TY>
void mv_rows(sycl::queue& q, const uint8_t* w, size_t stride, size_t row_bytes, const float* scale, const int32_t* ids, const float* x, int per_pair,
             float* y, int64_t T, int64_t K, int64_t n_in, int64_t n_out, bool moe) {
    constexpr int V = TY == 40 ? VB_FP4 : 32;
    const int64_t nb = n_in / V;
    const int64_t groups = (n_out + MV_SG - 1) / MV_SG;
    const int64_t pairs = moe ? T * K : 1;
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) pairs, (size_t) (groups * MV_SG * SG)), sycl::range<2>(1, MV_SG * SG)),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t p = it.get_group(0);
        const int64_t row = it.get_group(1) * MV_SG + sg.get_group_linear_id();
        if (row >= n_out) return;
        int64_t e = 0, nt = T, x0 = 0;
        if (moe) {
            e = ids[p];
            nt = 1;
            x0 = per_pair ? p : p / K;
        }
        const uint8_t* wr = w + (size_t) e * stride + (size_t) row * row_bytes;
        float acc[MAXT];
#pragma unroll
        for (int t = 0; t < MAXT; ++t) acc[t] = 0.0f;
        float v[V];
        for (int64_t b = lane; b < nb; b += SG) {
            decode<TY>(wr, b, v);
            for (int t = 0; t < nt; ++t) {
                const float* xr = x + (x0 + t) * n_in + b * V;
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < V; ++i) s += v[i] * xr[i];
                acc[t] += s;
            }
        }
        for (int t = 0; t < nt; ++t) {
            const float r = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
            if (lane == 0) {
                if (moe) y[p * n_out + row] = scale ? r * scale[e] : r;
                else y[t * n_out + row] = r;
            }
        }
    });
}

int mv_any(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const float* x, int per_pair, float* y, int64_t T,
           int64_t K, int64_t n_in, int64_t n_out, bool moe, const char* who) {
    if (!moe && (T < 1 || T > MAXT)) return ns_fail(std::string(who) + ": 1..16 columns");
    if (bb(type) == 0 || n_in % vb(type)) return ns_fail(std::string(who) + ": a type it does not take, or rows of part blocks");
    if (T <= 0 || n_out <= 0) return 0;
    const size_t rb = (size_t) (n_in / vb(type)) * bb(type);
    auto* wp = (const uint8_t*) w;
    switch (type) {
        case 0: mv_rows<0>(g->q, wp, stride, rb, scale, ids, x, per_pair, y, T, K, n_in, n_out, moe); break;
        case 8: mv_rows<8>(g->q, wp, stride, rb, scale, ids, x, per_pair, y, T, K, n_in, n_out, moe); break;
        case 30: mv_rows<30>(g->q, wp, stride, rb, scale, ids, x, per_pair, y, T, K, n_in, n_out, moe); break;
        case 40: mv_rows<40>(g->q, wp, stride, rb, scale, ids, x, per_pair, y, T, K, n_in, n_out, moe); break;
    }
    return 0;
}

template <int TY, typename O>
void expand(sycl::queue& q, const uint8_t* src, int64_t n, O* dst) {
    constexpr int V = TY == 40 ? VB_FP4 : 32;
    q.parallel_for(sycl::range<1>((size_t) (n / V)), [=](sycl::id<1> id) {
        const int64_t b = id[0];
        float v[V];
        decode<TY>(src, b, v);
#pragma unroll
        for (int i = 0; i < V; ++i) {
            if constexpr (sizeof(O) == 2) dst[b * V + i] = sycl::bit_cast<O>(half(v[i]));
            else dst[b * V + i] = v[i];
        }
    });
}

template <typename O>
int expand_any(ns_gpu* g, int type, const void* src, int64_t n, O* dst, const char* who) {
    if (bb(type) == 0 || n % vb(type)) return ns_fail(std::string(who) + ": a type it does not take, or part blocks");
    if (n <= 0) return 0;
    auto* s = (const uint8_t*) src;
    switch (type) {
        case 0: expand<0>(g->q, s, n, dst); break;
        case 8: expand<8>(g->q, s, n, dst); break;
        case 30: expand<30>(g->q, s, n, dst); break;
        case 40: expand<40>(g->q, s, n, dst); break;
    }
    return 0;
}

// ---- the attention ----
constexpr int D_ATT = 128, DL = D_ATT / SG, KT = 32;   // a head's dims, a lane's, keys a tile
constexpr int64_t KB = 256;                             // decode rows' keys a split (absolute positions)

// rows [r0, r0 + RT) x the G query heads of key head hk, keys of split sp (KB keys at absolute [(s0 + sp) KB, ..); KB 0:
// all): a sub-group a (row, head), lanes over the dims; the keys a tile in local memory
template <int RT, int G>
void attn_launch(sycl::queue& q, const float* qp, const half* kc, const half* vc, const float* r, const float* relb, int64_t E, int64_t R,
                 const float* sink, float* out, int64_t T, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win, float scale, int64_t kb,
                 int64_t s0, int64_t ns, float* part) {
    constexpr int WG = RT * G * SG;
    const int64_t tiles = (T + RT - 1) / RT;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> ks(sycl::range<1>((size_t) KT * D_ATT), h);
        sycl::local_accessor<half, 1> vs(sycl::range<1>((size_t) KT * D_ATT), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (tiles * Hkv * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id(), tid = (int) it.get_local_id(0);
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, tile = gi / Hkv;
            const int64_t rr = tile * RT + sgi / G, hq = hk * G + sgi % G;
            const bool live = rr < T;
            // the tile's keys: the union of its rows' [first, end), cut to the split
            const int64_t r_lo = tile * RT, r_hi = sycl::min<int64_t>(T, r_lo + RT) - 1;
            int64_t lo = win > 0 ? sycl::max<int64_t>(0, p0 + r_lo + 1 - win) : 0, hi = p0 + r_hi + 1;
            if (kb > 0) {
                lo = sycl::max<int64_t>(lo, (s0 + sp) * kb);
                hi = sycl::min<int64_t>(hi, (s0 + sp + 1) * kb);
            }
            const int64_t pos = p0 + rr, first = win > 0 ? sycl::max<int64_t>(0, pos + 1 - win) : 0, end = pos + 1;
            float qv[DL];
            float rv = 0.0f;
            if (live) {
#pragma unroll
                for (int i = 0; i < DL; ++i) qv[i] = qp[(rr * Hq + hq) * D_ATT + lane * DL + i] * scale;
                if (lane < R) rv = r[(rr * Hq + hq) * R + lane];
            }
            // with no split the sink is the first logit (value 0); split rows merge it later
            float m = -INFINITY, s = 0.0f;
            if (live && kb == 0) {
                m = sink[rr * Hq + hq];
                s = 1.0f;
            }
            float acc[DL];
#pragma unroll
            for (int i = 0; i < DL; ++i) acc[i] = 0.0f;
            for (int64_t j0 = lo; j0 < hi; j0 += KT) {
                const int64_t nk = sycl::min<int64_t>(KT, hi - j0);
                sycl::group_barrier(it.get_group());
                for (int i = tid; i < nk * D_ATT; i += WG) {
                    const int64_t j = j0 + i / D_ATT, slot = j % cap;
                    ks[i] = kc[(hk * cap + slot) * D_ATT + i % D_ATT];
                    vs[i] = vc[(hk * cap + slot) * D_ATT + i % D_ATT];
                }
                sycl::group_barrier(it.get_group());
                if (!live) continue;
                for (int jj = 0; jj < nk; ++jj) {
                    const int64_t j = j0 + jj;
                    if (j < first || j >= end) continue;
                    const int64_t d = pos - j;
                    float part_l = 0.0f;
#pragma unroll
                    for (int i = 0; i < DL; ++i) part_l += qv[i] * (float) ks[jj * D_ATT + lane * DL + i];
                    if (d < E && lane < R) part_l += rv * relb[d * R + lane];
                    const float lg = sycl::reduce_over_group(sg, part_l, sycl::plus<float>());
                    const float mn = sycl::fmax(m, lg);
                    const float c = sycl::exp(m - mn), p = sycl::exp(lg - mn);
                    s = s * c + p;
#pragma unroll
                    for (int i = 0; i < DL; ++i) acc[i] = acc[i] * c + p * (float) vs[jj * D_ATT + lane * DL + i];
                    m = mn;
                }
            }
            if (!live) return;
            if (kb == 0) {
#pragma unroll
                for (int i = 0; i < DL; ++i) out[(rr * Hq + hq) * D_ATT + lane * DL + i] = acc[i] / s;
            } else {
                float* pp = part + ((rr * Hq + hq) * ns + sp) * (D_ATT + 2);
#pragma unroll
                for (int i = 0; i < DL; ++i) pp[lane * DL + i] = acc[i];
                if (lane == 0) {
                    pp[D_ATT] = m;
                    pp[D_ATT + 1] = s;
                }
            }
        });
    });
}

// the splits of a decode row merged in order, with its sink
void attn_merge(sycl::queue& q, const float* part, const float* sink, float* out, int64_t T, int64_t Hq, int64_t ns) {
    q.parallel_for(sycl::range<2>((size_t) (T * Hq), (size_t) D_ATT), [=](sycl::id<2> id) {
        const int64_t rh = id[0], dd = id[1];
        const float* pp = part + rh * ns * (D_ATT + 2);
        float M = sink[rh];
        for (int64_t i = 0; i < ns; ++i) M = sycl::fmax(M, pp[i * (D_ATT + 2) + D_ATT]);
        float S = sycl::exp(sink[rh] - M), A = 0.0f;
        for (int64_t i = 0; i < ns; ++i) {
            const float c = sycl::exp(pp[i * (D_ATT + 2) + D_ATT] - M);
            S += pp[i * (D_ATT + 2) + D_ATT + 1] * c;
            A += pp[i * (D_ATT + 2) + dd] * c;
        }
        out[rh * D_ATT + dd] = A / S;
    });
}

// decode rows' splits: [s0, s0 + ns) over every row's keys
void splits(int64_t T, int64_t p0, int64_t win, int64_t& s0, int64_t& ns) {
    const int64_t lo = win > 0 ? std::max<int64_t>(0, p0 + 1 - win) : 0, hi = p0 + T;
    s0 = lo / KB;
    ns = (hi - 1) / KB - s0 + 1;
}

struct Ptrs {
    const float* p[8];
};

}  // namespace

extern "C" {

int ns_ink2_type_supported(int type, int64_t n_in) {
    return bb(type) > 0 && n_in % vb(type) == 0;
}

int ns_ink2_dequant(ns_gpu* g, int type, const void* src, int64_t n, float* dst) {
    NS_TRY
    return expand_any<float>(g, type, src, n, dst, "ns_ink2_dequant");
    NS_CATCH
}

int ns_ink2_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst) {
    NS_TRY
    return expand_any<uint16_t>(g, type, src, n, dst, "ns_ink2_dequant_f16");
    NS_CATCH
}

int ns_ink2_mv(ns_gpu* g, int type, const void* w, const float* x, float* y, int64_t n_in, int64_t n_out, int64_t T) {
    NS_TRY
    return mv_any(g, type, w, 0, nullptr, nullptr, x, 0, y, T, 1, n_in, n_out, false, "ns_ink2_mv");
    NS_CATCH
}

int ns_ink2_moe_mv(ns_gpu* g, int type, const void* w, size_t stride, const float* scale, const int32_t* ids, const float* x, int per_pair, float* y,
                   int64_t T, int64_t K, int64_t n_in, int64_t n_out) {
    NS_TRY
    return mv_any(g, type, w, stride, scale, ids, x, per_pair, y, T, K, n_in, n_out, true, "ns_ink2_moe_mv");
    NS_CATCH
}

int ns_ink2_route(ns_gpu* g, const float* logits, const float* bias, int32_t* ids, float* w, int64_t T, int64_t E, int64_t K, float scale) {
    NS_TRY
    if (E > 256 || K > 16) return ns_fail("ns_ink2_route: at most 256 experts, 16 chosen");
    // a sub-group a row: lane l holds experts l, l + 16, ...; K rounds of the largest p + bias (the lowest index of equals)
    constexpr int PER = 256 / SG;
    if (T > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * SG), sycl::range<1>(SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t t = it.get_group(0);
        const float* l = logits + t * E;
        float p[PER], v[PER];
        uint32_t used = 0;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int e = lane + i * SG;
            p[i] = e < E ? 1.0f / (1.0f + sycl::exp(-l[e])) : 0.0f;
            v[i] = e < E ? (bias ? p[i] + bias[e] : p[i]) : -INFINITY;
        }
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float bv = -INFINITY;
            int bi = 1 << 30;
#pragma unroll
            for (int i = 0; i < PER; ++i) {
                const int e = lane + i * SG;
                if (!(used >> i & 1) && e < E && (v[i] > bv || (v[i] == bv && e < bi))) {
                    bv = v[i];
                    bi = e;
                }
            }
            const float m = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
            const int best = sycl::reduce_over_group(sg, bv == m ? bi : (1 << 30), sycl::minimum<int>());
            float pb = 0.0f;
            if (best % SG == lane) {
                const int i = best / SG;
#pragma unroll
                for (int j = 0; j < PER; ++j)
                    if (j == i) {
                        used |= 1u << j;
                        pb = p[j];
                    }
            }
            pb = sycl::reduce_over_group(sg, pb, sycl::plus<float>());
            if (lane == 0) {
                ids[t * K + k] = best;
                w[t * K + k] = pb;
            }
            sum += pb;
        }
        sycl::group_barrier(sg);
        sum += 1e-20f;
        if (lane < K) w[t * K + lane] = w[t * K + lane] / sum * scale;
    });
    return 0;
    NS_CATCH
}

int ns_ink2_situ(ns_gpu* g, const float* gu, const float* rs, float* act, uint16_t* act_h, int64_t rows, int64_t F, float beta, float lb) {
    NS_TRY
    if (rows > 0) g->q.parallel_for(sycl::range<1>((size_t) (rows * F)), [=](sycl::id<1> id) {
        const int64_t i = id[0], r = i / F, c = i % F;
        const float sc = rs ? rs[r] : 1.0f;
        const float gt = gu[r * 2 * F + c] * sc, up = gu[r * 2 * F + F + c] * sc;
        const float a = beta * sycl::tanh(gt / beta) * (1.0f / (1.0f + sycl::exp(-gt)));
        const float u = lb > 0.0f ? lb * sycl::tanh(up / lb) : up;
        const float y = a * u;
        if (act) act[i] = y;
        if (act_h) act_h[i] = sycl::bit_cast<uint16_t>(half(y));
    });
    return 0;
    NS_CATCH
}

int ns_ink2_combine(ns_gpu* g, const float* e, const float* w, float* y, int64_t T, int64_t K, int64_t C) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t t = id[0] / C, c = id[0] % C;
        float s = 0.0f;
        for (int64_t k = 0; k < K; ++k) s += w[t * K + k] * e[(t * K + k) * C + c];
        y[t * C + c] = s;
    });
    return 0;
    NS_CATCH
}

int ns_ink2_combine_rows(ns_gpu* g, const float* rows, const int32_t* t_ptr, const int32_t* ent, const float* w, float* y, int64_t T, int64_t C) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t t = id[0] / C, c = id[0] % C;
        float s = 0.0f;
        for (int32_t j = t_ptr[t]; j < t_ptr[t + 1]; ++j) s += w[j] * rows[(int64_t) ent[j] * C + c];
        y[t * C + c] += s;
    });
    return 0;
    NS_CATCH
}

int ns_ink2_gather_f16(ns_gpu* g, const float* x, const int32_t* idx, uint16_t* out, int64_t n, int64_t C) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const int64_t i = id[0] / C, c = id[0] % C;
        out[id[0]] = sycl::bit_cast<uint16_t>(half(x[(int64_t) idx[i] * C + c]));
    });
    return 0;
    NS_CATCH
}

int ns_ink2_mix(ns_gpu* g, const float* const* states, int64_t ns, const float* query, int64_t nq, float* out, int64_t T, int64_t C, float eps) {
    NS_TRY
    if (ns < 1 || ns > 8 || nq < 1 || nq > 4) return ns_fail("ns_ink2_mix: 1..8 states, 1..4 queries");
    Ptrs st{};
    for (int i = 0; i < ns; ++i) st.p[i] = states[i];
    constexpr int W = 256;
    if (T > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) {
        auto grp = it.get_group();
        const int64_t t = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        float inv[8], pr[4][8];
        for (int s = 0; s < ns; ++s) {
            const float* x = st.p[s] + t * C;
            float ss = 0.0f;
            for (int64_t c = tid; c < C; c += W) ss += x[c] * x[c];
            inv[s] = sycl::rsqrt(sycl::reduce_over_group(grp, ss, sycl::plus<float>()) / (float) C + eps);
        }
        const float sc = 1.0f / sycl::sqrt((float) C);
        for (int qi = 0; qi < nq; ++qi) {
            float mx = -INFINITY;
            for (int s = 0; s < ns; ++s) {
                const float* x = st.p[s] + t * C;
                float d = 0.0f;
                for (int64_t c = tid; c < C; c += W) d += x[c] * inv[s] * query[qi * C + c];
                pr[qi][s] = sycl::reduce_over_group(grp, d, sycl::plus<float>()) * sc;
                mx = sycl::fmax(mx, pr[qi][s]);
            }
            float sum = 0.0f;
            for (int s = 0; s < ns; ++s) {
                pr[qi][s] = sycl::exp(pr[qi][s] - mx);
                sum += pr[qi][s];
            }
            for (int s = 0; s < ns; ++s) pr[qi][s] /= sum;
        }
        for (int64_t c = tid; c < C; c += W) {
            for (int qi = 0; qi < nq; ++qi) {
                float y = 0.0f;
                for (int s = 0; s < ns; ++s) y += pr[qi][s] * (st.p[s][t * C + c] * inv[s]);
                out[(qi * T + t) * C + c] = y;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_ink2_ctrl_z(ns_gpu* g, const float* cur, const float* msg, float* z, int64_t T, int64_t C, float eps) {
    NS_TRY
    constexpr int W = 256;
    if (T > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) {
        auto grp = it.get_group();
        const int64_t t = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const float* a = cur + t * C;
        const float* b = msg + t * C;
        float sa = 0.0f, sb = 0.0f;
        for (int64_t c = tid; c < C; c += W) {
            sa += a[c] * a[c];
            sb += b[c] * b[c];
        }
        const float ia = sycl::rsqrt(sycl::reduce_over_group(grp, sa, sycl::plus<float>()) / (float) C + eps);
        const float ib = sycl::rsqrt(sycl::reduce_over_group(grp, sb, sycl::plus<float>()) / (float) C + eps);
        for (int64_t c = tid; c < C; c += W) {
            z[t * 2 * C + c] = a[c] * ia;
            z[t * 2 * C + C + c] = b[c] * ib;
        }
    });
    return 0;
    NS_CATCH
}

int ns_ink2_ctrl_apply(ns_gpu* g, const float* cur, const float* msg, const float* f, const float* gn, float* out, int64_t T, int64_t C) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        const float x = cur[i];
        out[i] = (x - sycl::tanh(f[i]) * x) + sycl::tanh(gn[i]) * msg[i];
    });
    return 0;
    NS_CATCH
}

int ns_ink2_conv(ns_gpu* g, const float* x, float* state, const float* w, const float* dyn, float* out, float* snap, int64_t T, int64_t C, int64_t H,
                 float inv) {
    NS_TRY
    if (T <= 0) return 0;
    if (dyn && (H <= 0 || C % H)) return ns_fail("ns_ink2_conv: heads that divide the channels");
    const int64_t hc = dyn ? C / H : 1;
    // the rows (the state's three before them)
    g->q.parallel_for(sycl::range<1>((size_t) (T * C)), [=](sycl::id<1> id) {
        const int64_t t = id[0] / C, c = id[0] % C;
        auto at = [&](int64_t j) { return j < 3 ? state[j * C + c] : x[(j - 3) * C + c]; };   // the extended rows
        float s = 0.0f;
        for (int k = 0; k < 4; ++k) s += at(t + k) * w[c * 4 + k];
        if (dyn) {
            const float* dt = dyn + (t * H + c / hc) * 4;
            for (int l = 0; l < 4; ++l) s += dt[l] * at(t + 3 - l);
        }
        out[t * C + c] = (x[t * C + c] + s) * inv;
    });
    // then the state (and the snapshot): a work-item a channel
    g->q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = id[0];
        float s0 = state[c], s1 = state[C + c], s2 = state[2 * C + c];
        if (snap) {
            snap[c] = s0;
            snap[C + c] = s1;
            snap[2 * C + c] = s2;
            for (int64_t t = 0; t < T; ++t) snap[(3 + t) * C + c] = x[t * C + c];
        }
        for (int64_t t = sycl::max<int64_t>(0, T - 3); t < T; ++t) {
            s0 = s1;
            s1 = s2;
            s2 = x[t * C + c];
        }
        state[c] = s0;
        state[C + c] = s1;
        state[2 * C + c] = s2;
    });
    return 0;
    NS_CATCH
}

int ns_ink2_conv_keep(ns_gpu* g, const float* snap, float* state, int64_t keep, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>((size_t) (3 * C)), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        state[i] = snap[keep * C + i];
    });
    return 0;
    NS_CATCH
}

int ns_ink2_clip(ns_gpu* g, float* x, const float* scale, int64_t T, int64_t H, int64_t D, float thr, float log_alpha, float log_floor, int64_t p0) {
    NS_TRY
    constexpr int W = SG * 4;
    const int64_t n = T * H;
    if (n > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((n + 3) / 4 * W)), sycl::range<1>(W)),
                                 [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t rh = it.get_group(0) * 4 + sg.get_group_linear_id();
        if (rh >= n) return;
        const int64_t t = rh / H, h = rh % H;
        float* xr = x + rh * D;
        float ss = 0.0f;
        for (int64_t d = lane; d < D; d += SG) {
            const float v = scale ? xr[d] * scale[h * D + d] : xr[d];
            ss += v * v;
        }
        const float norm = sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>())) + 1e-6f;
        const float f = sycl::clamp(thr / norm, 0.0f, 1.0f);
        float tau = 1.0f;
        if (log_alpha > 0.0f) tau = log_alpha * sycl::log(sycl::fmax((float) (p0 + t + 1) * (1.0f / log_floor), 1.0f)) + 1.0f;
        for (int64_t d = lane; d < D; d += SG) {
            const float v = scale ? xr[d] * scale[h * D + d] : xr[d];
            xr[d] = log_alpha > 0.0f ? (v * f) * tau : v * f;
        }
    });
    return 0;
    NS_CATCH
}

int ns_ink2_rms_rows(ns_gpu* g, const float* x, float* y, int64_t rows, int64_t D, float eps) {
    NS_TRY
    constexpr int W = SG * 4;
    if (rows > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((rows + 3) / 4 * W)), sycl::range<1>(W)),
                                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t r = it.get_group(0) * 4 + sg.get_group_linear_id();
        if (r >= rows) return;
        float ss = 0.0f;
        for (int64_t d = lane; d < D; d += SG) ss += x[r * D + d] * x[r * D + d];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / (float) D + eps);
        for (int64_t d = lane; d < D; d += SG) y[r * D + d] = x[r * D + d] * inv;
    });
    return 0;
    NS_CATCH
}

int ns_ink2_kv_store(ns_gpu* g, const float* k, const float* v, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, void* kc, void* vc) {
    NS_TRY
    auto* kh = (half*) kc;
    auto* vh = (half*) vc;
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * Hkv * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
        const int64_t slot = (p0 + t) % cap;
        kh[(h * cap + slot) * D + d] = half(k[i]);
        vh[(h * cap + slot) * D + d] = half(v[i]);
    });
    return 0;
    NS_CATCH
}

int64_t ns_ink2_attn_scratch(int64_t T, int64_t Hq, int64_t D, int64_t p0, int64_t win) {
    if (T > MAXT) return 0;
    int64_t s0, ns;
    splits(T, p0, win, s0, ns);
    return T * Hq * ns * (D + 2);
}

int ns_ink2_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, const float* r, const float* relb, int64_t E, int64_t R, const float* sink,
                 float* out, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, int64_t win, float scale, float* part) {
    NS_TRY
    if (D != D_ATT || Hq != 3 * Hkv || R > SG) return ns_fail("ns_ink2_attn: heads of 128, 3 query heads a key head, a bias of at most 16");
    if (T <= 0) return 0;
    auto* kh = (const half*) kc;
    auto* vh = (const half*) vc;
    if (T > MAXT) {
        attn_launch<8, 3>(g->q, q, kh, vh, r, relb, E, R, sink, out, T, Hq, Hkv, cap, p0, win, scale, 0, 0, 1, nullptr);
        return 0;
    }
    if (!part) return ns_fail("ns_ink2_attn: decode rows need the scratch");
    int64_t s0, ns;
    splits(T, p0, win, s0, ns);
    attn_launch<1, 3>(g->q, q, kh, vh, r, relb, E, R, sink, out, T, Hq, Hkv, cap, p0, win, scale, KB, s0, ns, part);
    attn_merge(g->q, part, sink, out, T, Hq, ns);
    return 0;
    NS_CATCH
}

int ns_ink2_gate(ns_gpu* g, float* x, const float* gate, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        x[i] *= 1.0f / (1.0f + sycl::exp(-gate[i]));
    });
    return 0;
    NS_CATCH
}

int ns_ink2_add(ns_gpu* g, const float* a, const float* b, float* y, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { y[id[0]] = a[id[0]] + b[id[0]]; });
    return 0;
    NS_CATCH
}

int ns_ink2_softcap(ns_gpu* g, const float* x, float* out, int64_t rows, int64_t n, int64_t ld, float cap) {
    NS_TRY
    if (rows > 0) g->q.parallel_for(sycl::range<1>((size_t) (rows * n)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / n, c = id[0] % n;
        const float v = x[r * ld + c];
        out[id[0]] = cap > 0.0f ? cap * sycl::tanh(v * (1.0f / cap)) : v;
    });
    return 0;
    NS_CATCH
}

int ns_ink2_argmax(ns_gpu* g, const float* x, int32_t* out, int64_t T, int64_t n) {
    NS_TRY
    constexpr int W = 1024;
    if (T > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const float* xr = x + r * n;
        float best = -INFINITY;
        int64_t at = n;
        for (int64_t i = tid; i < n; i += W) {
            if (xr[i] > best) {
                best = xr[i];
                at = i;
            }
        }
        const float m = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
        const int64_t first = sycl::reduce_over_group(it.get_group(), best == m ? at : n, sycl::minimum<int64_t>());
        if (tid == 0) out[r] = (int32_t) (first < n ? first : 0);
    });
    return 0;
    NS_CATCH
}

// the museglimmer engine's GPU draw (mg.cpp), copied: the 256 largest, temperature, top-p, one uniform a row
int ns_ink2_sample(ns_gpu* g, const float* x, int32_t* out, int64_t T, int64_t n, float temp, float top_p, uint64_t seed, int64_t pos0) {
    NS_TRY
    constexpr int W = 1024, K = 256;
    if (T <= 0) return 0;
    if (n < K) return ns_fail("ns_ink2_sample: fewer than 256 values a row");
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> cv(sycl::range<1>(K), h);
        sycl::local_accessor<int32_t, 1> ci(sycl::range<1>(K), h);
        sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) {
            auto grp = it.get_group();
            const int64_t r = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const float* xr = x + r * n;
            auto key = [](float v) {
                const uint32_t b = sycl::bit_cast<uint32_t>(v);
                return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            };
            uint32_t kth = 0;
            for (int bit = 31; bit >= 0; --bit) {
                const uint32_t c = kth | (1u << bit);
                int64_t m = 0;
                for (int64_t i = tid; i < n; i += W) m += key(xr[i]) >= c;
                if (sycl::reduce_over_group(grp, m, sycl::plus<int64_t>()) >= K) kth = c;
            }
            if (tid == 0) cnt[0] = 0;
            sycl::group_barrier(grp);
            int64_t gt = 0;
            for (int64_t i = tid; i < n; i += W) gt += key(xr[i]) > kth;
            const int64_t n_gt = sycl::reduce_over_group(grp, gt, sycl::plus<int64_t>());
            for (int64_t i = tid; i < n; i += W) {
                if (key(xr[i]) > kth) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> a(cnt[0]);
                    const int at = a.fetch_add(1);
                    cv[at] = xr[i];
                    ci[at] = (int32_t) i;
                }
            }
            const int64_t chunk = (n + W - 1) / W, b0 = tid * chunk, b1 = sycl::min<int64_t>(n, b0 + chunk);
            int64_t eq = 0;
            for (int64_t i = b0; i < b1; ++i) eq += key(xr[i]) == kth;
            int64_t before = sycl::exclusive_scan_over_group(grp, eq, sycl::plus<int64_t>());
            for (int64_t i = b0; i < b1 && n_gt + before < K; ++i) {
                if (key(xr[i]) == kth) {
                    cv[n_gt + before] = xr[i];
                    ci[n_gt + before] = (int32_t) i;
                    ++before;
                }
            }
            sycl::group_barrier(grp);
            for (int size = 2; size <= K; size <<= 1) {
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                    if (tid < K) {
                        const int partner = tid ^ stride;
                        if (partner > tid) {
                            const bool up = (tid & size) == 0;
                            const float va = cv[tid], vb2 = cv[partner];
                            const int32_t ia = ci[tid], ib = ci[partner];
                            const bool a_first = va > vb2 || (va == vb2 && ia < ib);
                            if (a_first != up) {
                                cv[tid] = vb2;
                                cv[partner] = va;
                                ci[tid] = ib;
                                ci[partner] = ia;
                            }
                        }
                    }
                    sycl::group_barrier(grp);
                }
            }
            if (tid == 0) {
                uint64_t z = (uint64_t) seed + (uint64_t) (pos0 + r) * 0x9E3779B97F4A7C15ull;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                z ^= z >> 31;
                float u = (float) (z >> 40) / (float) (1u << 24);
                const float mx = cv[0];
                float sum = 0.0f;
                for (int i = 0; i < K; ++i) sum += sycl::exp((cv[i] - mx) / temp);
                float acc = 0.0f;
                int keep = K;
                for (int i = 0; i < K; ++i) {
                    acc += sycl::exp((cv[i] - mx) / temp) / sum;
                    if (acc >= top_p) {
                        keep = i + 1;
                        break;
                    }
                }
                float total = 0.0f;
                for (int i = 0; i < keep; ++i) total += sycl::exp((cv[i] - mx) / temp) / sum;
                int32_t pick = ci[keep - 1];
                for (int i = 0; i < keep; ++i) {
                    u -= sycl::exp((cv[i] - mx) / temp) / sum / total;
                    if (u <= 0.0f) {
                        pick = ci[i];
                        break;
                    }
                }
                out[r] = pick;
            }
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
