// ov.cpp: the OmniVoice engine's own kernels (ov.h) - small item-per-element or sub-group-per-row kernels around the
// shared products. Sub-groups of 16 (Xe2's native width).
#include "ns.h"
#include "ns_internal.hpp"
#include "ov.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;
}  // namespace

extern "C" {

int ns_audio_ov_audio_embed(ns_gpu* g, const float* table, int64_t C, int64_t V, int64_t D, const int32_t* ids, const int32_t* rows, int64_t n,
                            float* out, int64_t ldo) {
    NS_TRY
    if (n <= 0 || D <= 0) return 0;
    if (C <= 0 || C > 32 || V <= 0) return ns_fail("ns_audio_ov_audio_embed: 1 to 32 codebooks");
    g->q.parallel_for(sycl::range<2>((size_t) n, (size_t) D), [=](sycl::id<2> id) {
        const int64_t i = (int64_t) id[0], j = (int64_t) id[1];
        float s = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            int64_t code = ids[i * C + c];
            code = code < 0 ? 0 : (code >= V ? V - 1 : code);
            s += table[(code + c * V) * D + j];
        }
        out[(int64_t) rows[i] * ldo + j] = s;
    });
    return 0;
    NS_CATCH
}

int ns_audio_ov_expand_qkv(ns_gpu* g, const void* qkv, int64_t R, int64_t Hq, int64_t Hkv, int64_t D, void* out) {
    NS_TRY
    if (R <= 0) return 0;
    if (Hkv <= 0 || Hq % Hkv != 0) return ns_fail("ns_audio_ov_expand_qkv: the query heads a multiple of the key / value heads");
    const half* x = (const half*) qkv;
    half* o = (half*) out;
    const int64_t in_w = (Hq + 2 * Hkv) * D, out_w = 3 * Hq * D, rep = Hq / Hkv;
    g->q.parallel_for(sycl::range<2>((size_t) R, (size_t) out_w), [=](sycl::id<2> id) {
        const int64_t r = (int64_t) id[0], j = (int64_t) id[1];
        const int64_t part = j / (Hq * D), within = j % (Hq * D);
        const int64_t h = within / D, d = within % D;
        int64_t src;
        if (part == 0) src = j;
        else if (part == 1) src = Hq * D + (h / rep) * D + d;
        else src = (Hq + Hkv) * D + (h / rep) * D + d;
        o[r * out_w + j] = x[r * in_w + src];
    });
    return 0;
    NS_CATCH
}

int ns_audio_ov_score(ns_gpu* g, const float* logits, int64_t T, int64_t C, int64_t V, int64_t mask, float guidance, int32_t* tok, float* score) {
    NS_TRY
    if (T <= 0 || C <= 0) return 0;
    // a sub-group a (codebook, position): four passes over the V codes of the two rows
    constexpr int NSG = 4;
    const int64_t rows = C * T, groups = (rows + NSG - 1) / NSG, W = C * V;
    const float gs = guidance;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SG * NSG), sycl::range<1>(SG * NSG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t r = (int64_t) it.get_group(0) * NSG + sg.get_group_linear_id();
        if (r >= rows) return;
        const int64_t c = r / T, t = r % T;
        const float* lc = logits + t * W + c * V;
        const float* lu = logits + (T + t) * W + c * V;
        float mc = -INFINITY, mu = -INFINITY;
        for (int64_t v = lane; v < V; v += SG) {
            mc = sycl::fmax(mc, lc[v]);
            mu = sycl::fmax(mu, lu[v]);
        }
        mc = sycl::reduce_over_group(sg, mc, sycl::maximum<float>());
        mu = sycl::reduce_over_group(sg, mu, sycl::maximum<float>());
        float sc = 0.0f, su = 0.0f;
        for (int64_t v = lane; v < V; v += SG) {
            sc += sycl::exp(lc[v] - mc);
            su += sycl::exp(lu[v] - mu);
        }
        sc = sycl::reduce_over_group(sg, sc, sycl::plus<float>());
        su = sycl::reduce_over_group(sg, su, sycl::plus<float>());
        const float nc = mc + sycl::log(sc), nu = mu + sycl::log(su);
        // z over every code (the normaliser), the best of the codes but the mask
        float mz = -INFINITY, best = -INFINITY;
        int bi = (int) V;
        for (int64_t v = lane; v < V; v += SG) {
            const float a = lc[v] - nc;
            const float z = gs != 0.0f ? a + gs * (a - (lu[v] - nu)) : a;
            mz = sycl::fmax(mz, z);
            if (v != mask && (z > best || (z == best && v < bi))) {
                best = z;
                bi = (int) v;
            }
        }
        mz = sycl::reduce_over_group(sg, mz, sycl::maximum<float>());
        const float top = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
        const int first = sycl::reduce_over_group(sg, best == top ? bi : (int) V, sycl::minimum<int>());
        float sz = 0.0f;
        for (int64_t v = lane; v < V; v += SG) {
            const float a = lc[v] - nc;
            const float z = gs != 0.0f ? a + gs * (a - (lu[v] - nu)) : a;
            sz += sycl::exp(z - mz);
        }
        sz = sycl::reduce_over_group(sg, sz, sycl::plus<float>());
        if (lane == 0) {
            tok[r] = first;
            score[r] = top - (mz + sycl::log(sz));
        }
    });
    return 0;
    NS_CATCH
}

int ns_audio_ov_chan_norm(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* b, float eps, float* out) {
    NS_TRY
    if (C <= 0 || L <= 0) return 0;
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) C * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t c = (int64_t) it.get_group(0);
        const int64_t i0 = (int64_t) it.get_local_id(0);
        const float* row = x + c * L;
        float s = 0.0f;
        for (int64_t i = i0; i < L; i += WG) s += row[i];
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) L;
        float q = 0.0f;
        for (int64_t i = i0; i < L; i += WG) {
            const float d = row[i] - mean;
            q += d * d;
        }
        const float var = sycl::reduce_over_group(it.get_group(), q, sycl::plus<float>()) / (float) L;
        const float k = sycl::rsqrt(var + eps) * (w ? w[c] : 1.0f), o = b ? b[c] : 0.0f;
        for (int64_t i = i0; i < L; i += WG) out[c * L + i] = (row[i] - mean) * k + o;
    });
    return 0;
    NS_CATCH
}

int ns_audio_ov_elu(ns_gpu* g, const float* x, int64_t n, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        const float v = x[i];
        out[i] = v > 0.0f ? v : sycl::expm1(v);
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
