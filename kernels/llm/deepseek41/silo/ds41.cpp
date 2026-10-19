// ds41.cpp: the deepseek41 engine's kernels copied from the deepseek4 silo (ds41.h) - YaRN's RoPE on a head's tail and
// back, the compressor's pooling, the raw keys' ring, attention over the window and the compressed blocks with sinks.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdlib>

namespace {

// ggml's rope_yarn_ramp and the angle for pair i0 (ggml-cpu ops.cpp's rope cache: the angle by repeated products)
inline float yarn_ramp(float low, float high, int i0) {
    const float y = ((float) (i0 / 2) - low) / sycl::fmax(0.001f, high - low);
    return 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y));
}
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

}  // namespace

extern "C" {

int ns_ds41_rope(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t row, int64_t head, int64_t off, int64_t n_dims, int64_t pos0, int64_t pstep,
                float freq_base, float freq_scale, float ext_factor, float mscale, float corr0, float corr1, int inverse) {
    NS_TRY
    if (n_dims % 2 != 0) return ns_fail("ns_ds41_rope: an even rotated width");
    const int64_t pairs = n_dims / 2;
    const float sgn = inverse ? -1.f : 1.f;
    g->q.parallel_for(sycl::range<3>(T, H, pairs), [=](sycl::id<3> id) {
        const int64_t t = id[0], h = id[1], i = id[2];
        const int i0 = (int) (2 * i);
        float c, s;
        yarn_cs((float) (pos0 + t * pstep), i0, (int) n_dims, freq_base, freq_scale, ext_factor, mscale, corr0, corr1, c, s);
        s *= sgn;
        float* v = x + t * row + h * head + off + i0;
        const float x0 = v[0], x1 = v[1];
        v[0] = x0 * c - x1 * s;
        v[1] = x0 * s + x1 * c;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_pool(ns_gpu* g, float* ring, const float* kv, const float* sc, const float* ape, float* out, int64_t pos0, int64_t T, int64_t D,
                int64_t ratio, int overlap, int64_t ring_n) {
    NS_TRY
    const int64_t W = overlap ? 2 * D : D;
    const int64_t j_lo = pos0 / ratio, j_hi = (pos0 + T) / ratio;
    auto& q = g->q;
    if (j_hi > j_lo) {
        q.parallel_for(sycl::range<2>(j_hi - j_lo, D), [=](sycl::id<2> id) {
            const int64_t j = j_lo + id[0], c = id[1];
            // a member's kv and score (+ ape) at value v of its row
            auto member = [&](int64_t p, int64_t v, float& k, float& s) {
                const int64_t m = p % ratio;
                const float* kr = p >= pos0 ? kv + (p - pos0) * W : ring + (p % ring_n) * 2 * W;
                const float* sr = p >= pos0 ? sc + (p - pos0) * W : ring + (p % ring_n) * 2 * W + W;
                k = kr[v];
                s = sr[v] + ape[m * W + v];
            };
            float mx = -INFINITY;
            // pass 1: the max; pass 2: the sums (members read twice - cheaper than holding 2 x 128 of them)
            const int64_t passes = overlap ? 2 : 1;
            for (int64_t half = 0; half < passes; ++half) {
                const int64_t b = overlap && half == 0 ? j - 1 : j;
                if (b < 0) continue;
                const int64_t v = overlap && half == 1 ? D + c : c;
                for (int64_t m = 0; m < ratio; ++m) {
                    float k, s;
                    member(b * ratio + m, v, k, s);
                    mx = sycl::fmax(mx, s);
                }
            }
            float sum = 0.f, acc = 0.f;
            for (int64_t half = 0; half < passes; ++half) {
                const int64_t b = overlap && half == 0 ? j - 1 : j;
                if (b < 0) continue;
                const int64_t v = overlap && half == 1 ? D + c : c;
                for (int64_t m = 0; m < ratio; ++m) {
                    float k, s;
                    member(b * ratio + m, v, k, s);
                    const float e = sycl::exp(s - mx);
                    sum += e;
                    acc += e * k;
                }
            }
            out[(j - j_lo) * D + c] = acc / sum;
        });
    }
    // the ring keeps the last ring_n positions (every row of a pass: a verify pass rolled back to any length still
    // finds the tokens its next block needs)
    const int64_t first = sycl::max<int64_t>(pos0, pos0 + T - ring_n);
    q.parallel_for(sycl::range<2>(pos0 + T - first, W), [=](sycl::id<2> id) {
        const int64_t p = first + id[0], c = id[1];
        ring[(p % ring_n) * 2 * W + c] = kv[(p - pos0) * W + c];
        ring[(p % ring_n) * 2 * W + W + c] = sc[(p - pos0) * W + c];
    });
    return 0;
    NS_CATCH
}

int ns_ds41_ring_put(ns_gpu* g, const float* kv, uint16_t* ring, int64_t T, int64_t pos0, int64_t R, int64_t D) {
    NS_TRY
    sycl::half* r16 = reinterpret_cast<sycl::half*>(ring);
    g->q.parallel_for(sycl::range<2>(T, D), [=](sycl::id<2> id) {
        const int64_t t = id[0], c = id[1];
        r16[((pos0 + t) % R) * D + c] = (sycl::half) kv[t * D + c];
    });
    return 0;
    NS_CATCH
}

int ns_ds41_rows_f16(ns_gpu* g, const float* x, uint16_t* cache, int64_t n, int64_t j0, int64_t D) {
    NS_TRY
    sycl::half* c16 = reinterpret_cast<sycl::half*>(cache);
    g->q.parallel_for(sycl::range<2>(n, D), [=](sycl::id<2> id) { c16[(j0 + id[0]) * D + id[1]] = (sycl::half) x[id[0] * D + id[1]]; });
    return 0;
    NS_CATCH
}

}  // extern "C"

namespace {
// a work-group a (row, HG heads): the row's cells' scores into local memory, a softmax with the sinks, the values
template <int HG>
void attend(sycl::queue& q, const float* qv, const sycl::half* raw, int64_t R, const sycl::half* comp, int64_t ratio, const int32_t* sel, int64_t K,
            const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window, float scale, int64_t maxn) {
    constexpr int WG = 256, D = 512;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(HG * D), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>(HG * maxn), h);
        h.parallel_for(sycl::nd_range<1>(T * (H / HG) * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t grp = it.get_group(0), t = grp / (H / HG), h0 = (grp % (H / HG)) * HG;
            const int lid = it.get_local_id(0);
            const int64_t p = pos0 + t;
            const int64_t n_raw = sycl::min<int64_t>(window, p + 1);
            const int64_t vis = comp ? (p + 1) / ratio : 0;
            const bool pick = sel && vis > K;
            const int64_t n_comp = pick ? K : vis;
            const int64_t n = n_raw + n_comp;
            auto cell = [&](int64_t i) -> const sycl::half* {
                if (i < n_raw) return raw + ((p - n_raw + 1 + i) % R) * D;
                const int64_t k = i - n_raw;
                return comp + (pick ? (int64_t) sel[t * K + k] : k) * D;
            };
            for (int64_t r = lid; r < HG * D; r += WG) qs[r] = qv[(t * H + h0) * D + r];
            sycl::group_barrier(it.get_group());
            float mx[HG];
            for (int k = 0; k < HG; ++k) mx[k] = sinks[h0 + k];
            for (int64_t i = lid; i < n; i += WG) {
                const sycl::vec<sycl::half, 4>* cs = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(cell(i));
                float dot[HG] = {};
                for (int64_t r = 0; r < D / 4; ++r) {
                    const sycl::float4 v = cs[r].convert<float>();
                    for (int k = 0; k < HG; ++k)
                        dot[k] += qs[k * D + 4 * r] * v.x() + qs[k * D + 4 * r + 1] * v.y() + qs[k * D + 4 * r + 2] * v.z() + qs[k * D + 4 * r + 3] * v.w();
                }
                for (int k = 0; k < HG; ++k) {
                    dot[k] *= scale;
                    sc[k * maxn + i] = dot[k];
                    mx[k] = sycl::fmax(mx[k], dot[k]);
                }
            }
            float sum[HG];
            for (int k = 0; k < HG; ++k) {
                mx[k] = sycl::reduce_over_group(it.get_group(), mx[k], sycl::maximum<float>());
                float sk = 0.f;
                for (int64_t i = lid; i < n; i += WG) {
                    const float e = sycl::exp(sc[k * maxn + i] - mx[k]);
                    sc[k * maxn + i] = e;
                    sk += e;
                }
                // the sink: in the sum, no value
                sum[k] = sycl::reduce_over_group(it.get_group(), sk, sycl::plus<float>()) + sycl::exp(sinks[h0 + k] - mx[k]);
            }
            sycl::group_barrier(it.get_group());
            float acc[HG][D / WG] = {};
            for (int64_t i = 0; i < n; ++i) {
                const sycl::half* cs = cell(i);
                float cv[D / WG];
                for (int f = 0; f < D / WG; ++f) cv[f] = (float) cs[lid + f * WG];
                for (int k = 0; k < HG; ++k) {
                    const float w = sc[k * maxn + i];
                    for (int f = 0; f < D / WG; ++f) acc[k][f] += w * cv[f];
                }
            }
            for (int k = 0; k < HG; ++k)
                for (int f = 0; f < D / WG; ++f) out[(t * H + h0 + k) * D + lid + f * WG] = acc[k][f] / sum[k];
        });
    });
}
}  // namespace

extern "C" {

int ns_ds41_attend(ns_gpu* g, const float* q, const uint16_t* raw, int64_t R, const uint16_t* comp, int64_t ratio, const int32_t* sel, int64_t K,
                  const float* sinks, float* out, int64_t T, int64_t H, int64_t pos0, int64_t window, float scale) {
    NS_TRY
    static const bool xmx = [] { const char* v = getenv("NS_DS41_ATTN_XMX"); return !v || v[0] != '0'; }();
    if (xmx && ns_ds41_attend_xmx_supported(H, 512)) return ns_ds41_attend_xmx(g, q, raw, R, comp, ratio, sel, K, sinks, out, T, H, pos0, window, scale);
    // the most cells a row reads: the window, and the blocks the last row sees (or K of them)
    const int64_t last = pos0 + T - 1;
    int64_t vis = comp ? (last + 1) / ratio : 0;
    if (sel && vis > K) vis = K;
    const int64_t maxn = (window + vis + 15) / 16 * 16;
    // local memory: HG x maxn scores, HG x 512 queries (64 KiB at most)
    const int64_t cap = 60 * 1024 / 4;
    const sycl::half* r16 = reinterpret_cast<const sycl::half*>(raw);
    const sycl::half* c16 = reinterpret_cast<const sycl::half*>(comp);
    if (H % 4 == 0 && 4 * (maxn + 512) <= cap) attend<4>(g->q, q, r16, R, c16, ratio, sel, K, sinks, out, T, H, pos0, window, scale, maxn);
    else if (maxn + 512 <= cap) attend<1>(g->q, q, r16, R, c16, ratio, sel, K, sinks, out, T, H, pos0, window, scale, maxn);
    else return ns_fail("ns_ds41_attend: " + std::to_string(maxn) + " cells a row (past the context this kernel holds)");
    return 0;
    NS_CATCH
}

int ns_ds41_permute(ns_gpu* g, const float* src, float* dst, int64_t A, int64_t B, int64_t C) {
    NS_TRY
    g->q.parallel_for(sycl::range<3>(A, B, C), [=](sycl::id<3> id) {
        const int64_t a = id[0], b = id[1], c = id[2];
        dst[(b * A + a) * C + c] = src[(a * B + b) * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_ds41_kcopy(ns_gpu* g, void* dst, const void* src, size_t n, int64_t* ticket) {
    NS_TRY
    if (n % 4 != 0) return ns_fail("ns_ds41_kcopy: a multiple of 4 bytes");
    sycl::event e;
    if (n % 16 == 0) {
        const sycl::uint4* s = static_cast<const sycl::uint4*>(src);
        sycl::uint4* d = static_cast<sycl::uint4*>(dst);
        e = g->q.parallel_for(sycl::range<1>(n / 16), [=](sycl::id<1> i) { d[i] = s[i]; });
    } else {
        const uint32_t* s = static_cast<const uint32_t*>(src);
        uint32_t* d = static_cast<uint32_t*>(dst);
        e = g->q.parallel_for(sycl::range<1>(n / 4), [=](sycl::id<1> i) { d[i] = s[i]; });
    }
    const int64_t t = g->next_ev++;
    g->ev[t % g->ev.size()] = e;
    *ticket = t;
    return 0;
    NS_CATCH
}

int ns_ds41_ticket_wait(ns_gpu* g, int64_t t) {
    NS_TRY
    g->ev[t % g->ev.size()].wait();
    return 0;
    NS_CATCH
}

}  // extern "C"
