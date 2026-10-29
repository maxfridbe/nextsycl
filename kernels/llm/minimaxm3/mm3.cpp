// mm3.cpp: the minimaxm3 engine's bring-up kernels (mm3.h). Plain and exact: float32 arithmetic, fp16 caches as
// llama.cpp keeps them, one work-group a query row where a reduction is needed.
#include "ns.h"
#include "ns_internal.hpp"
#include "mm3.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>

namespace {
constexpr int BLK = 128;  // keys a block, values a head
}

extern "C" {

int ns_llm_minimaxm3_rope(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t row, int64_t head, int64_t n_rot, int64_t pos0, float base) {
    NS_TRY
    if (n_rot <= 0 || n_rot % 2 || n_rot > head) return ns_fail("ns_llm_minimaxm3_rope: n_rot");
    const int64_t half = n_rot / 2;
    const float ts = powf(base, -2.0f / (float) n_rot);
    g->q.parallel_for(sycl::range<1>((size_t) (T * H * half)), [=](sycl::id<1> id) {
        const int64_t k = id[0], i = k % half, th = k / half, h = th % H, t = th / H;
        float theta = (float) (pos0 + t);
        for (int64_t j = 0; j < i; ++j) theta *= ts;
        const float c = sycl::cos(theta), s = sycl::sin(theta);
        float* p = x + t * row + h * head;
        const float a = p[i], b = p[i + half];
        p[i] = a * c - b * s;
        p[i + half] = a * s + b * c;
    });
    return 0;
    NS_CATCH
}

int ns_llm_minimaxm3_swiglu_oai(ns_gpu* g, const float* gate, const float* up, float* out, int64_t n, float alpha, float limit) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float x = sycl::fmin(gate[id[0]], limit);
        const float y = sycl::fmin(sycl::fmax(up[id[0]], -limit), limit);
        out[id[0]] = (x / (1.f + sycl::exp(alpha * (-x)))) * (y + 1.f);
    });
    return 0;
    NS_CATCH
}

int ns_llm_minimaxm3_idx_scores(ns_gpu* g, const float* iq, const uint16_t* ik, float* bs, int64_t T, int64_t G, int64_t nblk, int64_t pos0, int64_t local) {
    NS_TRY
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(BLK), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (T * G * nblk * BLK), BLK), [=](sycl::nd_item<1> it) {
            const int64_t grp = it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const int64_t b = grp % nblk, tg = grp / nblk, gi = tg % G, t = tg / G;
            const int64_t pos = pos0 + t;
            qs[lid] = iq[(t * G + gi) * BLK + lid];
            sycl::group_barrier(it.get_group());
            float v = -INFINITY;
            const int64_t j = b * BLK + lid;
            if (j <= pos) {
                const sycl::half* kr = (const sycl::half*) ik + j * BLK;
                float s = 0.f;
                for (int d = 0; d < BLK; ++d) s += qs[d] * (float) kr[d];
                v = s;
            }
            float m = sycl::reduce_over_group(it.get_group(), v, sycl::maximum<float>());
            if (lid == 0) {
                if (b * BLK > pos) m = -INFINITY;
                else if (b + local > pos / BLK) m = m + 1e30f;
                bs[grp] = m;
            }
        });
    });
    return 0;
    NS_CATCH
}

int ns_llm_minimaxm3_idx_topk(ns_gpu* g, float* bs, int32_t* sel, int64_t T, int64_t G, int64_t nblk, int64_t K) {
    NS_TRY
    constexpr int WG = 64;
    g->q.parallel_for(sycl::nd_range<1>((size_t) (T * G * WG), WG), [=](sycl::nd_item<1> it) {
        const int64_t row = it.get_group(0);
        const int lid = (int) it.get_local_id(0);
        float* s = bs + row * nblk;
        for (int64_t k = 0; k < K; ++k) {
            float best = -INFINITY;
            for (int64_t b = lid; b < nblk; b += WG) best = sycl::fmax(best, s[b]);
            best = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
            int32_t at = 0x7fffffff;
            if (best > -INFINITY)
                for (int64_t b = lid; b < nblk; b += WG)
                    if (s[b] == best) { at = (int32_t) b; break; }
            at = sycl::reduce_over_group(it.get_group(), at, sycl::minimum<int32_t>());
            sycl::group_barrier(it.get_group());
            if (lid == 0) {
                sel[row * K + k] = best > -INFINITY ? at : -1;
                if (best > -INFINITY) s[at] = -INFINITY;
            }
            sycl::group_barrier(it.get_group());
        }
    });
    return 0;
    NS_CATCH
}

int ns_llm_minimaxm3_attend(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T,
                            int64_t H, int64_t G, int64_t pos0, float scale) {
    NS_TRY
    constexpr int CH = 2048;
    if (H % G) return ns_fail("ns_llm_minimaxm3_attend: heads");
    const int64_t per = H / G;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(BLK), h);
        sycl::local_accessor<float, 1> sc(sycl::range<1>(CH), h);
        sycl::local_accessor<int32_t, 1> jj(sycl::range<1>(CH), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (T * H * BLK), BLK), [=](sycl::nd_item<1> it) {
            const int64_t grp = it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const int64_t t = grp / H, hq = grp % H, gi = hq / per;
            const int64_t pos = pos0 + t;
            qs[lid] = q[(t * H + hq) * BLK + lid];
            sycl::group_barrier(it.get_group());
            const int64_t total = sel ? K * BLK : pos + 1;
            float m = -INFINITY, l = 0.f, acc = 0.f;
            for (int64_t c0 = 0; c0 < total; c0 += CH) {
                const int n = (int) sycl::min<int64_t>(CH, total - c0);
                for (int i = lid; i < n; i += BLK) {
                    const int64_t idx = c0 + i;
                    int64_t j = idx;
                    if (sel) {
                        const int32_t b = sel[(t * G + gi) * K + idx / BLK];
                        j = b < 0 ? -1 : (int64_t) b * BLK + idx % BLK;
                    }
                    float s = -INFINITY;
                    if (j >= 0 && j <= pos) {
                        const sycl::half* kr = (const sycl::half*) kc + (j * G + gi) * BLK;
                        float a = 0.f;
                        for (int d = 0; d < BLK; ++d) a += qs[d] * (float) kr[d];
                        s = a * scale;
                    } else {
                        j = -1;
                    }
                    sc[i] = s;
                    jj[i] = (int32_t) j;
                }
                sycl::group_barrier(it.get_group());
                float cm = -INFINITY;
                for (int i = lid; i < n; i += BLK) cm = sycl::fmax(cm, sc[i]);
                cm = sycl::reduce_over_group(it.get_group(), cm, sycl::maximum<float>());
                const float nm = sycl::fmax(m, cm);
                const float corr = m == -INFINITY ? 0.f : sycl::exp(m - nm);
                float ls = 0.f;
                if (nm > -INFINITY) {
                    for (int i = lid; i < n; i += BLK) {
                        const float e = sc[i] == -INFINITY ? 0.f : sycl::exp(sc[i] - nm);
                        sc[i] = e;
                        ls += e;
                    }
                }
                ls = sycl::reduce_over_group(it.get_group(), ls, sycl::plus<float>());
                sycl::group_barrier(it.get_group());
                l = l * corr + ls;
                acc *= corr;
                if (nm > -INFINITY) {
                    for (int i = 0; i < n; ++i) {
                        const int32_t j = jj[i];
                        if (j >= 0) acc += sc[i] * (float) ((const sycl::half*) vc)[((int64_t) j * G + gi) * BLK + lid];
                    }
                }
                m = nm;
                sycl::group_barrier(it.get_group());
            }
            out[(t * H + hq) * BLK + lid] = l > 0.f ? acc / l : 0.f;
        });
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
