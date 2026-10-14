// mg.cpp: the museglimmer engine's own kernels (mg.h) - Muse Glimmer's attention: q / k norms with rotary positions on
// the sliding-window layers (llama.cpp's NORM rotation: adjacent pairs), ring caches, the causal attention over a
// window, the sigmoid output gate. After llama.cpp's muse-glimmer graph (src/models/muse-glimmer.cpp).
#include "ns.h"
#include "ns_internal.hpp"
#include "mg.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;

// ---- cached causal attention (the qwen35 engine's, q35.cpp, with a window and ring slots): a work-group per (row,
// group of G query heads, split of the keys), NSG sub-groups taking the split's keys in turn, each lane DL features of
// the G heads; the sub-groups' (max, sum, weighted values) merged in local memory, the splits (decode) by a second
// kernel
constexpr int NSG_A = 4;

// decode rows' (at most 16: a verify pass) keys a split: ~512 work-groups whatever the context, at least 32 keys each;
// prompt rows one split
int64_t split_keys(int64_t S, int64_t kv, int64_t groups) {
    if (S > 16) return kv;
    const int64_t want = (kv * groups + 511) / 512;
    return sycl::max<int64_t>(32, (want + 31) / 32 * 32);
}
// the most keys a row sees (causal: the last row's; not causal: the first row's, the block's end its last)
int64_t seen(int64_t S, int64_t p0, int64_t win, int causal) {
    const int64_t end = p0 + S;
    if (win <= 0) return end;
    return causal ? sycl::min(end, win) : end - sycl::max<int64_t>(0, p0 + 1 - win);
}
int64_t splits(int64_t S, int64_t kv, int64_t groups) {
    const int64_t kb = split_keys(S, kv, groups);
    return kv > kb ? (kv + kb - 1) / kb : 1;
}

// KB keys a split, ns splits (the most a row has); a row's splits start at its first key
template <int DL, int G>
void attn(sycl::queue& q, const float* qp, const half* kc, const half* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t cap, int64_t p0, int64_t win,
          int causal, float* out, float* part, int64_t KB, int64_t ns) {
    constexpr int D = SG * DL, WG = SG * NSG_A;
    const int64_t groups = Hq / G, per_kv = Hq / Hkv;
    const float scale = 1.0f / sycl::sqrt((float) D);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NSG_A * G * (D + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * groups * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hg = gi % groups, r = gi / groups;
            const int64_t hk = hg * G / per_kv;
            // positions this row sees: [first, end)
            const int64_t pos = p0 + r, end = causal ? pos + 1 : p0 + S, first = win > 0 ? sycl::max<int64_t>(0, pos + 1 - win) : 0;
            const int64_t n = end - first;
            const int64_t j0 = first + sp * (ns > 1 ? KB : n), j1 = sycl::min(end, ns > 1 ? first + (sp + 1) * KB : end);
            float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float* qr = qp + r * Hq * D + (hg * G + g) * D + lane * DL;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    qv[g][i] = qr[i] * scale;
                    acc[g][i] = 0.0f;
                }
                m[g] = -INFINITY;
                l[g] = 0.0f;
            }
            const half* kb = kc + hk * cap * D + lane * DL;
            const half* vb = vc + hk * cap * D + lane * DL;
            for (int64_t j = j0 + sgi; j < j1; j += NSG_A) {
                const int64_t slot = j % cap;
                const sycl::vec<half, DL> kv = *reinterpret_cast<const sycl::vec<half, DL>*>(kb + slot * D);
                const sycl::vec<half, DL> vv = *reinterpret_cast<const sycl::vec<half, DL>*>(vb + slot * D);
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    float d = 0.0f;
#pragma unroll
                    for (int i = 0; i < DL; ++i) d += qv[g][i] * (float) kv[i];
                    const float sc = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                    const float mn = sycl::fmax(m[g], sc);
                    const float c = sycl::exp(m[g] - mn), e = sycl::exp(sc - mn);
#pragma unroll
                    for (int i = 0; i < DL; ++i) acc[g][i] = acc[g][i] * c + e * (float) vv[i];
                    l[g] = l[g] * c + e;
                    m[g] = mn;
                }
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float* o = &slm[(sgi * G + g) * (D + 2)];
#pragma unroll
                for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
                if (lane == 0) {
                    o[D] = m[g];
                    o[D + 1] = l[g];
                }
            }
            sycl::group_barrier(it.get_group());
            for (int idx = (int) it.get_local_id(0); idx < G * D; idx += WG) {
                const int g = idx / D, d = idx % D;
                float mx = -INFINITY;
                for (int k = 0; k < NSG_A; ++k) mx = sycl::fmax(mx, slm[(k * G + g) * (D + 2) + D]);
                float L = 0.0f, A = 0.0f;
                if (mx != -INFINITY) {
                    for (int k = 0; k < NSG_A; ++k) {
                        const float* o = &slm[(k * G + g) * (D + 2)];
                        const float e = sycl::exp(o[D] - mx);
                        L += o[D + 1] * e;
                        A += o[d] * e;
                    }
                }
                const int64_t hq = hg * G + g;
                if (ns == 1) {
                    out[r * Hq * D + hq * D + d] = L > 0.0f ? A / L : 0.0f;
                } else {
                    float* p = part + ((r * Hq + hq) * ns + sp) * (D + 2);
                    p[d] = A;
                    if (d == 0) {
                        p[D] = mx;
                        p[D + 1] = L;
                    }
                }
            }
        });
    });
    if (ns > 1) {
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * Hq) * D), sycl::range<1>(D)), [=](sycl::nd_item<1> it) {
            const int64_t rh = it.get_group(0), d = it.get_local_id(0);
            const float* p = part + rh * ns * (D + 2);
            // a row's splits past its keys were not written
            float mx = -INFINITY;
            for (int64_t k = 0; k < ns; ++k) mx = sycl::fmax(mx, p[k * (D + 2) + D]);
            float L = 0.0f, A = 0.0f;
            if (mx != -INFINITY) {
                for (int64_t k = 0; k < ns; ++k) {
                    const float e = sycl::exp(p[k * (D + 2) + D] - mx);
                    L += p[k * (D + 2) + D + 1] * e;
                    A += p[k * (D + 2) + d] * e;
                }
            }
            out[rh * D + d] = L > 0.0f ? A / L : 0.0f;
        });
    }
}

// query heads a work-group: 4 (G 16 would hold 16 heads' queries and sums in registers)
constexpr int GQ = 4;
}  // namespace

extern "C" {

int ns_mg_qk_norm_rope(ns_gpu* g, const float* x, int64_t xs, float* y, int64_t ys, int64_t T, int64_t H, int64_t D, const float* w, float eps,
                       int rope, float theta, int64_t p0) {
    NS_TRY
    if (T <= 0) return 0;
    if (D % (2 * SG) != 0) return ns_fail("ns_mg_qk_norm_rope: D a multiple of 32");
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t th = it.get_group(0), t = th / H, h = th % H;
        const float* xr = x + t * xs + h * D;
        float* yr = y + t * ys + h * D;
        float s = 0.0f;
        for (int64_t i = l; i < D; i += SG) s += xr[i] * xr[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) D + eps);
        const float pos = (float) (p0 + t);
        // a lane a pair at a time: (2i, 2i + 1) NORM, (i, i + D / 2) NEOX
        for (int64_t i = l; i < D / 2; i += SG) {
            const int64_t i0 = rope == 2 ? i : 2 * i, i1 = rope == 2 ? i + D / 2 : 2 * i + 1;
            const float a0 = xr[i0] * inv * w[i0], a1 = xr[i1] * inv * w[i1];
            if (rope) {
                const float a = pos * sycl::pow(theta, -2.0f * (float) i / (float) D);
                const float c = sycl::cos(a), sn = sycl::sin(a);
                yr[i0] = a0 * c - a1 * sn;
                yr[i1] = a0 * sn + a1 * c;
            } else {
                yr[i0] = a0;
                yr[i1] = a1;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_mg_kv_store(ns_gpu* g, const float* k, const float* v, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0, void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    if (T > cap) return ns_fail("ns_mg_kv_store: more rows than the cache's slots");
    half* ko = (half*) kc;
    half* vo = (half*) vc;
    g->q.parallel_for(sycl::range<1>((size_t) (T * Hkv * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
        const int64_t at = (h * cap + (p0 + t) % cap) * D + d;
        ko[at] = (half) k[t * Hkv * D + h * D + d];
        vo[at] = (half) v[t * Hkv * D + h * D + d];
    });
    return 0;
    NS_CATCH
}

int64_t ns_mg_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0, int64_t win, int causal) {
    int64_t ns = splits(T, seen(T, p0, win, causal), Hq / GQ);
    if (T <= 16 && causal)
        for (int64_t r = 0; r < T; ++r) ns = sycl::max(ns, splits(1, seen(1, p0 + r, win, 1), Hq / GQ));
    return ns > 1 ? T * Hq * ns * (D + 2) : 0;
}

int ns_mg_attn(ns_gpu* g, const float* q, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
               int64_t win, int causal, float* out, float* scratch) {
    NS_TRY
    if (T <= 0) return 0;
    if (D != 128 || Hq % Hkv != 0 || (Hq / Hkv) % GQ != 0) return ns_fail("ns_mg_attn: head 128 with a multiple of 4 query heads a key head");
    // the ring holds the oldest position the first row sees and the newest the last writes
    const int64_t need = win > 0 ? sycl::min(p0 + T, win + T - 1) : p0 + T;
    if (need > cap) return ns_fail("ns_mg_attn: the cache's ring is shorter than the window and the pass");
    if (ns_mg_attn_scratch(T, Hq, Hkv, D, p0, win, causal) > 0 && !scratch) return ns_fail("ns_mg_attn: no scratch");
    const half* k = (const half*) kc;
    const half* v = (const half*) vc;
    const int64_t groups = Hq / GQ;
    if (T > 16 || !causal) {
        const int64_t kvn = seen(T, p0, win, causal);
        attn<8, GQ>(g->q, q, k, v, T, Hq, Hkv, cap, p0, win, causal, out, scratch, split_keys(T, kvn, groups), splits(T, kvn, groups));
        return 0;
    }
    // decode rows (a verify pass's): each row split as its one-row pass would be - the same arithmetic, bit for bit -
    // in one launch when the rows' split sizes agree, else a row at a time
    auto row_keys = [&](int64_t r) { return seen(1, p0 + r, win, 1); };
    const int64_t kb0 = split_keys(1, row_keys(0), groups);
    bool same = true;
    int64_t ns = 1;
    for (int64_t r = 0; r < T; ++r) {
        same = same && split_keys(1, row_keys(r), groups) == kb0;
        ns = sycl::max(ns, splits(1, row_keys(r), groups));
    }
    if (same) {
        attn<8, GQ>(g->q, q, k, v, T, Hq, Hkv, cap, p0, win, 1, out, scratch, kb0, ns);
    } else {
        for (int64_t r = 0; r < T; ++r) {
            const int64_t kn = row_keys(r);
            attn<8, GQ>(g->q, q + r * Hq * D, k, v, 1, Hq, Hkv, cap, p0 + r, win, 1, out + r * Hq * D, scratch ? scratch + r * Hq * ns * (D + 2) : nullptr,
                        split_keys(1, kn, groups), splits(1, kn, groups));
        }
    }
    return 0;
    NS_CATCH
}

int ns_mg_rms_add(ns_gpu* g, const float* x, const float* w, float* res, int64_t rows, int64_t C, float eps) {
    NS_TRY
    constexpr int W = 256;
    if (rows > 0) g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int tid = (int) it.get_local_id(0);
        const float* xr = x + r * C;
        float ss = 0.0f;
        for (int64_t i = tid; i < C; i += W) ss += xr[i] * xr[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>()) / (float) C + eps);
        for (int64_t i = tid; i < C; i += W) res[r * C + i] += xr[i] * inv * w[i];
    });
    return 0;
    NS_CATCH
}

int ns_mg_put_cols(ns_gpu* g, const float* src, float* dst, int64_t T, int64_t n, int64_t ld, int64_t col0) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * n)), [=](sycl::id<1> id) {
        const int64_t i = id[0], t = i / n, c = i % n;
        dst[t * ld + col0 + c] = src[i];
    });
    return 0;
    NS_CATCH
}

int ns_mg_get_cols(ns_gpu* g, const float* src, float* dst, int64_t T, int64_t n, int64_t ld, int64_t col0) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * n)), [=](sycl::id<1> id) {
        const int64_t i = id[0], t = i / n, c = i % n;
        dst[i] = src[t * ld + col0 + c];
    });
    return 0;
    NS_CATCH
}

int ns_mg_sample(ns_gpu* g, const float* x, int32_t* out, int64_t T, int64_t n, float temp, float top_p, unsigned long long seed, int64_t pos0) {
    NS_TRY
    constexpr int W = 1024, K = 256;
    if (T <= 0) return 0;
    if (n < K) return ns_fail("ns_mg_sample: fewer than 256 values a row");
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> cv(sycl::range<1>(K), h);
        sycl::local_accessor<int32_t, 1> ci(sycl::range<1>(K), h);
        sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) T * W), sycl::range<1>(W)), [=](sycl::nd_item<1> it) {
            auto grp = it.get_group();
            const int64_t r = it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const float* xr = x + r * n;
            // a value's order as an unsigned key (larger value, larger key)
            auto key = [](float v) {
                const uint32_t b = sycl::bit_cast<uint32_t>(v);
                return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            };
            // the 256th largest key: built from the top bit down, keeping a bit while at least 256 keys are >= it
            uint32_t kth = 0;
            for (int bit = 31; bit >= 0; --bit) {
                const uint32_t c = kth | (1u << bit);
                int64_t m = 0;
                for (int64_t i = tid; i < n; i += W) m += key(xr[i]) >= c;
                if (sycl::reduce_over_group(grp, m, sycl::plus<int64_t>()) >= K) kth = c;
            }
            // the candidates: every key above it, then the equal ones in index order to 256 (contiguous ranges a work-item,
            // their counts scanned)
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
            // sorted: a higher value first, then a lower index (bitonic, 256 in local memory)
            for (int size = 2; size <= K; size <<= 1) {
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                    if (tid < K) {
                        const int partner = tid ^ stride;
                        if (partner > tid) {
                            const bool up = (tid & size) == 0;
                            const float va = cv[tid], vb = cv[partner];
                            const int32_t ia = ci[tid], ib = ci[partner];
                            const bool a_first = va > vb || (va == vb && ia < ib);
                            if (a_first != up) {
                                cv[tid] = vb;
                                cv[partner] = va;
                                ci[tid] = ib;
                                ci[partner] = ia;
                            }
                        }
                    }
                    sycl::group_barrier(grp);
                }
            }
            // the distribution and the draw, as the host's (one work-item: 256 values)
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

int ns_mg_argmax(ns_gpu* g, const float* x, int32_t* out, int64_t T, int64_t n) {
    NS_TRY
    constexpr int W = 1024;
    // a work-group a row: each work-item its stride, then the group's (value, index) maximum, the first of equals
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

int ns_mg_softcap(ns_gpu* g, float* x, int64_t n, float scale, float cap) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = x[id[0]] * scale;
        x[id[0]] = cap > 0.0f ? cap * sycl::tanh(v / cap) : v;
    });
    return 0;
    NS_CATCH
}

int ns_mg_gate_mul(ns_gpu* g, float* x, const float* gate, int64_t n) {
    NS_TRY
    if (n > 0) g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        x[i] *= 1.0f / (1.0f + sycl::exp(-gate[i]));
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
