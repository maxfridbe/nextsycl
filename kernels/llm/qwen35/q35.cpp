// q35.cpp: the qwen35 engine's own kernels (q35.h) - what dense Qwen3.5 / Qwen3.8 has that the shared ones do not:
// the gated full attention (head 256, a gate per head in the query projection, partial rotary positions) against a
// half cache, and DeltaNet's gates and SiLU-gated output norm. After llama.cpp's qwen35 graph (src/models/qwen35.cpp).
#include "ns.h"
#include "ns_internal.hpp"
#include "q35.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
using half = sycl::half;
constexpr int SG = 16;

// value i of a cache row of form kt (0 half, 1 q8_0, 2 q4_0: silo/attn.cpp has the layout)
inline float kv_value(const uint8_t* row, int kt, int i) {
    if (kt == 0) return (float) reinterpret_cast<const half*>(row)[i];
    const int b = i / 32, j = i % 32;
    if (kt == 1) return (float) (int8_t) row[i] * (float) reinterpret_cast<const half*>(row + 256)[b];
    const uint8_t w = row[b * 16 + (j & 15)];
    return (float) ((int) (j < 16 ? w & 15 : w >> 4) - 8) * (float) reinterpret_cast<const half*>(row + 128)[b];
}

// ---- cached causal attention (the MiniMax Music engine's, mm3.cpp): a work-group per (row, key head, split of the
// keys), NSG sub-groups taking the split's keys in turn, each lane DL features of the G query heads that share the key
// head; the sub-groups' (max, sum, weighted values) merged in local memory, the splits (decode) by a second kernel
constexpr int NSG_A = 4;

// a decode row's keys a split: enough splits that the card has ~512 work-groups (key heads x splits) whatever the
// context - a fixed 256 left 36 work-groups at 2K (8% of the bandwidth) - at least 32 keys each; prompt rows one split
int64_t split_keys(int64_t S, int64_t kv, int64_t Hkv) {
    if (S != 1) return kv;
    const int64_t want = (kv * Hkv + 511) / 512;
    return sycl::max<int64_t>(32, (want + 31) / 32 * 32);
}
int64_t splits(int64_t S, int64_t kv, int64_t Hkv) {
    const int64_t kb = split_keys(S, kv, Hkv);
    return kv > kb ? (kv + kb - 1) / kb : 1;
}

template <int DL, int G>
void attn(sycl::queue& q, const float* qp, int64_t qs, const half* kc, const half* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t T, int64_t p0,
          float* out, float* part) {
    constexpr int D = SG * DL, WG = SG * NSG_A;
    const int64_t ns = splits(S, p0 + S, Hkv), KB = split_keys(S, p0 + S, Hkv);
    const float scale = 1.0f / sycl::sqrt((float) D);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NSG_A * G * (D + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * Hkv * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, r = gi / Hkv;
            const int64_t kv_end = p0 + r + 1;
            const int64_t j0 = sp * (ns > 1 ? KB : kv_end), j1 = sycl::min(kv_end, ns > 1 ? (sp + 1) * KB : kv_end);
            float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float* qr = qp + r * qs + (hk * G + g) * D + lane * DL;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    qv[g][i] = qr[i] * scale;
                    acc[g][i] = 0.0f;
                }
                m[g] = -INFINITY;
                l[g] = 0.0f;
            }
            const half* kb = kc + hk * T * D + lane * DL;
            const half* vb = vc + hk * T * D + lane * DL;
            for (int64_t j = j0 + sgi; j < j1; j += NSG_A) {
                const sycl::vec<half, DL> kv = *reinterpret_cast<const sycl::vec<half, DL>*>(kb + j * D);
                const sycl::vec<half, DL> vv = *reinterpret_cast<const sycl::vec<half, DL>*>(vb + j * D);
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
                const int64_t hq = hk * G + g;
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
// floats as unsigned keys in the same order
inline uint32_t f_key(float f) {
    const uint32_t b = sycl::bit_cast<uint32_t>(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// A work-group's k largest of n keys (key(i)): a radix select a byte at a time, high byte first (a sub-group counts
// its lanes of a bin together - the scores crowd into a few bins), then emit(slot, i) for the keys above the k-th and
// as many equal to it as are left. hist: 256 local counters, st: 5
template <typename Key, typename Emit>
inline void group_topk(sycl::nd_item<1> it, int64_t n, int64_t k, Key key, Emit emit, uint32_t* hist, uint32_t* st) {
    using lref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
    auto sg = it.get_sub_group();
    const int lid = (int) it.get_local_id(0), wg = (int) it.get_local_range(0), lane = (int) sg.get_local_linear_id();
    if (lid == 0) {
        st[0] = 0;
        st[1] = 0;
        st[2] = (uint32_t) k;
        st[3] = 0;
        st[4] = 0;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lid; i < 256; i += wg) hist[i] = 0;
        sycl::group_barrier(it.get_group());
        const uint32_t prefix = st[0], mask = st[1];
        for (int64_t i0 = lid - lane; i0 < n; i0 += wg) {
            const int64_t i = i0 + lane;
            const uint32_t u = i < n ? key(i) : 0u;
            bool pending = i < n && (u & mask) == prefix;
            const uint32_t bin = (u >> shift) & 255;
            while (sycl::any_of_group(sg, pending)) {
                const uint32_t b = sycl::reduce_over_group(sg, pending ? bin : 256u, sycl::minimum<uint32_t>());
                const uint32_t c = sycl::reduce_over_group(sg, pending && bin == b ? 1u : 0u, sycl::plus<uint32_t>());
                if (lane == 0) lref(hist[b]).fetch_add(c);
                if (bin == b) pending = false;
            }
        }
        sycl::group_barrier(it.get_group());
        if (lid == 0) {
            const uint32_t need = st[2];
            uint32_t c = 0;
            int d = 255;
            for (; d > 0; --d) {
                if (c + hist[d] >= need) break;
                c += hist[d];
            }
            st[0] = prefix | ((uint32_t) d << shift);
            st[1] = mask | (255u << shift);
            st[2] = need - c;
        }
        sycl::group_barrier(it.get_group());
    }
    const uint32_t t = st[0], eq = st[2], above = (uint32_t) k - eq;
    for (int64_t i = lid; i < n; i += wg) {
        const uint32_t u = key(i);
        if (u > t) {
            emit(lref(st[3]).fetch_add(1u), i);
        } else if (u == t) {
            const uint32_t j = lref(st[4]).fetch_add(1u);
            if (j < eq) emit(above + j, i);
        }
    }
}

// the first stage's work-groups: each a slice's own top k (its keys and indices), the second picks among them
constexpr int TOPK_PARTS = 128;

}  // namespace

extern "C" {

int ns_q35_qk_norm_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                        const float* w, float eps, int64_t n_rot, float theta, int64_t p0) {
    NS_TRY
    if (T <= 0) return 0;
    if (D % SG != 0 || n_rot > D || n_rot % 2) return ns_fail("ns_q35_qk_norm_rope: D a multiple of 16, n_rot even and at most D");
    const int64_t half_rot = n_rot / 2;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t th = it.get_group(0), t = th / H, h = th % H;
        const float* x = src + t * ss + h * hs;
        float* y = dst + t * ds + h * D;
        float s = 0.0f;
        for (int64_t i = l; i < D; i += SG) s += x[i] * x[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) D + eps);
        const float pos = (float) (p0 + t);
        for (int64_t i = l; i < D; i += SG) {
            const float v = x[i] * inv * w[i];
            if (i < n_rot) {
                // rotate-half over the first n_rot: pair (i, i + n_rot / 2)
                const int64_t k = i < half_rot ? i : i - half_rot;
                const float a = pos * sycl::pow(theta, -2.0f * (float) k / (float) n_rot);
                const float c = sycl::cos(a), sn = sycl::sin(a);
                const int64_t o = i < half_rot ? i + half_rot : i - half_rot;
                const float other = x[o] * inv * w[o];
                y[i] = i < half_rot ? v * c - other * sn : v * c + other * sn;
            } else {
                y[i] = v;
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_q35_rope(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D, int64_t n_rot,
                float theta, int64_t p0) {
    NS_TRY
    if (T <= 0) return 0;
    if (D % SG != 0 || n_rot > D || n_rot % 2) return ns_fail("ns_q35_rope: D a multiple of 16, n_rot even and at most D");
    const int64_t half_rot = n_rot / 2;
    g->q.parallel_for(sycl::range<1>((size_t) (T * H * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0] % D, th = id[0] / D, t = th / H, h = th % H;
        const float* x = src + t * ss + h * hs;
        float* y = dst + t * ds + h * D;
        if (i >= n_rot) {
            y[i] = x[i];
            return;
        }
        // rotate-half over the first n_rot: pair (i, i + n_rot / 2)
        const int64_t k = i < half_rot ? i : i - half_rot;
        const float a = (float) (p0 + t) * sycl::pow(theta, -2.0f * (float) k / (float) n_rot);
        const float c = sycl::cos(a), sn = sycl::sin(a);
        const float other = x[i < half_rot ? i + half_rot : i - half_rot];
        y[i] = i < half_rot ? x[i] * c - other * sn : x[i] * c + other * sn;
    });
    return 0;
    NS_CATCH
}

int ns_q35_kv_store_q(ns_gpu* g, int kt, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap,
                      int64_t p0, void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    if (D != 256 || (kt != 1 && kt != 2)) return ns_fail("ns_q35_kv_store_q: heads of 256, form 1 (q8_0) or 2 (q4_0)");
    if (p0 + T > cap) return ns_fail("ns_q35_kv_store_q: past the cache's capacity");
    const int64_t rb = kt == 1 ? 272 : 144;
    uint8_t* kd = (uint8_t*) kc;
    uint8_t* vd = (uint8_t*) vc;
    // a work-item a block of 32 values (8 a row), keys then values
    g->q.parallel_for(sycl::range<1>((size_t) (2 * T * Hkv * 8)), [=](sycl::id<1> id) {
        int64_t i = id[0];
        const int b = (int) (i % 8);
        i /= 8;
        const int64_t h = i % Hkv;
        i /= Hkv;
        const int64_t t = i % T;
        const bool isv = i / T == 1;
        const float* x = (isv ? v + t * vs : k + t * ks) + h * 256 + b * 32;
        uint8_t* row = (isv ? vd : kd) + (h * cap + p0 + t) * rb;
        if (kt == 1) {
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(x[j]));
            const float d = amax / 127.0f, id_ = d != 0.0f ? 1.0f / d : 0.0f;
            for (int j = 0; j < 32; ++j) row[b * 32 + j] = (uint8_t) (int8_t) sycl::round(x[j] * id_);
            reinterpret_cast<half*>(row + 256)[b] = (half) d;
        } else {
            // ggml's quantize_row_q4_0: the value of the largest magnitude (the first of a tie), d = it / -8
            float m = 0.0f, am = 0.0f;
            for (int j = 0; j < 32; ++j) {
                if (sycl::fabs(x[j]) > am) {
                    am = sycl::fabs(x[j]);
                    m = x[j];
                }
            }
            const float d = m / -8.0f, id_ = d != 0.0f ? 1.0f / d : 0.0f;
            for (int j = 0; j < 16; ++j) {
                const int lo = sycl::min(15, (int) (x[j] * id_ + 8.5f)), hi = sycl::min(15, (int) (x[j + 16] * id_ + 8.5f));
                row[b * 16 + j] = (uint8_t) (lo | (hi << 4));
            }
            reinterpret_cast<half*>(row + 128)[b] = (half) d;
        }
    });
    return 0;
    NS_CATCH
}

int ns_q35_fwht32(ns_gpu* g, float* x, int64_t rows, int64_t n, int64_t stride) {
    NS_TRY
    if (rows <= 0) return 0;
    if (n % 32) return ns_fail("ns_q35_fwht32: rows of a multiple of 32");
    const int64_t nb = n / 32;
    g->q.parallel_for(sycl::range<1>((size_t) (rows * nb)), [=](sycl::id<1> id) {
        float* p = x + (id[0] / nb) * stride + (id[0] % nb) * 32;
        float v[32];
        for (int i = 0; i < 32; ++i) v[i] = p[i];
        for (int o = 1; o < 32; o <<= 1)
            for (int i = 0; i < 32; ++i)
                if (!(i & o)) {
                    const float a = v[i], b = v[i + o];
                    v[i] = a + b;
                    v[i + o] = a - b;
                }
        for (int i = 0; i < 32; ++i) p[i] = v[i] * 0.17677669529663687f;
    });
    return 0;
    NS_CATCH
}

int ns_q35_qk_norm_rope_pos(ns_gpu* g, const float* src, int64_t ss, int64_t hs, float* dst, int64_t ds, int64_t T, int64_t H, int64_t D,
                            const float* w, float eps, int64_t n_rot, float theta, const int32_t* pos, int64_t s0, int64_t s1, int64_t s2) {
    NS_TRY
    if (T <= 0) return 0;
    if (D % SG != 0 || n_rot > D || n_rot % 2 || s0 + s1 + s2 <= 0) return ns_fail("ns_q35_qk_norm_rope_pos: D a multiple of 16, n_rot even and at most D, sections");
    const int64_t half_rot = n_rot / 2, sect = s0 + s1 + s2;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t th = it.get_group(0), t = th / H, h = th % H;
        const float* x = src + t * ss + h * hs;
        float* y = dst + t * ds + h * D;
        float s = 0.0f;
        for (int64_t i = l; i < D; i += SG) s += x[i] * x[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) D + eps);
        for (int64_t i = l; i < D; i += SG) {
            const float v = x[i] * inv * w[i];
            if (i < n_rot) {
                const int64_t k = i < half_rot ? i : i - half_rot;
                // the interleaved sections (llama.cpp's IMROPE, Qwen3-VL's): pair k turns by the row's t, h or w position
                const int64_t sec = k % sect;
                const int c = (sec % 3 == 1 && sec < 3 * s1) ? 1 : (sec % 3 == 2 && sec < 3 * s2) ? 2 : 0;
                const float p = (float) pos[t * 3 + c];
                const float a = p * sycl::pow(theta, -2.0f * (float) k / (float) n_rot);
                const float c_ = sycl::cos(a), sn = sycl::sin(a);
                const int64_t o = i < half_rot ? i + half_rot : i - half_rot;
                const float other = x[o] * inv * w[o];
                y[i] = i < half_rot ? v * c_ - other * sn : v * c_ + other * sn;
            } else {
                y[i] = v;
            }
        }
    });
    return 0;
    NS_CATCH
}

// ---- the vision tower's (Qwen3-VL ViT in the mmproj file): a picture's patches, float32 rows

int ns_q35_vit_ln(ns_gpu* g, const float* x, const float* w, const float* b, float* y, int64_t rows, int64_t C, float eps) {
    NS_TRY
    if (rows <= 0) return 0;
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const float* xr = x + r * C;
        float s = 0.0f;
        for (int64_t i = it.get_local_id(0); i < C; i += WG) s += xr[i];
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float v = 0.0f;
        for (int64_t i = it.get_local_id(0); i < C; i += WG) {
            const float d = xr[i] - mean;
            v += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / (float) C + eps);
        for (int64_t i = it.get_local_id(0); i < C; i += WG) y[r * C + i] = (xr[i] - mean) * inv * w[i] + b[i];
    });
    return 0;
    NS_CATCH
}

int ns_q35_vit_bias_act(ns_gpu* g, float* y, const float* b, int64_t rows, int64_t N, int act) {
    NS_TRY
    if (rows <= 0) return 0;
    const int64_t n = rows * N;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t i = id[0];
        float v = y[i] + (b ? b[i % N] : 0.0f);
        if (act == 1) {
            // GELU, tanh form (gelu_pytorch_tanh)
            v = 0.5f * v * (1.0f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        } else if (act == 2) {
            // GELU, erf form (nn.GELU())
            v = 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
        }
        y[i] = v;
    });
    return 0;
    NS_CATCH
}

int ns_q35_vit_rope(ns_gpu* g, float* qkv, int64_t n, int64_t H, int64_t D, const float* cs, float* kh) {
    NS_TRY
    if (n <= 0) return 0;
    const int64_t C = H * D, half = D / 2;
    g->q.parallel_for(sycl::range<2>((size_t) (n * H), (size_t) half), [=](sycl::id<2> id) {
        const int64_t r = id[0] / H, h = id[0] % H, i = id[1];
        const float c = cs[(r * half + i) * 2], s = cs[(r * half + i) * 2 + 1];
        float* q = qkv + r * 3 * C + h * D;
        const float q0 = q[i], q1 = q[i + half];
        q[i] = q0 * c - q1 * s;
        q[i + half] = q1 * c + q0 * s;
        const float* k = qkv + r * 3 * C + C + h * D;
        const float k0 = k[i], k1 = k[i + half];
        float* o = kh + (h * n + r) * D;
        o[i] = k0 * c - k1 * s;
        o[i + half] = k1 * c + k0 * s;
    });
    return 0;
    NS_CATCH
}

int ns_q35_vit_heads(ns_gpu* g, const float* ob, int64_t H, int64_t r, int64_t D, float* out) {
    NS_TRY
    if (r <= 0) return 0;
    g->q.parallel_for(sycl::range<2>((size_t) r, (size_t) (H * D)), [=](sycl::id<2> id) {
        const int64_t i = id[0], j = id[1], hh = j / D, d = j % D;
        out[i * H * D + j] = ob[(hh * r + i) * D + d];
    });
    return 0;
    NS_CATCH
}

int ns_q35_attn_probs(ns_gpu* g, int kt, const float* q, const void* kc, int64_t Hq, int64_t Hkv, int64_t D, int64_t cap, int64_t n, int mode,
                      float* scratch, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    if (n > cap || Hq % Hkv || D != 256 || kt < 0 || kt > 2) return ns_fail("ns_q35_attn_probs: keys past the cache, heads that do not share, or not heads of 256");
    const uint8_t* k = (const uint8_t*) kc;
    const int64_t G = Hq / Hkv, rb = kt == 0 ? 512 : kt == 1 ? 272 : 144;
    const float scale = 1.0f / sycl::sqrt((float) D);
    auto score = [=](int64_t h, int64_t j) {
        const float* qh = q + h * D;
        const uint8_t* row = k + ((h / G) * cap + j) * rb;
        float s = 0.0f;
        for (int64_t i = 0; i < D; ++i) s += qh[i] * kv_value(row, kt, (int) i);
        return s * scale;
    };
    constexpr int WG = 256;
    // each head's softmax: its largest score and its sum
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) Hq * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t h = it.get_group(0);
        float m = -INFINITY;
        for (int64_t j = it.get_local_id(0); j < n; j += WG) m = sycl::fmax(m, score(h, j));
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float z = 0.0f;
        for (int64_t j = it.get_local_id(0); j < n; j += WG) z += sycl::exp(score(h, j) - m);
        z = sycl::reduce_over_group(it.get_group(), z, sycl::plus<float>());
        if (it.get_local_id(0) == 0) {
            scratch[h * 2] = m;
            scratch[h * 2 + 1] = z;
        }
    });
    // per key: the heads' weights, their mean (mode 0) or largest (1), added to out
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t j = id[0];
        float acc = 0.0f;
        for (int64_t h = 0; h < Hq; ++h) {
            const float p = sycl::exp(score(h, j) - scratch[h * 2]) / scratch[h * 2 + 1];
            acc = mode == 1 ? sycl::fmax(acc, p) : acc + p / (float) Hq;
        }
        out[j] += acc;
    });
    return 0;
    NS_CATCH
}

// ---- SYCL graphs of a decode pass's position-free stretches (the engine records each once a session and row count,
// then replays it: one submission for its ~hundred kernels)
namespace {
namespace X = sycl::ext::oneapi::experimental;
struct Q35Graph {
    std::unique_ptr<X::command_graph<X::graph_state::modifiable>> rec;
    std::unique_ptr<X::command_graph<X::graph_state::executable>> exec;
};
}  // namespace

int ns_q35_graph_begin(ns_gpu* g, void** out) {
    NS_TRY
    auto* gr = new Q35Graph();
    gr->rec = std::make_unique<X::command_graph<X::graph_state::modifiable>>(g->q.get_context(), g->q.get_device());
    gr->rec->begin_recording(g->q);
    *out = gr;
    return 0;
    NS_CATCH
}

int ns_q35_graph_end(ns_gpu* g, void* h) {
    NS_TRY
    auto* gr = static_cast<Q35Graph*>(h);
    gr->rec->end_recording(g->q);
    gr->exec = std::make_unique<X::command_graph<X::graph_state::executable>>(gr->rec->finalize());
    gr->rec.reset();
    return 0;
    NS_CATCH
}

int ns_q35_graph_launch(ns_gpu* g, void* h) {
    NS_TRY
    auto* gr = static_cast<Q35Graph*>(h);
    if (!gr->exec) return ns_fail("ns_q35_graph_launch: a graph not finished");
    g->q.ext_oneapi_graph(*gr->exec);
    return 0;
    NS_CATCH
}

void ns_q35_graph_free(void* h) {
    delete static_cast<Q35Graph*>(h);
}

int ns_q35_softmax_rows(ns_gpu* g, float* x, int64_t rows, int64_t n, float scale) {
    NS_TRY
    if (rows <= 0) return 0;
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        float* xr = x + it.get_group(0) * n;
        float m = -INFINITY;
        for (int64_t i = it.get_local_id(0); i < n; i += WG) m = sycl::fmax(m, xr[i] * scale);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t i = it.get_local_id(0); i < n; i += WG) {
            const float e = sycl::exp(xr[i] * scale - m);
            xr[i] = e;
            s += e;
        }
        const float inv = 1.0f / sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        for (int64_t i = it.get_local_id(0); i < n; i += WG) xr[i] *= inv;
    });
    return 0;
    NS_CATCH
}

int ns_q35_kv_store(ns_gpu* g, const float* k, int64_t ks, const float* v, int64_t vs, int64_t T, int64_t Hkv, int64_t D, int64_t cap, int64_t p0,
                    void* kc, void* vc) {
    NS_TRY
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_q35_kv_store: past the cache's capacity");
    half* ko = (half*) kc;
    half* vo = (half*) vc;
    g->q.parallel_for(sycl::range<1>((size_t) (T * Hkv * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % Hkv, t = i / (D * Hkv);
        const int64_t at = (h * cap + p0 + t) * D + d;
        ko[at] = (half) k[t * ks + h * D + d];
        vo[at] = (half) v[t * vs + h * D + d];
    });
    return 0;
    NS_CATCH
}

int64_t ns_q35_attn_scratch(int64_t T, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0) {
    const int64_t ns = splits(T, p0 + T, Hkv);
    return ns > 1 ? T * Hq * ns * (D + 2) : 0;
}

int ns_q35_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t T, int64_t Hq, int64_t Hkv, int64_t D,
                int64_t cap, int64_t p0, float* out, float* scratch) {
    NS_TRY
    if (T <= 0) return 0;
    if (p0 + T > cap) return ns_fail("ns_q35_attn: past the cache's capacity");
    if (ns_q35_attn_scratch(T, Hq, Hkv, D, p0) > 0 && !scratch) return ns_fail("ns_q35_attn: no scratch");
    const half* k = (const half*) kc;
    const half* v = (const half*) vc;
    if (D == 256 && Hq == 6 * Hkv) attn<16, 6>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else if (D == 256 && Hq == 4 * Hkv) attn<16, 4>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else if (D == 256 && Hq == 2 * Hkv) attn<16, 2>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else if (D == 256 && Hq == 8 * Hkv) attn<16, 8>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    // the EAGLE-4 draft's attention: heads of 128, a key head a query head
    else if (D == 128 && Hq == Hkv) attn<8, 1>(g->q, q, qs, k, v, T, Hq, Hkv, cap, p0, out, scratch);
    else return ns_fail("ns_q35_attn: head 256 with 2, 4, 6 or 8 query heads a key head, or 128 with one");
    return 0;
    NS_CATCH
}

int ns_q35_gate_mul(ns_gpu* g, float* out, const float* q_full, int64_t gs, int64_t T, int64_t H, int64_t D) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * H * D)), [=](sycl::id<1> id) {
        const int64_t i = id[0], d = i % D, h = (i / D) % H, t = i / (D * H);
        const float z = q_full[t * gs + h * 2 * D + D + d];
        out[i] *= 1.0f / (1.0f + sycl::exp(-z));
    });
    return 0;
    NS_CATCH
}

int ns_q35_gdn_gates(ns_gpu* g, const float* alpha, float* beta, const float* dt, const float* a, float* eg, int64_t T, int64_t H, int64_t d) {
    NS_TRY
    if (T <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (T * H * d)), [=](sycl::id<1> id) {
        const int64_t i = id[0], h = (i / d) % H, t = i / (d * H);
        const float x = alpha[t * H + h] + dt[h];
        // ggml's softplus: x past 20 as it is
        const float sp = x > 20.0f ? x : sycl::log1p(sycl::exp(x));
        eg[i] = sycl::exp(sp * a[h]);
    });
    g->q.parallel_for(sycl::range<1>((size_t) (T * H)), [=](sycl::id<1> i) { beta[i] = 1.0f / (1.0f + sycl::exp(-beta[i])); });
    return 0;
    NS_CATCH
}

int ns_q35_expand(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t T, int64_t Hk, int64_t Hv, int64_t d) {
    NS_TRY
    if (T > 0) g->q.parallel_for(sycl::range<1>((size_t) (T * Hv * d)), [=](sycl::id<1> id) {
        const int64_t i = id[0], c = i % d, h = (i / d) % Hv, t = i / (d * Hv);
        dst[i] = src[t * ss + (h % Hk) * d + c];
    });
    return 0;
    NS_CATCH
}

int ns_q35_gdn_out(ns_gpu* g, const float* o, const float* z, int64_t zs, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps) {
    NS_TRY
    if (T <= 0) return 0;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (T * H) * SG), sycl::range<1>(SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int l = (int) sg.get_local_linear_id();
        const int64_t r = it.get_group(0), t = r / H, h = r % H;
        const float* x = o + r * d;
        float s = 0.0f;
        for (int64_t i = l; i < d; i += SG) s += x[i] * x[i];
        const float inv = sycl::rsqrt(sycl::reduce_over_group(sg, s, sycl::plus<float>()) / (float) d + eps);
        for (int64_t i = l; i < d; i += SG) {
            const float zz = z[t * zs + h * d + i];
            y[r * d + i] = x[i] * inv * w[i] * (zz / (1.0f + sycl::exp(-zz)));
        }
    });
    return 0;
    NS_CATCH
}

int64_t ns_q35_topk_scratch(int64_t n, int64_t k) {
    return n > 8 * TOPK_PARTS * k ? 2 * TOPK_PARTS * k : 0;
}

int ns_q35_topk(ns_gpu* g, const float* x, int64_t n, int64_t k, int32_t* ids, void* scratch) {
    NS_TRY
    if (k <= 0 || k > n) return ns_fail("ns_q35_topk: k from 1 to n");
    if (ns_q35_topk_scratch(n, k) == 0) {
        g->q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(256), h), st(sycl::range<1>(5), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                group_topk(it, n, k, [=](int64_t i) { return f_key(x[i]); }, [=](uint32_t slot, int64_t i) { ids[slot] = (int32_t) i; }, &hist[0], &st[0]);
            });
        });
        return 0;
    }
    if (!scratch) return ns_fail("ns_q35_topk: no scratch");
    uint32_t* ck = (uint32_t*) scratch;
    int32_t* ci = (int32_t*) scratch + TOPK_PARTS * k;
    const int64_t slice = (n + TOPK_PARTS - 1) / TOPK_PARTS;
    // each slice's top k (a slice of fewer than k: all of it, the rest of its slots the lowest key)
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(256), h), st(sycl::range<1>(5), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) TOPK_PARTS * 256), sycl::range<1>(256)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int64_t p = it.get_group(0), a = p * slice, m = sycl::max<int64_t>(0, sycl::min<int64_t>(slice, n - a));
            uint32_t* pk = ck + p * k;
            int32_t* pi = ci + p * k;
            if (m <= k) {
                for (int64_t j = it.get_local_id(0); j < k; j += 256) {
                    pk[j] = j < m ? f_key(x[a + j]) : 0u;
                    pi[j] = j < m ? (int32_t) (a + j) : 0;
                }
                return;
            }
            group_topk(it, m, k, [=](int64_t i) { return f_key(x[a + i]); }, [=](uint32_t slot, int64_t i) {
                pk[slot] = f_key(x[a + i]);
                pi[slot] = (int32_t) (a + i);
            }, &hist[0], &st[0]);
        });
    });
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(256), h), st(sycl::range<1>(5), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            group_topk(it, TOPK_PARTS * k, k, [=](int64_t i) { return ck[i]; }, [=](uint32_t slot, int64_t i) { ids[slot] = ci[i]; }, &hist[0], &st[0]);
        });
    });
    return 0;
    NS_CATCH
}

int ns_q35_scores_i8(ns_gpu* g, const void* qv, const void* dv, int64_t rows, const float* x, float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    const int8_t* q = (const int8_t*) qv;
    const half* d = (const half*) dv;
    // rows of 128: a sub-group a row, a lane 8 values (one 8-byte load), its block's scale every fourth lane's
    constexpr int NSG = 8;
    const int64_t groups = (rows + NSG - 1) / NSG;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * NSG * SG), sycl::range<1>(NSG * SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0) * NSG + sg.get_group_linear_id();
        if (r >= rows) return;
        const int lane = (int) sg.get_local_linear_id();
        const sycl::vec<int8_t, 8> v = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(q + r * 128 + lane * 8);
        const float* xb = x + lane * 8;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += (float) v[j] * xb[j];
        acc *= (float) d[r * 4 + lane / 4];
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) out[r] = acc;
    });
    return 0;
    NS_CATCH
}

int ns_q35_rows_q8_0(ns_gpu* g, const void* w, int64_t cols, const int32_t* ids, int64_t rows, const float* x, float* out) {
    NS_TRY
    if (rows <= 0) return 0;
    if (cols % 32) return ns_fail("ns_q35_rows_q8_0: columns a multiple of 32");
    constexpr int NSG = 4;
    const int64_t rb = cols / 32 * 34;
    const int64_t groups = (rows + NSG - 1) / NSG;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * NSG * SG), sycl::range<1>(NSG * SG)),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0) * NSG + sg.get_group_linear_id();
        if (r >= rows) return;
        const int lane = (int) sg.get_local_linear_id();
        const uint8_t* row = (const uint8_t*) w + (int64_t) (ids ? ids[r] : r) * rb;
        float acc = 0.0f;
        // lane l takes values l, l + 16, ...: a block's 32 values two lanes' turns
        for (int64_t e = lane; e < cols; e += SG) {
            const uint8_t* b = row + (e / 32) * 34;
            acc += (float) *reinterpret_cast<const half*>(b) * (float) (int8_t) b[2 + e % 32] * x[e];
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) out[r] = acc;
    });
    return 0;
    NS_CATCH
}

int ns_q35_argmax(ns_gpu* g, const float* x, int64_t rows, int64_t n, int32_t* out) {
    NS_TRY
    // one row over many work-groups (a 248K vocabulary walked by one took 318 us): each its slice's best, then one
    // work-group the best of those - the lowest index on a tie, as before
    if (rows == 1 && n >= 65536) {
        constexpr int WG = 256, PARTS = 256;
        // the slices' bests: a small buffer a GPU, kept for the process
        static std::mutex mu;
        static std::unordered_map<ns_gpu*, void*> scratch;
        void* sp;
        {
            std::lock_guard<std::mutex> l(mu);
            auto& e = scratch[g];
            if (!e) e = sycl::malloc_device(PARTS * 8, g->q);
            sp = e;
        }
        float* bv = static_cast<float*>(sp);
        int32_t* bi = reinterpret_cast<int32_t*>(bv + PARTS);
        const int64_t per = (n + PARTS - 1) / PARTS;
        g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>(PARTS * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
            const int64_t p = it.get_group(0), lo = p * per, hi = sycl::min(n, lo + per);
            float best = -INFINITY;
            int32_t idx = INT32_MAX;
            for (int64_t i = lo + it.get_local_id(0); i < hi; i += WG) {
                const float v = x[i];
                if (v > best) {
                    best = v;
                    idx = (int32_t) i;
                }
            }
            const float m = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
            const int32_t j = sycl::reduce_over_group(it.get_group(), best == m ? idx : INT32_MAX, sycl::minimum<int32_t>());
            if (it.get_local_id(0) == 0) {
                bv[p] = m;
                bi[p] = j;
            }
        });
        g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>(PARTS), sycl::range<1>(PARTS)), [=](sycl::nd_item<1> it) {
            const float v = bv[it.get_local_id(0)];
            const float m = sycl::reduce_over_group(it.get_group(), v, sycl::maximum<float>());
            const int32_t j = sycl::reduce_over_group(it.get_group(), v == m ? bi[it.get_local_id(0)] : INT32_MAX, sycl::minimum<int32_t>());
            if (it.get_local_id(0) == 0) out[0] = j;
        });
        return 0;
    }
    constexpr int WG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        float best = -INFINITY;
        int32_t bi = 0;
        for (int64_t i = it.get_local_id(0); i < n; i += WG) {
            const float v = x[r * n + i];
            if (v > best) {
                best = v;
                bi = (int32_t) i;
            }
        }
        const float m = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
        const int32_t cand = best == m ? bi : INT32_MAX;
        const int32_t idx = sycl::reduce_over_group(it.get_group(), cand, sycl::minimum<int32_t>());
        if (it.get_local_id(0) == 0) out[r] = idx;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
