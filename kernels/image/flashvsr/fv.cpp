// fv.cpp: the FlashVSR engine's own kernels - what the shared diffusion kernels (nsd) do not have: the DiT's tokens in
// window order (patches in and out, the LQ projection's residual), its norms and gated residuals, the block-sparse
// attention's draft (the scores between block means, the top-k threshold, the lists of selected key blocks) and a
// plain kernel for the attention itself (the silo's sycl-tla kernel is the fast one), the LQ projection's causal 3-D
// convolution laid out as a 2-D one, the tiny decoder's input and upsample, the colour fix and the noise. A port of
// nextcuda's fv.cu (kernels/image/flashvsr) to SYCL; half buffers, float32 arithmetic, on the GPU's queue.
#include "ns.h"
#include "ns_internal.hpp"
#include "fv.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <string>

namespace {
using half = sycl::half;
constexpr int WG = 256;

inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
inline size_t up(int64_t n, int64_t m) { return sz((n + m - 1) / m * m); }

// token index (window order) of patch (frame t, y, x) in a grid of f x hp x wp patches
inline int64_t win_index(int64_t t, int64_t y, int64_t x, int64_t hp, int64_t wp) {
    const int64_t nbw = wp / 8, nb = (hp / 8) * nbw;
    const int64_t g = t >> 1, ft = t & 1;
    const int64_t b = (y >> 3) * nbw + (x >> 3);
    return ((g * nb + b) << 7) + ft * 64 + (y & 7) * 8 + (x & 7);
}

inline uint64_t splitmix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

template <typename F>
int each(ns_gpu* g, int64_t n, F f) {
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>(sz(n)), [=](sycl::id<1> i) { f((int64_t) i[0]); });
    return 0;
}
}  // namespace

extern "C" {

// ---- the DiT's tokens ---------------------------------------------------------------------------------------------

int ns_image_flashvsr_patchify(ns_gpu* g, const float* lat, int64_t f, int64_t h8, int64_t w8, void* out_) {
    NS_TRY
    if (f % 2 || h8 % 16 || w8 % 16) return ns_fail("ns_image_flashvsr_patchify: frames in pairs, sides multiples of 16 latents");
    half* out = (half*) out_;
    const int64_t hp = h8 / 2, wp = w8 / 2;
    return each(g, f * hp * wp * 64, [=](int64_t id) {
        const int64_t k = id % 64, p = id / 64;
        const int64_t x = p % wp, y = (p / wp) % hp, t = p / (wp * hp);
        const int64_t c = k / 4, ky = (k / 2) % 2, kx = k % 2;
        out[win_index(t, y, x, hp, wp) * 64 + k] = (half) lat[((c * f + t) * h8 + 2 * y + ky) * w8 + 2 * x + kx];
    });
    NS_CATCH
}

int ns_image_flashvsr_unpatchify_sub(ns_gpu* g, const float* pred, const float* noise, int64_t f, int64_t h8, int64_t w8, float* out) {
    NS_TRY
    const int64_t hp = h8 / 2, wp = w8 / 2;
    return each(g, 16 * f * h8 * w8, [=](int64_t id) {
        const int64_t xx = id % w8, yy = (id / w8) % h8, t = (id / (w8 * h8)) % f, c = id / (w8 * h8 * f);
        // the head's features are (ky, kx, c): Wan's unpatchify, '(x y z c)'
        const int64_t k = ((yy % 2) * 2 + (xx % 2)) * 16 + c;
        out[id] = noise[id] - pred[win_index(t, yy / 2, xx / 2, hp, wp) * 64 + k];
    });
    NS_CATCH
}

int ns_image_flashvsr_add_lq(ns_gpu* g, float* x, const float* lq, int64_t f, int64_t hp, int64_t wp, int64_t C) {
    NS_TRY
    return each(g, f * hp * wp * C, [=](int64_t id) {
        const int64_t c = id % C, p = id / C;
        const int64_t xx = p % wp, yy = (p / wp) % hp, t = p / (wp * hp);
        x[win_index(t, yy, xx, hp, wp) * C + c] += lq[id];
    });
    NS_CATCH
}

int ns_image_flashvsr_ln(ns_gpu* g, const float* x, int64_t M, int64_t C, float eps, const float* a, const float* b, void* out_) {
    NS_TRY
    if (M <= 0) return 0;
    half* out = (half*) out_;
    g->q.parallel_for(sycl::nd_range<1>(sz(M) * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        const float* row = x + r * C;
        float s = 0.0f;
        for (int64_t i = l; i < C; i += WG) s += row[i];
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C;
        float s2 = 0.0f;
        for (int64_t i = l; i < C; i += WG) {
            const float d = row[i] - mean;
            s2 += d * d;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s2, sycl::plus<float>()) / (float) C + eps);
        half* o = out + r * C;
        for (int64_t i = l; i < C; i += WG) {
            float v = (row[i] - mean) * inv;
            if (a) v *= a[i];
            if (b) v += b[i];
            o[i] = (half) v;
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_flashvsr_gate_add(ns_gpu* g, float* x, const float* y, int64_t M, int64_t C, const float* gate) {
    NS_TRY
    return each(g, M * C, [=](int64_t id) { x[id] += gate ? y[id] * gate[id % C] : y[id]; });
    NS_CATCH
}

int ns_image_flashvsr_rms_rope(ns_gpu* g, void* x_, int64_t M, int64_t C, int64_t ld, const float* w, float eps, const float* ts, const float* hw,
                               int64_t hp, int64_t wp) {
    NS_TRY
    if (M <= 0) return 0;
    if (C % 128) return ns_fail("ns_image_flashvsr_rms_rope: heads of 128");
    if (ts && (hp % 8 || wp % 8 || hp <= 0 || wp <= 0)) return ns_fail("ns_image_flashvsr_rms_rope: a grid of whole 8 x 8 windows");
    const int64_t nbw = wp / 8, nb = (hp / 8) * nbw;
    half* x = (half*) x_;
    g->q.parallel_for(sycl::nd_range<1>(sz(M) * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        half* row = x + r * ld;
        float s = 0.0f;
        for (int64_t i = l; i < C / 2; i += WG) {
            const float a = (float) row[2 * i], b = (float) row[2 * i + 1];
            s += a * a + b * b;
        }
        const float inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) C + eps);
        // the token's frame and patch: window order (fv.h)
        const int64_t blk = r >> 7, in = r & 127, b = blk % nb;
        const int64_t t = 2 * (blk / nb) + (in >> 6), y = (b / nbw) * 8 + ((in >> 3) & 7), xx = (b % nbw) * 8 + (in & 7);
        for (int64_t i = l; i < C / 2; i += WG) {
            const float a = (float) row[2 * i] * inv * w[2 * i], b = (float) row[2 * i + 1] * inv * w[2 * i + 1];
            if (ts) {
                const int64_t pr = i % 64;
                const float* p = pr < 22 ? ts + (t * 22 + pr) * 2 : hw + ((y * wp + xx) * 42 + pr - 22) * 2;
                row[2 * i] = (half) (a * p[0] - b * p[1]);
                row[2 * i + 1] = (half) (a * p[1] + b * p[0]);
            } else {
                row[2 * i] = (half) a;
                row[2 * i + 1] = (half) b;
            }
        }
    });
    return 0;
    NS_CATCH
}

// ---- the draft ----------------------------------------------------------------------------------------------------

int ns_image_flashvsr_block_mean(ns_gpu* g, const void* x_, int64_t n, int64_t stride, int64_t C, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    const half* x = (const half*) x_;
    g->q.parallel_for(sycl::nd_range<1>(sz(n) * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t blk = it.get_group(0);
        for (int64_t c = it.get_local_id(0); c < C; c += WG) {
            float s = 0.0f;
            for (int r = 0; r < 128; ++r) s += (float) x[(blk * 128 + r) * stride + c];
            out[blk * C + c] = s / 128.0f;
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_flashvsr_block_probs(ns_gpu* g, const float* qm, const float* km, int64_t nq, int64_t nk, int64_t H, int64_t nbh, int64_t nbw,
                                  int64_t lr, const int32_t* valid, float* probs) {
    NS_TRY
    if (nq <= 0 || nk <= 0) return 0;
    if (nk > 8192) return ns_fail("ns_image_flashvsr_block_probs: more than 8192 key blocks");
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sc(sycl::range<1>(sz(nk)), h);
        sycl::local_accessor<float, 1> qv(sycl::range<1>(128), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(sz(H), sz(nq) * WG), sycl::range<2>(1, WG)), [=](sycl::nd_item<2> it) {
            const int64_t hh = it.get_group(0), i = it.get_group(1);
            const int l = (int) it.get_local_id(1);
            const int64_t C = H * 128, nb = nbh * nbw;
            if (l < 128) qv[l] = qm[i * C + hh * 128 + l];
            sycl::group_barrier(it.get_group());
            const int64_t qb = i % nb, qr = qb / nbw, qc = qb % nbw, half_ = lr / 2;
            const int64_t r0 = qr - half_, c0 = qc - half_;
            const float scale = 1.0f / sycl::sqrt(128.0f);
            float mx = -INFINITY;
            for (int64_t j = l; j < nk; j += WG) {
                const int64_t kb = j % nb, kr = kb / nbw, kc = kb % nbw;
                float s = -INFINITY;
                if (kr >= r0 && kr <= r0 + lr - 1 && kc >= c0 && kc <= c0 + lr - 1 && (!valid || valid[j / nb])) {
                    const float* kv = km + j * C + hh * 128;
                    float d = 0.0f;
                    for (int e = 0; e < 128; ++e) d += qv[e] * kv[e];
                    s = d * scale;
                }
                sc[j] = s;
                mx = sycl::fmax(mx, s);
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t j = l; j < nk; j += WG) {
                const float e = sc[j] == -INFINITY ? 0.0f : sycl::exp(sc[j] - mx);
                sc[j] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            float* o = probs + (hh * nq + i) * nk;
            for (int64_t j = l; j < nk; j += WG) o[j] = sc[j] / sum;
        });
    });
    return 0;
    NS_CATCH
}

// the rank-th largest of a segment's values (all >= 0, so their bits order as they do): a radix select, 8 bits a pass
int ns_image_flashvsr_topk_thr(ns_gpu* g, const float* p, int64_t S, int64_t n, int64_t rank, float* thr) {
    NS_TRY
    if (S <= 0) return 0;
    constexpr int T = 1024;
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(256), h);
        sycl::local_accessor<uint32_t, 1> st(sycl::range<1>(2), h);
        h.parallel_for(sycl::nd_range<1>(sz(S) * T, T), [=](sycl::nd_item<1> it) {
            const int l = (int) it.get_local_id(0);
            const float* d = p + it.get_group(0) * n;
            uint32_t prefix = 0, mask = 0;
            int64_t r = rank;
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (int i = l; i < 256; i += T) hist[i] = 0;
                sycl::group_barrier(it.get_group());
                for (int64_t i = l; i < n; i += T) {
                    const uint32_t u = sycl::bit_cast<uint32_t>(sycl::fmax(d[i], 0.0f));
                    if ((u & mask) == prefix) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
                            a(hist[(u >> shift) & 255u]);
                        a.fetch_add(1u);
                    }
                }
                sycl::group_barrier(it.get_group());
                if (l == 0) {
                    int64_t acc = 0;
                    int dg = 255;
                    for (; dg > 0; --dg) {
                        if (acc + hist[dg] >= r) break;
                        acc += hist[dg];
                    }
                    st[0] = prefix | ((uint32_t) dg << shift);
                    st[1] = (uint32_t) (r - acc);
                }
                sycl::group_barrier(it.get_group());
                prefix = st[0];
                r = st[1];
                mask |= 255u << shift;
                sycl::group_barrier(it.get_group());
            }
            if (l == 0) thr[it.get_group(0)] = sycl::bit_cast<float>(prefix);
        });
    });
    return 0;
    NS_CATCH
}

int ns_image_flashvsr_block_lists(ns_gpu* g, const float* probs, const float* thr, int64_t H, int64_t nq, int64_t nk, int64_t nb, int delta,
                                  int64_t rep, int32_t* lists, int32_t* counts) {
    NS_TRY
    if (rep < 1) return ns_fail("ns_image_flashvsr_block_lists: rep of at least 1");
    const int64_t G = nq / nb;
    return each(g, H * nq, [=](int64_t w) {
        const int64_t hh = w / nq, i = w % nq;
        const float t = thr[hh * G + i / nb];
        const float* row = probs + w * nk;
        int32_t* out = lists + w * rep * nk;
        int32_t cnt = 0, last = 0;
        for (int64_t j = 0; j < nk; ++j) {
            if (row[j] > t) {
                out[cnt++] = delta ? (int32_t) j - last : (int32_t) j;
                last = (int32_t) j;
            }
        }
        for (int64_t j = cnt; j < nk; ++j) out[j] = 0;
        for (int64_t r = 1; r < rep; ++r)
            for (int64_t j = 0; j < nk; ++j) out[r * nk + j] = out[j];
        for (int64_t r = 0; r < rep; ++r) counts[w * rep + r] = cnt;
    });
    NS_CATCH
}

int ns_image_flashvsr_zero_empty(ns_gpu* g, void* out_, int64_t H, int64_t nr, int64_t qrow, const int32_t* counts) {
    NS_TRY
    half* out = (half*) out_;
    const int64_t ld = H * 128;
    return each(g, H * nr * qrow * 16, [=](int64_t id) {
        const int64_t c8 = id % 16, row = (id / 16) % (nr * qrow), hh = id / (16 * nr * qrow);
        if (counts[hh * nr + row / qrow] != 0) return;
        half* o = out + row * ld + hh * 128 + c8 * 8;
        for (int e = 0; e < 8; ++e) o[e] = (half) 0.0f;
    });
    NS_CATCH
}

// The plain block-sparse attention: a sub-group of 16 a query row (8 features a lane), 16 rows a work-group; keys read
// from global memory. Correct for every list; the silo's kernel is the fast one.
int ns_image_flashvsr_sparse_attn(ns_gpu* g, const void* q_, int64_t Sq, int64_t H, const void* k_, const void* v_, int64_t kv_rs, const int32_t* lists,
                                  const int32_t* counts, int64_t nk, void* out_) {
    NS_TRY
    if (Sq % 128) return ns_fail("ns_image_flashvsr_sparse_attn: query rows in whole blocks of 128");
    const half* q = (const half*) q_;
    const half* k = (const half*) k_;
    const half* v = (const half*) v_;
    half* out = (half*) out_;
    const int64_t nq = Sq / 128;
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>(sz(H), sz(Sq) * 16), sycl::range<2>(1, 256)), [=](sycl::nd_item<2> it)
                      [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t hh = it.get_group(0);
        const int64_t row = it.get_global_id(1) / 16;
        const int64_t ld = H * 128, qb = row / 128;
        float qv[8], acc[8];
        const float scale = 1.0f / sycl::sqrt(128.0f);
        for (int e = 0; e < 8; ++e) {
            qv[e] = (float) q[row * ld + hh * 128 + lane * 8 + e] * scale;
            acc[e] = 0.0f;
        }
        float m = -INFINITY, l = 0.0f;
        const int32_t* list = lists + (hh * nq + qb) * nk;
        const int32_t cnt = counts[hh * nq + qb];
        for (int32_t li = 0; li < cnt; ++li) {
            const int64_t j = list[li];
            for (int r = 0; r < 128; ++r) {
                const half* kr = k + (j * 128 + r) * kv_rs + hh * 128 + lane * 8;
                float d = 0.0f;
                for (int e = 0; e < 8; ++e) d += qv[e] * (float) kr[e];
                const float s = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                const float mn = sycl::fmax(m, s);
                const float c = sycl::exp(m - mn), pe = sycl::exp(s - mn);
                l = l * c + pe;
                const half* vr = v + (j * 128 + r) * kv_rs + hh * 128 + lane * 8;
                for (int e = 0; e < 8; ++e) acc[e] = acc[e] * c + pe * (float) vr[e];
                m = mn;
            }
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int e = 0; e < 8; ++e) out[row * ld + hh * 128 + lane * 8 + e] = (half) (acc[e] * inv);
    });
    return 0;
    NS_CATCH
}

// ---- the LQ projection --------------------------------------------------------------------------------------------

int ns_image_flashvsr_unshuffle(ns_gpu* g, const void* x_, int64_t T, int64_t H, int64_t W, int64_t ff, int64_t r, void* out_) {
    NS_TRY
    const half* x = (const half*) x_;
    half* out = (half*) out_;
    const int64_t Ho = H / r, Wo = W / r, Co = 3 * ff * r * r;
    return each(g, (T / ff) * Ho * Wo * Co, [=](int64_t id) {
        const int64_t ch = id % Co, p = id / Co;
        const int64_t xo = p % Wo, yo = (p / Wo) % Ho, to = p / (Wo * Ho);
        const int64_t xx = ch % r, yy = (ch / r) % r, tt = (ch / (r * r)) % ff, c = ch / (r * r * ff);
        out[id] = x[(((to * ff + tt) * H) + yo * r + yy) * W * 3 + (xo * r + xx) * 3 + c];
    });
    NS_CATCH
}

int ns_image_flashvsr_stack_taps(ns_gpu* g, const void* x_, int64_t T, int64_t H, int64_t W, int64_t C, int64_t kt, int64_t st, void* out_) {
    NS_TRY
    if (C % 8) return ns_fail("ns_image_flashvsr_stack_taps: channels in eights");
    if (T < kt) return ns_fail("ns_image_flashvsr_stack_taps: fewer frames than the kernel's depth");
    using v8 = sycl::vec<uint16_t, 8>;
    const v8* x = (const v8*) x_;
    v8* out = (v8*) out_;
    const int64_t To = (T - kt) / st + 1, C8 = C / 8, K8 = kt * C8, Hp = H + 2, Wp = W + 2;
    return each(g, To * Hp * Wp * K8, [=](int64_t id) {
        const int64_t c = id % K8, p = id / K8;
        const int64_t xx = p % Wp, yy = (p / Wp) % Hp, to = p / (Wp * Hp);
        const int64_t dt = c / C8, cc = c % C8;
        const int64_t yi = sycl::clamp(yy - 1, (int64_t) 0, H - 1), xi = sycl::clamp(xx - 1, (int64_t) 0, W - 1);
        out[id] = x[(((to * st + dt) * H + yi) * W + xi) * C8 + cc];
    });
    NS_CATCH
}

int ns_image_flashvsr_rms_silu(ns_gpu* g, void* x_, int64_t M, int64_t C, const float* gamma) {
    NS_TRY
    if (M <= 0) return 0;
    half* x = (half*) x_;
    g->q.parallel_for(sycl::nd_range<1>(sz(M) * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int l = (int) it.get_local_id(0);
        half* row = x + r * C;
        float s = 0.0f;
        for (int64_t i = l; i < C; i += WG) {
            const float v = (float) row[i];
            s += v * v;
        }
        const float nrm = sycl::fmax(sycl::sqrt(sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>())), 1e-12f);
        const float k = sycl::sqrt((float) C) / nrm;
        for (int64_t i = l; i < C; i += WG) {
            const float v = (float) row[i] * k * gamma[i];
            row[i] = (half) (v / (1.0f + sycl::exp(-v)));
        }
    });
    return 0;
    NS_CATCH
}

// ---- the tiny decoder ---------------------------------------------------------------------------------------------

int ns_image_flashvsr_tcd_input(ns_gpu* g, const void* cond_, const float* lat, int64_t f, int64_t t, int64_t h8, int64_t w8, int64_t ld, void* out_) {
    NS_TRY
    if (ld < 784) return ns_fail("ns_image_flashvsr_tcd_input: rows of at least 784");
    const half* cond = (const half*) cond_;
    half* out = (half*) out_;
    return each(g, h8 * w8 * ld, [=](int64_t id) {
        const int64_t ch = id % ld, p = id / ld;
        const int64_t x = p % w8, y = p / w8;
        float v = 0.0f;
        if (ch < 768) {
            const int64_t xx = ch % 8, yy = (ch / 8) % 8, tt = (ch / 64) % 4, c = ch / 256;
            const int64_t Hh = h8 * 8, Ww = w8 * 8;
            v = (float) cond[((tt * Hh + y * 8 + yy) * Ww + x * 8 + xx) * 3 + c];
        } else if (ch < 784) {
            v = lat[(((ch - 768) * f + t) * h8 + y) * w8 + x];
        } else {
            out[id] = (half) 0.0f;
            return;
        }
        out[id] = (half) (sycl::tanh(v / 3.0f) * 3.0f);
    });
    NS_CATCH
}

int ns_image_flashvsr_up2(ns_gpu* g, const void* x_, int64_t H, int64_t W, int64_t C, int64_t ldx, int64_t c0, int64_t r0, int64_t rows, void* out_) {
    NS_TRY
    if (C % 8 || ldx % 8 || c0 % 8) return ns_fail("ns_image_flashvsr_up2: channels in eights");
    if (r0 < 0 || r0 + rows > 2 * H) return ns_fail("ns_image_flashvsr_up2: rows outside the upsample");
    using v8 = sycl::vec<uint16_t, 8>;
    const v8* x = (const v8*) x_;
    v8* out = (v8*) out_;
    const int64_t C8 = C / 8, L8 = ldx / 8, o8 = c0 / 8, Wo = 2 * W;
    return each(g, rows * Wo * C8, [=](int64_t id) {
        const int64_t c = id % C8, p = id / C8;
        const int64_t xx = p % Wo, yy = r0 + p / Wo;
        out[id] = x[((yy / 2) * W + xx / 2) * L8 + o8 + c];
    });
    NS_CATCH
}

// ---- frames, colour, noise ----------------------------------------------------------------------------------------

int ns_image_flashvsr_u8_to_half(ns_gpu* g, const uint8_t* x, int64_t n, void* out_) {
    NS_TRY
    half* out = (half*) out_;
    return each(g, n, [=](int64_t i) { out[i] = (half) ((float) x[i] / 255.0f * 2.0f - 1.0f); });
    NS_CATCH
}

int ns_image_flashvsr_to_half(ns_gpu* g, const float* x, int64_t n, void* out_) {
    NS_TRY
    half* out = (half*) out_;
    return each(g, n, [=](int64_t i) { out[i] = (half) x[i]; });
    NS_CATCH
}

int ns_image_flashvsr_to_float(ns_gpu* g, const void* x_, int64_t n, float* out) {
    NS_TRY
    const half* x = (const half*) x_;
    return each(g, n, [=](int64_t i) { out[i] = (float) x[i]; });
    NS_CATCH
}

int ns_image_flashvsr_adain(ns_gpu* g, const void* dec_, const void* lq_, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, float* scratch,
                            float* ms) {
    NS_TRY
    if (w <= 0 || h <= 0) return ns_fail("ns_image_flashvsr_adain: an empty box");
    const half* dec = (const half*) dec_;
    const half* lq = (const half*) lq_;
    constexpr int NG = 256;
    // partial sums: group gi, picture pi (0 the decoded, 1 the input): [pi][gi][6]
    g->q.parallel_for(sycl::nd_range<2>(sycl::range<2>(2, NG * WG), sycl::range<2>(1, WG)), [=](sycl::nd_item<2> it) {
        const int pi = (int) it.get_group(0);
        const int64_t gi = it.get_group(1);
        const half* x = pi == 0 ? dec : lq;
        const float a = pi == 0 ? 2.0f : 1.0f, b = pi == 0 ? -1.0f : 0.0f;
        float s[6] = {0, 0, 0, 0, 0, 0};
        for (int64_t i = gi * WG + it.get_local_id(1); i < w * h; i += (int64_t) NG * WG) {
            const int64_t xx = x0 + i % w, yy = y0 + i / w;
            for (int c = 0; c < 3; ++c) {
                const float v = (float) x[(yy * W + xx) * 3 + c] * a + b;
                s[c] += v;
                s[3 + c] += v * v;
            }
        }
        for (int c = 0; c < 6; ++c) {
            const float t = sycl::reduce_over_group(it.get_group(), s[c], sycl::plus<float>());
            if (it.get_local_id(1) == 0) scratch[(pi * NG + gi) * 6 + c] = t;
        }
    });
    const float n = (float) (w * h);
    g->q.single_task([=]() {
        for (int pi = 0; pi < 2; ++pi) {
            for (int c = 0; c < 3; ++c) {
                float s = 0.0f, s2 = 0.0f;
                for (int gi = 0; gi < NG; ++gi) {
                    s += scratch[(pi * NG + gi) * 6 + c];
                    s2 += scratch[(pi * NG + gi) * 6 + 3 + c];
                }
                const float m = s / n, var = sycl::fmax(s2 / n - m * m, 0.0f);
                ms[c * 4 + 2 * pi] = m;
                ms[c * 4 + 2 * pi + 1] = sycl::sqrt(var + 1e-5f);
            }
        }
    });
    return 0;
    NS_CATCH
}

int ns_image_flashvsr_to_u8(ns_gpu* g, const void* dec_, int64_t W, int64_t x0, int64_t y0, int64_t w, int64_t h, const float* ms, uint8_t* out) {
    NS_TRY
    const half* dec = (const half*) dec_;
    return each(g, w * h * 3, [=](int64_t id) {
        const int64_t c = id % 3, p = id / 3;
        const int64_t xx = x0 + p % w, yy = y0 + p / w;
        float v = (float) dec[(yy * W + xx) * 3 + c] * 2.0f - 1.0f;
        if (ms) v = (v - ms[c * 4]) / ms[c * 4 + 1] * ms[c * 4 + 3] + ms[c * 4 + 2];
        v = sycl::fmin(sycl::fmax(v, -1.0f), 1.0f);
        out[id] = (uint8_t) sycl::fmin(sycl::fmax((v + 1.0f) * 127.5f, 0.0f), 255.0f);
    });
    NS_CATCH
}

int ns_image_flashvsr_randn(ns_gpu* g, float* out, int64_t n, uint64_t seed, uint64_t offset) {
    NS_TRY
    return each(g, n, [=](int64_t id) {
        const uint64_t r = splitmix(seed * 0x2545f4914f6cdd1dull ^ splitmix((uint64_t) id + offset));
        // 24-bit uniforms in float (no double precision on the card): u1 in (0, 1], u2 in [0, 1)
        const float u1 = ((float) (r >> 40) + 1.0f) / 16777216.0f;
        const float u2 = (float) (splitmix(r) >> 40) / 16777216.0f;
        out[id] = sycl::sqrt(-2.0f * sycl::log(u1)) * sycl::cos(6.2831853071795865f * u2);
    });
    NS_CATCH
}

}  // extern "C"
