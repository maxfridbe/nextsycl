// vvlm.cpp: the VibeVoice engine's products and its language model's decode step (vv.h).
//
// The products are a copy of the MiniMax Music engine's small-batch product (mm3.cpp's gemv: CONTRIBUTING rule 6),
// changed for this model's shapes: three weight forms (half; bfloat16, the checkpoint's own values; and Q8_0 kept as
// the file has it - int8 codes and a half scale a block of 32 - so the language model reads half the bytes), up to 20
// rows, the reduction over K split between the sub-groups of a work-group when the matrix has few output rows (each
// sub-group then has a long way to go alone), and what follows the product done on the way out (a bias, accumulate,
// SwiGLU of interleaved gate / up rows, exact GELU, a layer-scaled residual).
//
// The decode step's own kernels take their positions from device memory, so a step's work does not depend on host
// values beyond the buffers' addresses: RoPE and the key / value store in one launch for the step's rows (each row its
// own sequence: the guidance's two branches), and the cached attention split over the keys in a fixed number of
// pieces (the GPU kept busy at any length), merged by a second kernel.
#include "ns.h"
#include "ns_internal.hpp"
#include "vv.h"

#include <sycl/sycl.hpp>
#include <cmath>

namespace {
using half = sycl::half;
constexpr int SG = 16;
constexpr int NSG = 8;

inline float bf16f(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t) b << 16);
}

inline float silu(float g) {
    return g / (1.0f + sycl::exp(-g));
}

inline float gelu(float v) {
    return 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f));
}

// what follows the product for output n of row m (v: the sum, its bias added)
inline void epi_one(int epi, float* out, int64_t ldo, int m, int64_t n, float v, const float* gamma, int64_t ldg) {
    float* o = out + m * ldo + n;
    switch (epi) {
    case NS_VV_STORE: *o = v; break;
    case NS_VV_ACC: *o += v; break;
    case NS_VV_GELU: *o = gelu(v); break;
    case NS_VV_RES: *o += gamma[m * ldg + n] * v; break;
    default: break;
    }
}

// WK: 0 half, 1 bfloat16, 2 Q8_0 (codes [N, K] int8, scales [N, K / 32] half), 3 float32. Each sub-group R output rows, its lanes
// V values at a time across K; KS sub-groups share a row group's K (interleaved), their sums met in local memory.
template <int WK, int MM, int KS>
void gemv(sycl::queue& q, const float* x, int M, int64_t ldx, int64_t K, const void* w, const half* dq, int64_t N, const float* bias, int epi,
          const float* gamma, int64_t ldg, float* out, int64_t ldo) {
    constexpr int R = 2, V = WK == 2 ? 16 : 8, RG = NSG / KS, ROWS = RG * R;
    const int64_t groups = (N + ROWS - 1) / ROWS;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) (KS > 1 ? NSG * MM * R : 1)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * SG * NSG), sycl::range<1>(SG * NSG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int ks = sgi % KS, rg = sgi / KS;
            const int64_t n0 = ((int64_t) it.get_group(0) * RG + rg) * R;
            float a[MM][R];
#pragma unroll
            for (int m = 0; m < MM; ++m)
#pragma unroll
                for (int r = 0; r < R; ++r) a[m][r] = 0.0f;
            for (int64_t k = ((int64_t) ks * SG + lane) * V; k < K; k += (int64_t) SG * V * KS) {
                float wf[R][V];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int64_t n = n0 + r < N ? n0 + r : N - 1;
                    if constexpr (WK == 0) {
                        const sycl::vec<half, V> wv = *reinterpret_cast<const sycl::vec<half, V>*>((const half*) w + n * K + k);
#pragma unroll
                        for (int i = 0; i < V; ++i) wf[r][i] = (float) wv[i];
                    } else if constexpr (WK == 1) {
                        const sycl::vec<uint16_t, V> wv = *reinterpret_cast<const sycl::vec<uint16_t, V>*>((const uint16_t*) w + n * K + k);
#pragma unroll
                        for (int i = 0; i < V; ++i) wf[r][i] = bf16f(wv[i]);
                    } else if constexpr (WK == 3) {
#pragma unroll
                        for (int i = 0; i < V; i += 4) {
                            const sycl::vec<float, 4> w4 = *reinterpret_cast<const sycl::vec<float, 4>*>((const float*) w + n * K + k + i);
                            wf[r][i] = w4[0];
                            wf[r][i + 1] = w4[1];
                            wf[r][i + 2] = w4[2];
                            wf[r][i + 3] = w4[3];
                        }
                    } else {
                        const sycl::vec<int8_t, V> wv = *reinterpret_cast<const sycl::vec<int8_t, V>*>((const int8_t*) w + n * K + k);
#pragma unroll
                        for (int i = 0; i < V; ++i) wf[r][i] = (float) wv[i];
                    }
                }
                float d[R];
                if constexpr (WK == 2) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const int64_t n = n0 + r < N ? n0 + r : N - 1;
                        d[r] = (float) dq[n * (K / 32) + k / 32];
                    }
                }
#pragma unroll
                for (int m = 0; m < MM; ++m) {
                    if (m < M) {
                        float xv[V];
#pragma unroll
                        for (int i = 0; i < V; i += 4) {
                            const sycl::vec<float, 4> x4 = *reinterpret_cast<const sycl::vec<float, 4>*>(x + m * ldx + k + i);
                            xv[i] = x4[0];
                            xv[i + 1] = x4[1];
                            xv[i + 2] = x4[2];
                            xv[i + 3] = x4[3];
                        }
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            float s = 0.0f;
#pragma unroll
                            for (int i = 0; i < V; ++i) s += wf[r][i] * xv[i];
                            if constexpr (WK == 2) a[m][r] += d[r] * s;
                            else a[m][r] += s;
                        }
                    }
                }
            }
            // the sums: row group rg's R outputs of each row m
#pragma unroll
            for (int m = 0; m < MM; ++m)
#pragma unroll
                for (int r = 0; r < R; ++r) a[m][r] = sycl::reduce_over_group(sg, a[m][r], sycl::plus<float>());
            auto finish = [&](int m, int64_t nb, const float* v) {
                float b[R];
#pragma unroll
                for (int r = 0; r < R; ++r) b[r] = v[r] + (bias && nb + r < N ? bias[nb + r] : 0.0f);
                if (epi == NS_VV_SWIGLU) {
                    // rows (2i, 2i + 1) are gate i and up i
#pragma unroll
                    for (int r = 0; r + 1 < R; r += 2)
                        if (nb + r + 1 < N) out[m * ldo + (nb + r) / 2] = silu(b[r]) * b[r + 1];
                } else {
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (nb + r < N) epi_one(epi, out, ldo, m, nb + r, b[r], gamma, ldg);
                }
            };
            if constexpr (KS == 1) {
                if (lane == 0) {
#pragma unroll
                    for (int m = 0; m < MM; ++m)
                        if (m < M) finish(m, n0, a[m]);
                }
            } else {
                if (lane == 0) {
#pragma unroll
                    for (int m = 0; m < MM; ++m)
#pragma unroll
                        for (int r = 0; r < R; ++r) slm[((rg * KS + ks) * MM + m) * R + r] = a[m][r];
                }
                sycl::group_barrier(it.get_group());
                // one work-item a (row group, row m): its KS partial sums in order
                for (int idx = (int) it.get_local_id(0); idx < RG * MM; idx += SG * NSG) {
                    const int g = idx / MM, m = idx % MM;
                    if (m >= M) continue;
                    float v[R];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        float s = 0.0f;
                        for (int j = 0; j < KS; ++j) s += slm[((g * KS + j) * MM + m) * R + r];
                        v[r] = s;
                    }
                    finish(m, ((int64_t) it.get_group(0) * RG + g) * R, v);
                }
            }
        });
    });
}

template <int WK, int MM>
void gemv_ks(sycl::queue& q, int ks, const float* x, int M, int64_t ldx, int64_t K, const void* w, const half* dq, int64_t N, const float* bias,
             int epi, const float* gamma, int64_t ldg, float* out, int64_t ldo) {
    if (ks >= 4) gemv<WK, MM, 4>(q, x, M, ldx, K, w, dq, N, bias, epi, gamma, ldg, out, ldo);
    else if (ks == 2) gemv<WK, MM, 2>(q, x, M, ldx, K, w, dq, N, bias, epi, gamma, ldg, out, ldo);
    else gemv<WK, MM, 1>(q, x, M, ldx, K, w, dq, N, bias, epi, gamma, ldg, out, ldo);
}

template <int WK>
void gemv_m(sycl::queue& q, int ks, const float* x, int M, int64_t ldx, int64_t K, const void* w, const half* dq, int64_t N, const float* bias,
            int epi, const float* gamma, int64_t ldg, float* out, int64_t ldo) {
    if (M == 1) gemv_ks<WK, 1>(q, ks, x, M, ldx, K, w, dq, N, bias, epi, gamma, ldg, out, ldo);
    else if (M == 2) gemv_ks<WK, 2>(q, ks, x, M, ldx, K, w, dq, N, bias, epi, gamma, ldg, out, ldo);
    else if (M <= 4) gemv_ks<WK, 4>(q, ks, x, M, ldx, K, w, dq, N, bias, epi, gamma, ldg, out, ldo);
    else if (M <= 8) gemv_ks<WK, 8>(q, ks, x, M, ldx, K, w, dq, N, bias, epi, gamma, ldg, out, ldo);
    else gemv_ks<WK, 20>(q, ks, x, M, ldx, K, w, dq, N, bias, epi, gamma, ldg, out, ldo);
}

// the split of K a shape gets (NS_VV_GEMV_KS overrides: a tuning switch): a matrix of few rows splits its long rows so
// that every row group's sub-groups share them (the GPU has room for ~2,000 sub-groups at once)
int pick_ks(int64_t N, int64_t K) {
    static const int force = [] { const char* v = std::getenv("NS_VV_GEMV_KS"); return v ? std::atoi(v) : 0; }();
    if (force > 0) return force;
    const int64_t sgs = (N + 1) / 2;
    if (sgs >= 4096 || K < 1024) return 1;
    if (sgs >= 1024) return 2;
    return 4;
}
}  // namespace

extern "C" {

int ns_audio_vv_gemv(ns_gpu* g, const float* x, int64_t M, int64_t ldx, int64_t K, const void* w, int wt, const void* scales, int64_t N,
                     const float* bias, int epi, const float* gamma, int64_t ldg, float* out, int64_t ldo) {
    NS_TRY
    if (M <= 0 || N <= 0) return 0;
    if (M > 20) return ns_fail("ns_audio_vv_gemv: at most 20 rows");
    if (K % (wt == 2 ? 32 : 8) != 0 || ldx % 4 != 0) return ns_fail("ns_audio_vv_gemv: K a multiple of 8 (Q8_0: 32), rows 16-byte aligned");
    if (epi == NS_VV_SWIGLU && N % 2 != 0) return ns_fail("ns_audio_vv_gemv: SwiGLU takes interleaved gate / up rows (N even)");
    if (epi == NS_VV_RES && !gamma) return ns_fail("ns_audio_vv_gemv: a residual's layer scale");
    const int ks = pick_ks(N, K);
    if (wt == 0) gemv_m<0>(g->q, ks, x, (int) M, ldx, K, w, nullptr, N, bias, epi, gamma, ldg, out, ldo);
    else if (wt == 1) gemv_m<1>(g->q, ks, x, (int) M, ldx, K, w, nullptr, N, bias, epi, gamma, ldg, out, ldo);
    else if (wt == 3) gemv_m<3>(g->q, ks, x, (int) M, ldx, K, w, nullptr, N, bias, epi, gamma, ldg, out, ldo);
    else if (wt == 2) {
        if (!scales) return ns_fail("ns_audio_vv_gemv: Q8_0 without its scales");
        gemv_m<2>(g->q, ks, x, (int) M, ldx, K, w, (const half*) scales, N, bias, epi, gamma, ldg, out, ldo);
    } else return ns_fail("ns_audio_vv_gemv: weights half (0), bfloat16 (1), Q8_0 (2) or float32 (3)");
    return 0;
    NS_CATCH
}

int ns_audio_vv_rms_norm(ns_gpu* g, const float* x, int64_t M, int64_t C, const float* w, float eps, const float* scale, const float* shift,
                         int64_t lds, float* out) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    constexpr int WG = 256;
    // a work-group a row: the sum of squares, then the row (nsd_rms_norm_mod's arithmetic, in one launch)
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) M * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int lid = (int) it.get_local_id(0);
        const float* xr = x + r * C;
        float s = 0.0f;
        for (int64_t i = lid; i < C; i += WG) s += xr[i] * xr[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = sycl::rsqrt(s / (float) C + eps);
        for (int64_t i = lid; i < C; i += WG) {
            float v = xr[i] * inv * (w ? w[i] : 1.0f);
            if (scale) v = v * (1.0f + scale[r * lds + i]) + shift[r * lds + i];
            out[r * C + i] = v;
        }
    });
    return 0;
    NS_CATCH
}

int ns_audio_vv_rope_kv(ns_gpu* g, float* qkv, int64_t ws, int64_t R, const int32_t* pos, int64_t Hq, int64_t Hkv, int64_t D,
                        const float* inv_freq, void* k0, void* v0, int64_t T0, void* k1, void* v1, int64_t T1) {
    NS_TRY
    if (R <= 0) return 0;
    if (R > 2 || D % 2 != 0) return ns_fail("ns_audio_vv_rope_kv: one or two rows, D even");
    const int64_t hd = D / 2, H = Hq + Hkv;
    // a work-item a (row, head, pair): the pair rotated; a key head's pair (and its values) stored in the row's cache
    g->q.parallel_for(sycl::range<1>((size_t) (R * H * hd)), [=](sycl::id<1> id) {
        const int64_t j = id[0] % hd, hh = (id[0] / hd) % H, r = id[0] / (hd * H);
        const int64_t p = pos[r];
        float* v = qkv + r * ws + hh * D;
        const float a = (float) p * inv_freq[j], c = sycl::cos(a), s = sycl::sin(a);
        const float x1 = v[j], x2 = v[j + hd];
        const float y1 = x1 * c - x2 * s, y2 = x2 * c + x1 * s;
        if (hh < Hq) {
            v[j] = y1;
            v[j + hd] = y2;
        } else {
            const int64_t kh = hh - Hq, T = r == 0 ? T0 : T1;
            half* kc = (half*) (r == 0 ? k0 : k1);
            half* vc = (half*) (r == 0 ? v0 : v1);
            const float* vv = qkv + r * ws + (Hq + Hkv) * D + kh * D;
            const int64_t at = (kh * T + p) * D;
            kc[at + j] = (half) y1;
            kc[at + j + hd] = (half) y2;
            vc[at + j] = (half) vv[j];
            vc[at + j + hd] = (half) vv[j + hd];
        }
    });
    return 0;
    NS_CATCH
}

int ns_audio_vv_attn_dec(ns_gpu* g, const float* qp, int64_t qs, int64_t R, const int32_t* pos, const void* k0, const void* v0, int64_t T0,
                         const void* k1, const void* v1, int64_t T1, int64_t Hq, int64_t Hkv, int64_t D, int64_t ns, float* out, float* part) {
    NS_TRY
    if (R <= 0) return 0;
    if (R > 2 || D != 128 || Hq != 6 * Hkv || ns < 1 || ns > 256) return ns_fail("ns_audio_vv_attn_dec: one or two rows, D 128, 6 query heads a key head, 1-256 pieces");
    constexpr int DL = 8, G = 6, NA = 4, WG = SG * NA;
    const float scale = 1.0f / sycl::sqrt(128.0f);
    // a work-group a (row, key head, piece of the keys): its sub-groups take the piece's keys in turn, each lane DL
    // features of the G query heads that read the key head; their running (max, sum, weighted values) merged in local
    // memory into the piece's partial
    g->q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>((size_t) NA * G * (128 + 2)), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (R * Hkv * ns) * WG), sycl::range<1>(WG)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            constexpr int Dc = 128;
            auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            int64_t gi = it.get_group(0);
            const int64_t sp = gi % ns;
            gi /= ns;
            const int64_t hk = gi % Hkv, r = gi / Hkv;
            const int64_t len = (int64_t) pos[r] + 1, chunk = (len + ns - 1) / ns;
            const int64_t j0 = sp * chunk, j1 = sycl::min(len, j0 + chunk);
            const int64_t T = r == 0 ? T0 : T1;
            const half* kb = (const half*) (r == 0 ? k0 : k1) + hk * T * Dc + lane * DL;
            const half* vb = (const half*) (r == 0 ? v0 : v1) + hk * T * Dc + lane * DL;
            float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
            for (int gg = 0; gg < G; ++gg) {
                const float* qr = qp + r * qs + (hk * G + gg) * Dc + lane * DL;
#pragma unroll
                for (int i = 0; i < DL; ++i) {
                    qv[gg][i] = qr[i] * scale;
                    acc[gg][i] = 0.0f;
                }
                m[gg] = -INFINITY;
                l[gg] = 0.0f;
            }
            for (int64_t j = j0 + sgi; j < j1; j += NA) {
                const sycl::vec<half, DL> kv = *reinterpret_cast<const sycl::vec<half, DL>*>(kb + j * Dc);
                const sycl::vec<half, DL> vv = *reinterpret_cast<const sycl::vec<half, DL>*>(vb + j * Dc);
#pragma unroll
                for (int gg = 0; gg < G; ++gg) {
                    float d = 0.0f;
#pragma unroll
                    for (int i = 0; i < DL; ++i) d += qv[gg][i] * (float) kv[i];
                    const float sc = sycl::reduce_over_group(sg, d, sycl::plus<float>());
                    const float mn = sycl::fmax(m[gg], sc);
                    const float c = sycl::exp(m[gg] - mn), e = sycl::exp(sc - mn);
#pragma unroll
                    for (int i = 0; i < DL; ++i) acc[gg][i] = acc[gg][i] * c + e * (float) vv[i];
                    l[gg] = l[gg] * c + e;
                    m[gg] = mn;
                }
            }
#pragma unroll
            for (int gg = 0; gg < G; ++gg) {
                float* o = &slm[(sgi * G + gg) * (Dc + 2)];
#pragma unroll
                for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[gg][i];
                if (lane == 0) {
                    o[Dc] = m[gg];
                    o[Dc + 1] = l[gg];
                }
            }
            sycl::group_barrier(it.get_group());
            for (int idx = (int) it.get_local_id(0); idx < G * Dc; idx += WG) {
                const int gg = idx / Dc, d = idx % Dc;
                float mx = -INFINITY;
                for (int k = 0; k < NA; ++k) mx = sycl::fmax(mx, slm[(k * G + gg) * (Dc + 2) + Dc]);
                float L = 0.0f, A = 0.0f;
                if (mx != -INFINITY) {
                    for (int k = 0; k < NA; ++k) {
                        const float* o = &slm[(k * G + gg) * (Dc + 2)];
                        const float e = sycl::exp(o[Dc] - mx);
                        L += o[Dc + 1] * e;
                        A += o[d] * e;
                    }
                }
                float* p = part + ((r * Hq + hk * G + gg) * ns + sp) * (Dc + 2);
                p[d] = A;
                if (d == 0) {
                    p[Dc] = mx;
                    p[Dc + 1] = L;
                }
            }
        });
    });
    // the pieces of each (row, query head) merged
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (R * Hq) * 128), sycl::range<1>(128)), [=](sycl::nd_item<1> it) {
        constexpr int Dc = 128;
        const int64_t rh = it.get_group(0), d = it.get_local_id(0);
        const float* p = part + rh * ns * (Dc + 2);
        float mx = -INFINITY;
        for (int64_t k = 0; k < ns; ++k) mx = sycl::fmax(mx, p[k * (Dc + 2) + Dc]);
        float L = 0.0f, A = 0.0f;
        if (mx != -INFINITY) {
            for (int64_t k = 0; k < ns; ++k) {
                const float e = sycl::exp(p[k * (Dc + 2) + Dc] - mx);
                L += p[k * (Dc + 2) + Dc + 1] * e;
                A += p[k * (Dc + 2) + d] * e;
            }
        }
        out[rh * Dc + d] = L > 0.0f ? A / L : 0.0f;
    });
    return 0;
    NS_CATCH
}

int ns_audio_vv_dequant_q8(ns_gpu* g, const void* q, const void* d, int64_t N, int64_t K, void* out) {
    NS_TRY
    if (N <= 0 || K <= 0) return 0;
    if (K % 32 != 0) return ns_fail("ns_audio_vv_dequant_q8: K a multiple of 32");
    const int8_t* qs = (const int8_t*) q;
    const half* ds = (const half*) d;
    half* o = (half*) out;
    g->q.parallel_for(sycl::range<1>((size_t) (N * K)), [=](sycl::id<1> id) {
        const int64_t i = id[0], n = i / K, k = i % K;
        o[i] = (half) ((float) ds[n * (K / 32) + k / 32] * (float) qs[i]);
    });
    return 0;
    NS_CATCH
}

int ns_audio_vv_swiglu_pairs(ns_gpu* g, const float* x, int64_t M, int64_t C, float* out) {
    NS_TRY
    if (M <= 0 || C <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (M * C)), [=](sycl::id<1> id) {
        const int64_t m = id[0] / C, i = id[0] % C;
        out[id[0]] = silu(x[m * 2 * C + 2 * i]) * x[m * 2 * C + 2 * i + 1];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
