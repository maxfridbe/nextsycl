// sp.cpp: the Sophon-Realtime engine's own kernels (sp.h): the brain's quantized matrices made on the card from the
// checkpoint's FP8 rows, their decode product and their expansion to half for the prompt's product (oneDNN, nsd.h),
// attention against the half cache, and the audio encoder's convolution patches. Sub-groups of 16 throughout.
#include "ns.h"
#include "ns_internal.hpp"
#include "sp.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstring>

namespace {

constexpr int SG = 16;

// FP8 E4M3 (fn: no infinities; S.1111.111 is NaN, read as 0): its bits shifted into a half's exponent and mantissa
// give the value / 256 (normals and subnormals alike)
inline float fp8_e4m3(uint8_t b) {
    if ((b & 0x7f) == 0x7f) return 0.0f;
    const uint16_t h = (uint16_t) (((b & 0x80) << 8) | ((b & 0x7f) << 7));
    return (float) sycl::bit_cast<sycl::half>(h) * 256.0f;
}

inline float bf16(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t) b << 16);
}

// a group of 32 into the q4 layout: candidate ranges (the full one shrunk toward its middle), each refitted by least
// squares on the codes it gives, rounded to half; the one with the smallest squared error kept
inline void quant_q4(const float* x, uint8_t* q, sycl::half* sm) {
    float lo = x[0], hi = x[0];
    for (int j = 1; j < 32; ++j) {
        lo = sycl::fmin(lo, x[j]);
        hi = sycl::fmax(hi, x[j]);
    }
    uint8_t best[32];
    float best_err = INFINITY;
    sycl::half bs = 0.0f, bm = (sycl::half) lo;
    for (int j = 0; j < 32; ++j) best[j] = 0;
    if (hi - lo > 1e-30f) {
        for (int t = 0; t < 12; ++t) {
            const float f = 1.0f - 0.02f * (float) t;
            const float s0 = (hi - lo) * f / 15.0f, m0 = lo + (hi - lo) * (1.0f - f) * 0.5f;
            uint8_t c[32];
            float sq = 0, sqq = 0, sx = 0, sqx = 0;
            for (int j = 0; j < 32; ++j) {
                const float v = sycl::rint((x[j] - m0) / s0);
                const int k = (int) sycl::fmin(15.0f, sycl::fmax(0.0f, v));
                c[j] = (uint8_t) k;
                sq += (float) k;
                sqq += (float) (k * k);
                sx += x[j];
                sqx += (float) k * x[j];
            }
            const float det = 32.0f * sqq - sq * sq;
            float s = s0, m = m0;
            if (det > 0.0f) {
                s = (32.0f * sqx - sq * sx) / det;
                m = (sx - s * sq) / 32.0f;
            }
            if (!(s > 0.0f)) {
                s = s0;
                m = m0;
            }
            const sycl::half sh = (sycl::half) s, mh = (sycl::half) m;
            const float sf = (float) sh, mf = (float) mh;
            // the codes again for the rounded pair, and their error
            float err = 0.0f;
            for (int j = 0; j < 32; ++j) {
                const float v = sf > 0.0f ? sycl::rint((x[j] - mf) / sf) : 0.0f;
                const int k = (int) sycl::fmin(15.0f, sycl::fmax(0.0f, v));
                c[j] = (uint8_t) k;
                const float d = x[j] - ((float) k * sf + mf);
                err += d * d;
            }
            if (err < best_err) {
                best_err = err;
                bs = sh;
                bm = mh;
                for (int j = 0; j < 32; ++j) best[j] = c[j];
            }
        }
    }
    // one 16-byte store of the codes, one 4-byte store of the pair
    sycl::uint4 w{0u, 0u, 0u, 0u};
    for (int j = 0; j < 16; ++j) w[j / 4] |= (uint32_t) (best[j] | (best[16 + j] << 4)) << (8 * (j % 4));
    *reinterpret_cast<sycl::uint4*>(q) = w;
    *reinterpret_cast<sycl::half2*>(sm) = sycl::half2(bs, bm);
}

inline void quant_q8(const float* x, int8_t* q, sycl::half* s) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(x[j]));
    const sycl::half sh = (sycl::half) (amax / 127.0f);
    const float sf = (float) sh;
    sycl::uint4 w[2] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
    for (int j = 0; j < 32; ++j) {
        const float v = sf > 0.0f ? sycl::rint(x[j] / sf) : 0.0f;
        const uint32_t b = (uint32_t) (uint8_t) (int8_t) sycl::fmin(127.0f, sycl::fmax(-127.0f, v));
        w[j / 16][(j % 16) / 4] |= b << (8 * (j % 4));
    }
    reinterpret_cast<sycl::uint4*>(q)[0] = w[0];
    reinterpret_cast<sycl::uint4*>(q)[1] = w[1];
    *s = sh;
}

template <int M, int WT>
void gemv_k(sycl::queue& q, const uint8_t* qw, const sycl::half* s, const float* x, int64_t K, int64_t N, float* out, int64_t ldo, int acc) {
    const int64_t G = K / 32;
    constexpr int ROWS = 8;
    const int64_t wgs = (N + ROWS - 1) / ROWS;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (wgs * ROWS * SG)), sycl::range<1>(ROWS * SG)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int64_t row = it.get_group(0) * ROWS + sg.get_group_linear_id();
        if (row >= N) return;
        const int lane = (int) sg.get_local_linear_id();
        float a[M];
        for (int m = 0; m < M; ++m) a[m] = 0.0f;
        for (int64_t gi = lane; gi < G; gi += SG) {
            const int64_t gidx = row * G + gi;
            if constexpr (WT == 1) {
                const sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(qw + gidx * 16);
                const sycl::half2 sm = reinterpret_cast<const sycl::half2*>(s)[gidx];
                const float sc = (float) sm[0], off = (float) sm[1];
                float lo[16], hi[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint32_t b = (v[j / 4] >> (8 * (j % 4))) & 0xffu;
                    lo[j] = (float) (b & 15u);
                    hi[j] = (float) (b >> 4);
                }
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const sycl::float4* xp = reinterpret_cast<const sycl::float4*>(x + m * K + gi * 32);
                    float d = 0.0f, sx = 0.0f;
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const sycl::float4 x0 = xp[c], x1 = xp[4 + c];
                        d += x0[0] * lo[c * 4] + x0[1] * lo[c * 4 + 1] + x0[2] * lo[c * 4 + 2] + x0[3] * lo[c * 4 + 3];
                        d += x1[0] * hi[c * 4] + x1[1] * hi[c * 4 + 1] + x1[2] * hi[c * 4 + 2] + x1[3] * hi[c * 4 + 3];
                        sx += x0[0] + x0[1] + x0[2] + x0[3] + x1[0] + x1[1] + x1[2] + x1[3];
                    }
                    a[m] += d * sc + sx * off;
                }
            } else {
                const sycl::uint4 v0 = *reinterpret_cast<const sycl::uint4*>(qw + gidx * 32);
                const sycl::uint4 v1 = *reinterpret_cast<const sycl::uint4*>(qw + gidx * 32 + 16);
                const float sc = (float) s[gidx];
                float w[32];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    w[j] = (float) (int8_t) ((v0[j / 4] >> (8 * (j % 4))) & 0xffu);
                    w[16 + j] = (float) (int8_t) ((v1[j / 4] >> (8 * (j % 4))) & 0xffu);
                }
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const sycl::float4* xp = reinterpret_cast<const sycl::float4*>(x + m * K + gi * 32);
                    float d = 0.0f;
#pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        const sycl::float4 xv = xp[c];
                        d += xv[0] * w[c * 4] + xv[1] * w[c * 4 + 1] + xv[2] * w[c * 4 + 2] + xv[3] * w[c * 4 + 3];
                    }
                    a[m] += d * sc;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float r = sycl::reduce_over_group(sg, a[m], sycl::plus<float>());
            if (lane == 0) {
                float* o = out + m * ldo + row;
                *o = acc ? *o + r : r;
            }
        }
    });
}

int64_t attn_chunks(int64_t S, int64_t Hkv, int64_t T) {
    int64_t n = 256 / (S * Hkv > 0 ? S * Hkv : 1);
    if (n < 1) n = 1;
    const int64_t most = (T + 31) / 32;
    return n < most ? n : most;
}

}  // namespace

extern "C" {

int ns_audio_sophon_quant(ns_gpu* g, int wt, const void* src, int st, const float* scale, int64_t N, int64_t K, void* q, void* s) {
    NS_TRY
    if (K % 32 != 0) return ns_fail("ns_audio_sophon_quant: K must be a multiple of 32");
    if (wt != 0 && wt != 1) return ns_fail("ns_audio_sophon_quant: layout 0 (q8) or 1 (q4)");
    if (st == 0 && !scale) return ns_fail("ns_audio_sophon_quant: FP8 rows need their scales");
    const int64_t G = K / 32, n = N * G;
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t gi = id[0], row = gi / G;
        float x[32];
        // the group's source in 16-byte loads
        if (st == 0) {
            const sycl::uint4* p = reinterpret_cast<const sycl::uint4*>(static_cast<const uint8_t*>(src) + gi * 32);
            const float sc = scale[row];
            for (int c = 0; c < 2; ++c) {
                const sycl::uint4 w = p[c];
                for (int j = 0; j < 16; ++j) x[c * 16 + j] = fp8_e4m3((uint8_t) ((w[j / 4] >> (8 * (j % 4))) & 0xffu)) * sc;
            }
        } else {
            const sycl::uint4* p = reinterpret_cast<const sycl::uint4*>(static_cast<const uint16_t*>(src) + gi * 32);
            for (int c = 0; c < 4; ++c) {
                const sycl::uint4 w = p[c];
                for (int j = 0; j < 8; ++j) x[c * 8 + j] = bf16((uint16_t) ((w[j / 2] >> (16 * (j % 2))) & 0xffffu));
            }
        }
        if (wt == 1) quant_q4(x, static_cast<uint8_t*>(q) + gi * 16, static_cast<sycl::half*>(s) + gi * 2);
        else quant_q8(x, static_cast<int8_t*>(q) + gi * 32, static_cast<sycl::half*>(s) + gi);
    });
    return 0;
    NS_CATCH
}

int ns_audio_sophon_gemv(ns_gpu* g, int wt, const void* q, const void* s, const float* x, int64_t M, int64_t K, int64_t N, float* out,
                         int64_t ldo, int acc) {
    NS_TRY
    if (K % 32 != 0 || M < 1 || M > 4) return ns_fail("ns_audio_sophon_gemv: K a multiple of 32, 1 <= M <= 4");
    if (N <= 0) return 0;
    auto qw = static_cast<const uint8_t*>(q);
    auto sh = static_cast<const sycl::half*>(s);
#define SP_GEMV(MM)                                                         \
    if (M == MM) {                                                          \
        if (wt == 1) gemv_k<MM, 1>(g->q, qw, sh, x, K, N, out, ldo, acc);  \
        else gemv_k<MM, 0>(g->q, qw, sh, x, K, N, out, ldo, acc);          \
        return 0;                                                           \
    }
    SP_GEMV(1)
    SP_GEMV(2)
    SP_GEMV(3)
    SP_GEMV(4)
#undef SP_GEMV
    return 0;
    NS_CATCH
}

int ns_audio_sophon_dequant(ns_gpu* g, int wt, const void* q, const void* s, int64_t K, int64_t r0, int64_t rows, void* out) {
    NS_TRY
    if (K % 32 != 0) return ns_fail("ns_audio_sophon_dequant: K must be a multiple of 32");
    const int64_t G = K / 32, n = rows * G;
    if (n <= 0) return 0;
    // a group a work-item: its codes in 16-byte loads, its 32 halves out in four 16-byte stores (byte and half-size
    // accesses are slow on Xe2)
    using H8 = sycl::vec<sycl::half, 8>;
    auto o = static_cast<H8*>(out);
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t local = id[0], gi = r0 * G + local;
        H8 v[4];
        if (wt == 1) {
            const sycl::uint4 c = *reinterpret_cast<const sycl::uint4*>(static_cast<const uint8_t*>(q) + gi * 16);
            const sycl::half2 sm = static_cast<const sycl::half2*>(s)[gi];
            const float sc = (float) sm[0], off = (float) sm[1];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint32_t b = (c[j / 4] >> (8 * (j % 4))) & 0xffu;
                v[j / 8][j % 8] = (sycl::half) ((float) (b & 15u) * sc + off);
                v[2 + j / 8][j % 8] = (sycl::half) ((float) (b >> 4) * sc + off);
            }
        } else {
            const uint8_t* p = static_cast<const uint8_t*>(q) + gi * 32;
            const sycl::uint4 c0 = *reinterpret_cast<const sycl::uint4*>(p), c1 = *reinterpret_cast<const sycl::uint4*>(p + 16);
            const float sc = (float) static_cast<const sycl::half*>(s)[gi];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                v[j / 8][j % 8] = (sycl::half) ((float) (int8_t) ((c0[j / 4] >> (8 * (j % 4))) & 0xffu) * sc);
                v[2 + j / 8][j % 8] = (sycl::half) ((float) (int8_t) ((c1[j / 4] >> (8 * (j % 4))) & 0xffu) * sc);
            }
        }
        H8* d = o + local * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) d[k] = v[k];
    });
    return 0;
    NS_CATCH
}

int64_t ns_audio_sophon_attn_scratch(int64_t S, int64_t Hq, int64_t Hkv, int64_t D, int64_t p0) {
    const int64_t nch = attn_chunks(S, Hkv, p0 + S);
    return nch > 1 ? S * Hq * nch * (D + 2) : 0;
}

int ns_audio_sophon_attn(ns_gpu* g, const float* q, int64_t qs, const void* kc, const void* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t D,
                         int64_t p0, float* out, float* part) {
    NS_TRY
    if (D != 128 || Hkv <= 0 || Hq % Hkv != 0 || Hq / Hkv > 16) return ns_fail("ns_audio_sophon_attn: D 128, Hq a multiple of Hkv, at most 16 a key head");
    if (S <= 0) return 0;
    const int64_t G = Hq / Hkv, T = p0 + S, nch = attn_chunks(S, Hkv, T), C = (T + nch - 1) / nch, HD = Hkv * D;
    if (nch > 1 && !part) return ns_fail("ns_audio_sophon_attn: the scratch is needed");
    const float scale = 1.0f / sycl::sqrt((float) D);
    auto K = static_cast<const sycl::half*>(kc);
    auto V = static_cast<const sycl::half*>(vc);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * Hkv * nch * G * SG)), sycl::range<1>((size_t) (G * SG))),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int64_t w = it.get_group(0), c = w % nch, kvh = (w / nch) % Hkv, s = w / (nch * Hkv);
        const int64_t h = kvh * G + sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t i = p0 + s, j0 = c * C;
        int64_t j1 = (c + 1) * C;
        if (j1 > i + 1) j1 = i + 1;
        float qv[8], acc[8];
        for (int e = 0; e < 8; ++e) {
            qv[e] = q[s * qs + h * D + lane * 8 + e] * scale;
            acc[e] = 0.0f;
        }
        float m = -INFINITY, l = 0.0f;
        for (int64_t j = j0; j < j1; ++j) {
            const sycl::vec<sycl::half, 8> kv = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(K + j * HD + kvh * D + lane * 8);
            float d = 0.0f;
            for (int e = 0; e < 8; ++e) d += qv[e] * (float) kv[e];
            const float sc = sycl::reduce_over_group(sg, d, sycl::plus<float>());
            const float mn = sycl::fmax(m, sc), a = sycl::exp(m - mn), p = sycl::exp(sc - mn);
            const sycl::vec<sycl::half, 8> vv = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(V + j * HD + kvh * D + lane * 8);
            for (int e = 0; e < 8; ++e) acc[e] = acc[e] * a + p * (float) vv[e];
            l = l * a + p;
            m = mn;
        }
        if (nch == 1) {
            for (int e = 0; e < 8; ++e) out[s * Hq * D + h * D + lane * 8 + e] = acc[e] / l;
        } else {
            float* pp = part + ((s * Hq + h) * nch + c) * (D + 2);
            if (lane == 0) {
                pp[0] = m;
                pp[1] = l;
            }
            for (int e = 0; e < 8; ++e) pp[2 + lane * 8 + e] = acc[e];
        }
    });
    if (nch > 1) {
        g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (S * Hq * D)), sycl::range<1>((size_t) D)), [=](sycl::nd_item<1> it) {
            const int64_t sh = it.get_group(0), d = it.get_local_id(0);
            const float* pp = part + sh * nch * (D + 2);
            float mx = -INFINITY;
            for (int64_t c = 0; c < nch; ++c) mx = sycl::fmax(mx, pp[c * (D + 2)]);
            float L = 0.0f, O = 0.0f;
            for (int64_t c = 0; c < nch; ++c) {
                const float mc = pp[c * (D + 2)];
                if (mc == -INFINITY) continue;
                const float e = sycl::exp(mc - mx);
                L += e * pp[c * (D + 2) + 1];
                O += e * pp[c * (D + 2) + 2 + d];
            }
            out[sh * D + d] = O / L;
        });
    }
    return 0;
    NS_CATCH
}

int ns_audio_sophon_kv_store(ns_gpu* g, const float* k, const float* v, int64_t ks, int64_t S, int64_t HD, int64_t p0, void* kc, void* vc) {
    NS_TRY
    if (S <= 0) return 0;
    auto K = static_cast<sycl::half*>(kc);
    auto V = static_cast<sycl::half*>(vc);
    g->q.parallel_for(sycl::range<1>((size_t) (S * HD)), [=](sycl::id<1> id) {
        const int64_t s = id[0] / HD, i = id[0] % HD;
        K[(p0 + s) * HD + i] = (sycl::half) k[s * ks + i];
        V[(p0 + s) * HD + i] = (sycl::half) v[s * ks + i];
    });
    return 0;
    NS_CATCH
}

int ns_audio_sophon_im2col_s2(ns_gpu* g, const float* x, int64_t B, int64_t H, int64_t W, int64_t C, float* out) {
    NS_TRY
    const int64_t Ho = (H + 1) / 2, Wo = (W + 1) / 2, cols = C * 9, n = B * Ho * Wo * cols;
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const int64_t col = id[0] % cols, r = id[0] / cols;
        const int64_t wo = r % Wo, ho = (r / Wo) % Ho, b = r / (Wo * Ho);
        const int64_t c = col / 9, kh = (col / 3) % 3, kw = col % 3;
        const int64_t hi = ho * 2 - 1 + kh, wi = wo * 2 - 1 + kw;
        out[id[0]] = (hi >= 0 && hi < H && wi >= 0 && wi < W) ? x[((b * H + hi) * W + wi) * C + c] : 0.0f;
    });
    return 0;
    NS_CATCH
}

int ns_audio_sophon_swap12(ns_gpu* g, const float* x, int64_t A, int64_t B, int64_t C, int64_t D, float* out) {
    NS_TRY
    const int64_t n = A * B * C * D;
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        // out index (a, c, b, d)
        const int64_t d = id[0] % D, b = (id[0] / D) % B, c = (id[0] / (D * B)) % C, a = id[0] / (D * B * C);
        out[id[0]] = x[((a * B + b) * C + c) * D + d];
    });
    return 0;
    NS_CATCH
}

int ns_audio_sophon_add_rows(ns_gpu* g, float* x, const float* y, int64_t B, int64_t R, int64_t C) {
    NS_TRY
    const int64_t n = B * R * C;
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { x[id[0]] += y[id[0] % (R * C)]; });
    return 0;
    NS_CATCH
}

int ns_audio_sophon_gather(ns_gpu* g, const float* x, const int32_t* idx, int64_t n, int64_t C, float* out) {
    NS_TRY
    if (n <= 0) return 0;
    g->q.parallel_for(sycl::range<1>((size_t) (n * C)), [=](sycl::id<1> id) {
        const int64_t r = id[0] / C, c = id[0] % C;
        out[id[0]] = x[(int64_t) idx[r] * C + c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_sophon_attn_win(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t H, int64_t D, int64_t stride, int64_t W,
                             float* out) {
    NS_TRY
    if (D > 64 || W > 128 || W <= 0) return ns_fail("ns_audio_sophon_attn_win: D <= 64, 0 < W <= 128");
    if (n <= 0 || H <= 0) return 0;
    const float scale = 1.0f / sycl::sqrt((float) D);
    // one sub-group a (row, head): a lane a key of the window (W <= 128: up to 8 keys a lane), then the values summed
    // by the lanes over their keys, a lane per 4 features
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (n * H * SG)), sycl::range<1>(SG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int64_t w = it.get_group(0), i = w / H, h = w % H;
        const int lane = (int) sg.get_local_linear_id();
        const int64_t w0 = (i / W) * W, w1 = sycl::min(w0 + W, n), m = w1 - w0;
        const float* qi = q + i * stride + h * D;
        float s[8];
        float mx = -INFINITY;
        for (int t = 0; t < 8; ++t) {
            const int64_t j = lane + t * SG;
            s[t] = -INFINITY;
            if (j < m) {
                const float* kj = k + (w0 + j) * stride + h * D;
                float d = 0.0f;
                for (int64_t e = 0; e < D; ++e) d += qi[e] * kj[e];
                s[t] = d * scale;
                mx = sycl::fmax(mx, s[t]);
            }
        }
        mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
        float sum = 0.0f;
        for (int t = 0; t < 8; ++t) {
            s[t] = (lane + t * SG < m) ? sycl::exp(s[t] - mx) : 0.0f;
            sum += s[t];
        }
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        // each lane's share of the values, then summed across the lanes a feature at a time
        float* o = out + i * H * D + h * D;
        for (int64_t e = 0; e < D; ++e) {
            float a = 0.0f;
            for (int t = 0; t < 8; ++t) {
                const int64_t j = lane + t * SG;
                if (j < m) a += s[t] * v[(w0 + j) * stride + h * D + e];
            }
            a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
            if (lane == 0) o[e] = a / sum;
        }
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
