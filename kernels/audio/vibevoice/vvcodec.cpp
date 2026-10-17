// vvcodec.cpp: the VibeVoice engine's σ-VAE tokenizer kernels (vv.h), streaming over time rows [L, C].
//
// A ConvNeXt block is two launches here and its MLP's two products: the first norm of the new rows written after the
// depthwise convolution's carried rows (copied from the previous call's buffer: the two alternate, so nothing is read
// and written in the same launch); then a work-group a row for the depthwise convolution, the layer-scaled residual
// and the second norm. Every convolution - strided, transposed, the stems and heads - is a product over windows of
// its input rows with the carried rows in front ([ctx; x] contiguous: a window of k rows is k * C consecutive floats),
// so a few output rows take the small-batch product (vvlm.cpp) and many take the tiled product here.
#include "ns.h"
#include "ns_internal.hpp"
#include "vv.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>

namespace {
using half = sycl::half;

inline float bf16f(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t) b << 16);
}

template <int WK>
inline float wload(const void* w, int64_t i) {
    if constexpr (WK == 0) return (float) ((const half*) w)[i];
    else if constexpr (WK == 1) return bf16f(((const uint16_t*) w)[i]);
    else return ((const float*) w)[i];
}

// out = epi(x . W^T + bias) for M rows of x (ldx apart; windows may overlap), W [N, K]: 64 x 64 tiles of the output a
// work-group of 256, each work-item 4 x 4 of them (rows ty + 16 i, columns tx + 16 j), K in steps of 16 through
// local memory
template <int WK>
void gemm(sycl::queue& q, const float* x, int64_t M, int64_t ldx, int64_t K, const void* w, int64_t N, const float* bias, int epi, const float* gamma,
          int64_t ldg, float* out, int64_t ldo) {
    constexpr int TM = 64, TN = 64, TK = 16;
    const size_t gm = (size_t) ((M + TM - 1) / TM), gn = (size_t) ((N + TN - 1) / TN);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> xs(sycl::range<2>(TK, TM + 1), h);
        sycl::local_accessor<float, 2> ws(sycl::range<2>(TK, TN + 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(gm * 16, gn * 16), sycl::range<2>(16, 16)), [=](sycl::nd_item<2> it) {
            const int ty = (int) it.get_local_id(0), tx = (int) it.get_local_id(1), lid = ty * 16 + tx;
            const int64_t m0 = (int64_t) it.get_group(0) * TM, n0 = (int64_t) it.get_group(1) * TN;
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
            // each work-item stages 4 values of each tile: row lid / 4, k (lid % 4) * 4 ...
            const int lr = lid / 4, lk = (lid % 4) * 4;
            for (int64_t k0 = 0; k0 < K; k0 += TK) {
                const int64_t m = m0 + lr, n = n0 + lr;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int64_t k = k0 + lk + e;
                    xs[lk + e][lr] = m < M && k < K ? x[m * ldx + k] : 0.0f;
                    ws[lk + e][lr] = n < N && k < K ? wload<WK>(w, n * K + k) : 0.0f;
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int kk = 0; kk < TK; ++kk) {
                    float a[4], b[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) a[i] = xs[kk][ty + 16 * i];
#pragma unroll
                    for (int j = 0; j < 4; ++j) b[j] = ws[kk][tx + 16 * j];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
                }
                sycl::group_barrier(it.get_group());
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t m = m0 + ty + 16 * i;
                if (m >= M) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int64_t n = n0 + tx + 16 * j;
                    if (n >= N) continue;
                    const float v = acc[i][j] + (bias ? bias[n] : 0.0f);
                    float* o = out + m * ldo + n;
                    switch (epi) {
                    case NS_VV_STORE: *o = v; break;
                    case NS_VV_ACC: *o += v; break;
                    case NS_VV_GELU: *o = 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f)); break;
                    case NS_VV_RES: *o += gamma[m * ldg + n] * v; break;
                    default: break;
                    }
                }
            }
        });
    });
}

// what follows a product for one output o (v: the sum and its bias; g: the residual's layer-scale index)
inline void epilogue(int epi, float* o, float v, const float* gamma, int64_t g) {
    switch (epi) {
    case NS_VV_STORE: *o = v; break;
    case NS_VV_ACC: *o += v; break;
    case NS_VV_GELU: *o = 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865476f)); break;
    case NS_VV_RES: *o += gamma[g] * v; break;
    default: break;
    }
}

// The same product on the matrix engines for bfloat16 weights, x given split into two bfloat16 parts (ns_audio_vv_split:
// hi = x rounded, lo = x - hi rounded; together x to ~2^-17 of itself): both multiplied by the exact weights and summed
// in float32 - float32 products to within ~1e-5, at the engines' rate. A work-group of 4 sub-groups computes 32 x 64 of
// the output (each sub-group 8 rows x 4 tiles of 16): x's tiles read from memory (rows ldx apart: windows), W's staged
// 32 values of K at a time in local memory as packed pairs (the engines' B layout). x's rows past M must be readable
// (the caller's buffers have room) and K a multiple of 16.
namespace jm = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

void gemm_xmx(sycl::queue& q, const uint16_t* xh, const uint16_t* xl, int64_t M, int64_t ldx, int64_t K, const uint16_t* w, int64_t N, const float* bias,
              int epi, const float* gamma, int64_t ldg, float* out, int64_t ldo, float* part, int64_t ks) {
    constexpr int SGS = 16, NSG = 4, WG = SGS * NSG, TM = 8 * NSG, TN = 64, TK = 32, NB = TN / 16;
    const int64_t gm = (M + TM - 1) / TM, gn = (N + TN - 1) / TN;
    // K in ks pieces (each a multiple of TK): with few output tiles, more work-groups share the work
    const int64_t kc = ((K + ks - 1) / ks + TK - 1) / TK * TK;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> bt(sycl::range<1>(TK / 2 * TN), h);
        sycl::local_accessor<float, 1> st(sycl::range<1>(TM * TN), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (gm * gn * ks) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SGS)]] {
            auto sg = it.get_sub_group();
            const int lid = (int) it.get_local_id(0), sgi = lid / SGS;
            const int64_t gi = it.get_group(0) / ks, kb = it.get_group(0) % ks;
            const int64_t m0 = (gi / gn) * TM, n0 = (gi % gn) * TN;
            const int64_t kbeg = kb * kc, kend = sycl::min(K, kbeg + kc);
            uint32_t* tl = bt.get_multi_ptr<sycl::access::decorated::no>().get();
            float* sp = st.get_multi_ptr<sycl::access::decorated::no>().get();
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> acc[NB];
#pragma unroll
            for (int j = 0; j < NB; ++j) jm::joint_matrix_fill(sg, acc[j], 0.0f);
            const int64_t mr = m0 + sgi * 8;
            for (int64_t k0 = kbeg; k0 < kend; k0 += TK) {
                sycl::group_barrier(it.get_group());   // (the last step's readers done)
                // W's [TN, TK / 2] pairs, neighbouring lanes on neighbouring pairs of a row
                for (int e = 0; e < TN * TK / 2 / WG; ++e) {
                    const int idx = e * WG + lid, nn = idx / (TK / 2), p = idx % (TK / 2);
                    const int64_t n = n0 + nn, k = k0 + 2 * p;
                    tl[p * TN + nn] = n < N && k < kend ? *reinterpret_cast<const uint32_t*>(w + n * K + k) : 0u;
                }
                sycl::group_barrier(it.get_group());
#pragma unroll
                for (int hk = 0; hk < TK / 16; ++hk) {
                    if (k0 + hk * 16 >= kend) break;
                    jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major> mh, ml;
                    jm::joint_matrix_load(sg, mh, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                                      reinterpret_cast<const bf16*>(xh + mr * ldx + k0 + hk * 16)), (size_t) ldx);
                    jm::joint_matrix_load(sg, ml, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                                      reinterpret_cast<const bf16*>(xl + mr * ldx + k0 + hk * 16)), (size_t) ldx);
#pragma unroll
                    for (int j = 0; j < NB; ++j) {
                        jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, 16, jm::layout::ext_intel_packed> bm;
                        jm::joint_matrix_load(sg, bm, sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                                                          reinterpret_cast<bf16*>(tl + hk * 8 * TN + j * 16)), (size_t) (TN * 2));
                        jm::joint_matrix_mad(sg, acc[j], mh, bm, acc[j]);
                        jm::joint_matrix_mad(sg, acc[j], ml, bm, acc[j]);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NB; ++j)
                jm::joint_matrix_store(sg, acc[j], sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                                                       sp + sgi * 8 * TN + j * 16), (size_t) TN, jm::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int idx = lid; idx < TM * TN; idx += WG) {
                const int64_t m = m0 + idx / TN, n = n0 + idx % TN;
                if (m >= M || n >= N) continue;
                if (ks > 1) {
                    part[(kb * M + m) * N + n] = sp[idx];
                    continue;
                }
                epilogue(epi, out + m * ldo + n, sp[idx] + (bias ? bias[n] : 0.0f), gamma, m * ldg + n);
            }
        });
    });
    if (ks > 1) {
        // the pieces summed in order, then what follows the product
        q.parallel_for(sycl::range<1>((size_t) (M * N)), [=](sycl::id<1> id) {
            const int64_t m = id[0] / N, n = id[0] % N;
            float s = 0.0f;
            for (int64_t j = 0; j < ks; ++j) s += part[(j * M + m) * N + n];
            epilogue(epi, out + m * ldo + n, s + (bias ? bias[n] : 0.0f), gamma, m * ldg + n);
        });
    }
}

// the split of K a matrix-engine product takes: doubled while the output's tiles (32 x 64) and pieces make fewer than
// 256 work-groups and each piece keeps 128 values of K or more
int64_t xmx_ks(int64_t M, int64_t N, int64_t K) {
    const int64_t tiles = ((M + 31) / 32) * ((N + 63) / 64);
    int64_t ks = 1;
    while (tiles * ks < 256 && K / (ks * 2) >= 128 && ks < 32) ks *= 2;
    return ks;
}

// NS_VV_GEMM_SIMT=1: the bfloat16 products on the plain tiled kernel too (an A/B of the matrix engines' form)
bool simt() {
    static const bool v = [] { const char* e = std::getenv("NS_VV_GEMM_SIMT"); return e && e[0] == '1'; }();
    return v;
}

// a work-group's size for rows of C channels
inline int row_wg(int64_t C) {
    return C >= 1024 ? 256 : C >= 256 ? 128 : 64;
}
}  // namespace

extern "C" {

int64_t ns_audio_vv_gemm_scratch(int64_t M, int64_t N, int64_t K) {
    const int64_t ks = xmx_ks(M, N, K);
    return ks > 1 ? ks * M * N : 0;
}

int ns_audio_vv_gemm(ns_gpu* g, const float* x, const void* xh, const void* xl, int64_t M, int64_t ldx, int64_t K, const void* w, int wt, int64_t N,
                     const float* bias, int epi, const float* gamma, int64_t ldg, float* out, int64_t ldo, float* part, int64_t part_len) {
    NS_TRY
    if (M <= 0 || N <= 0) return 0;
    if (epi == NS_VV_SWIGLU) return ns_fail("ns_audio_vv_gemm: no SwiGLU (the small-batch product has it)");
    if (epi == NS_VV_RES && !gamma) return ns_fail("ns_audio_vv_gemm: a residual's layer scale");
    if (wt == 0) gemm<0>(g->q, x, M, ldx, K, w, N, bias, epi, gamma, ldg, out, ldo);
    else if (wt == 1 && xh && xl && K % 16 == 0 && ldx % 8 == 0 && !simt()) {
        const int64_t ks = part && part_len >= ns_audio_vv_gemm_scratch(M, N, K) ? xmx_ks(M, N, K) : 1;
        gemm_xmx(g->q, (const uint16_t*) xh, (const uint16_t*) xl, M, ldx, K, (const uint16_t*) w, N, bias, epi, gamma, ldg, out, ldo, part, ks);
    }
    else if (wt == 1) gemm<1>(g->q, x, M, ldx, K, w, N, bias, epi, gamma, ldg, out, ldo);
    else if (wt == 3) gemm<3>(g->q, x, M, ldx, K, w, N, bias, epi, gamma, ldg, out, ldo);
    else return ns_fail("ns_audio_vv_gemm: weights half (0), bfloat16 (1) or float32 (3)");
    return 0;
    NS_CATCH
}

int ns_audio_vv_split(ns_gpu* g, const float* x, int64_t n, void* hi, void* lo) {
    NS_TRY
    if (n <= 0) return 0;
    bf16* h = (bf16*) hi;
    bf16* l = (bf16*) lo;
    g->q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const float v = x[id[0]];
        const bf16 a(v);
        h[id[0]] = a;
        l[id[0]] = bf16(v - (float) a);
    });
    return 0;
    NS_CATCH
}

int ns_audio_vv_norm_hist(ns_gpu* g, const float* prev, int64_t prev_off, const float* x, int64_t L, int64_t C, int64_t P, const float* w, float eps,
                          float* cur) {
    NS_TRY
    if (C <= 0 || L + P <= 0) return 0;
    const int WG = row_wg(C);
    // a work-group a row of cur: the carried rows copied, the new ones normed
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) (P + L) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        const int lid = (int) it.get_local_id(0);
        float* o = cur + r * C;
        if (r < P) {
            const float* s = prev + (prev_off + r) * C;
            for (int64_t i = lid; i < C; i += WG) o[i] = s[i];
            return;
        }
        const float* xr = x + (r - P) * C;
        float s = 0.0f;
        for (int64_t i = lid; i < C; i += WG) s += xr[i] * xr[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = sycl::rsqrt(s / (float) C + eps);
        for (int64_t i = lid; i < C; i += WG) o[i] = xr[i] * inv * w[i];
    });
    return 0;
    NS_CATCH
}

int ns_audio_vv_block_mix(ns_gpu* g, const float* cur, float* x, int64_t L, int64_t C, int64_t K, const float* dw_w, const float* dw_b, const float* gamma,
                          const float* w2, float eps, float* h) {
    NS_TRY
    if (L <= 0 || C <= 0) return 0;
    if (K < 1 || K > 32) return ns_fail("ns_audio_vv_block_mix: a kernel of 1 to 32 taps");
    const int WG = row_wg(C);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) L * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int64_t t = it.get_group(0);
        const int lid = (int) it.get_local_id(0);
        float* xr = x + t * C;
        float s = 0.0f;
        for (int64_t c = lid; c < C; c += WG) {
            float a = dw_b ? dw_b[c] : 0.0f;
            for (int64_t k = 0; k < K; ++k) a += dw_w[c * K + k] * cur[(t + k) * C + c];
            const float v = xr[c] + gamma[c] * a;
            xr[c] = v;
            s += v * v;
        }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = sycl::rsqrt(s / (float) C + eps);
        for (int64_t c = lid; c < C; c += WG) h[t * C + c] = xr[c] * inv * w2[c];
    });
    return 0;
    NS_CATCH
}

int ns_audio_vv_shift(ns_gpu* g, float* buf, int64_t P, int64_t L, int64_t C) {
    NS_TRY
    if (P <= 0 || C <= 0 || L <= 0) return 0;
    // a work-item a channel, its rows in order (a source row is past every row written before it)
    g->q.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int64_t c = id[0];
        for (int64_t i = 0; i < P; ++i) buf[i * C + c] = buf[(L + i) * C + c];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
