// Build in the oneAPI image beside ggml-common.h: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g31" esimd-fused.cpp -lmkl_sycl ...
// An expert's gate | up at prompt width in ESIMD: C[M,N] = X[M,K] (fp16) . W[N,K]^T with W in IQ2_XXS, decoded
// straight into xmx::dpas's B operand (its VNNI layout, built with vector selects - no joint_matrix fill); against
// expanding W to fp16 and oneMKL's GEMM.
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <oneapi/mkl/blas.hpp>
#include "ggml-common.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
using namespace sycl;
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

// dpas on Xe2: execution size 16, systolic depth 8, fp16 two a channel - A 8 x 16, B 16 x 16 (VNNI), C 8 x 16
constexpr int EX = 16, DEPTH = 8, RC = 8, KS = 16;

// a step's weight words for 16 rows from row n: d (its block's first word), the 32-group's two words
SYCL_ESIMD_FUNCTION inline void words(const uint8_t* wb, uint32_t rb, int n, int kk, es::simd<uint32_t, EX>& dw, es::simd<uint32_t, EX>& qx,
                                      es::simd<uint32_t, EX>& qy) {
    const uint32_t blk = (uint32_t) ((kk / 256) * sizeof(block_iq2_xxs));
    const uint32_t grp = (uint32_t) (2 + 8 * ((kk % 256) / 32));
    es::simd<uint32_t, EX> off(0, 1);
    off = off * rb + (uint32_t) (n * rb) + blk;
    dw = es::gather<uint32_t, EX>((const uint32_t*) wb, off);
    qx = es::gather<uint32_t, EX>((const uint32_t*) wb, off + grp);
    qy = es::gather<uint32_t, EX>((const uint32_t*) wb, off + grp + 4);
}

// a thread: RM row tiles (8 tokens each) x NB column tiles (16 each); K in steps of 32 (one IQ2_XXS group a weight
// row: two B tiles each); A by 2D block loads (one an 8 x 16 tile)
template <int RM, int NB>
void fused_esimd(queue& q, const half* A, const block_iq2_xxs* W, float* C, int M, int N, int K, const uint32_t* grid32) {
    const int threads = (M / (RC * RM)) * (N / (EX * NB));
    q.parallel_for(range<1>(threads), [=](id<1> tid) SYCL_ESIMD_KERNEL {
        const int nt = tid[0] % (N / (EX * NB)), mt = tid[0] / (N / (EX * NB));
        const int n0 = nt * EX * NB, m0 = mt * RC * RM;
        es::simd<float, RC * EX> acc[RM][NB];
        for (int i = 0; i < RM; ++i) for (int c = 0; c < NB; ++c) acc[i][c] = 0.f;
        const uint32_t rb = (uint32_t) ((K / 256) * sizeof(block_iq2_xxs));
        const uint8_t* wb = (const uint8_t*) W;
        for (int k = 0; k < K; k += 32) {
            const uint32_t blk = (uint32_t) ((k / 256) * sizeof(block_iq2_xxs));
            const uint32_t grp = (uint32_t) (2 + 8 * ((k % 256) / 32));
            es::simd<half, KS * EX> b[NB][2];   // VNNI: (k, n) at (k / 2) * 32 + n * 2 + k % 2
#pragma unroll
            for (int c = 0; c < NB; ++c) {
                es::simd<uint32_t, EX> off(0, 1);
                off = off * rb + (uint32_t) ((n0 + c * EX) * rb) + blk;
                es::simd<uint32_t, EX> dw = es::gather<uint32_t, EX>((const uint32_t*) wb, off);
                es::simd<uint32_t, EX> qx = es::gather<uint32_t, EX>((const uint32_t*) wb, off + grp);
                es::simd<uint32_t, EX> qy = es::gather<uint32_t, EX>((const uint32_t*) wb, off + grp + 4);
                es::simd<uint16_t, EX> dbits = dw & 0xffffu;
                es::simd<half, EX> dh = dbits.template bit_cast_view<half>();
                es::simd<float, EX> d = dh;
                d = d * (0.5f + es::simd<float, EX>(qy >> 28)) * 0.25f;
#pragma unroll
                for (int il = 0; il < 4; ++il) {
                    es::simd<uint32_t, EX> gi = (qx >> (8 * il)) & 0xffu;
                    es::simd<uint32_t, EX> glo = es::gather<uint32_t, EX>(grid32, gi * 8);
                    es::simd<uint32_t, EX> ghi = es::gather<uint32_t, EX>(grid32, gi * 8 + 4);
                    es::simd<uint32_t, EX> s7 = (qy >> (7 * il)) & 127u;
                    es::simd<uint32_t, EX> signs = s7 | ((es::cbit(s7) & 1u) << 7);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        es::simd<uint32_t, EX> byte = ((j < 4 ? glo : ghi) >> (8 * (j % 4))) & 0xffu;
                        es::simd<float, EX> v = d * es::simd<float, EX>(byte);
                        v.merge(-v, ((signs >> j) & 1u) != 0);
                        const int kk = (il % 2) * 8 + j;
                        b[c][il / 2].template select<EX, 2>((kk / 2) * 32 + (kk % 2)) = v;
                    }
                }
            }
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    es::simd<half, RC * KS> a = es::load_2d<half, KS, RC>(A, (unsigned) (K * 2 - 1), (unsigned) (M - 1), (unsigned) (K * 2 - 1),
                                                                          k + hh * 16, m0 + i * RC);
#pragma unroll
                    for (int c = 0; c < NB; ++c) acc[i][c] = xmx::dpas<DEPTH, RC, float, float, half, half>(acc[i][c], b[c][hh], a);
                }
            }
        }
        for (int i = 0; i < RM; ++i)
            for (int c = 0; c < NB; ++c)
#pragma unroll
                for (int r = 0; r < RC; ++r)
                    es::block_store<float, EX>(C + (size_t) (m0 + i * RC + r) * N + n0 + c * EX, acc[i][c].template select<EX, 1>(r * EX));
    });
}

// a thread: RM row tiles (8 tokens each) x NB column tiles (16 each); K in steps of 32 (one IQ2_XXS group a weight
// row: two B tiles each); A by 2D block loads (one an 8 x 16 tile)
template <int RM, int NB>
void fused_pipe(queue& q, const half* A, const block_iq2_xxs* W, float* C, int M, int N, int K, const uint32_t* grid32) {
    const int threads = (M / (RC * RM)) * (N / (EX * NB));
    q.parallel_for(range<1>(threads), [=](id<1> tid) SYCL_ESIMD_KERNEL {
        const int nt = tid[0] % (N / (EX * NB)), mt = tid[0] / (N / (EX * NB));
        const int n0 = nt * EX * NB, m0 = mt * RC * RM;
        es::simd<float, RC * EX> acc[RM][NB];
        for (int i = 0; i < RM; ++i) for (int c = 0; c < NB; ++c) acc[i][c] = 0.f;
        const uint32_t rb = (uint32_t) ((K / 256) * sizeof(block_iq2_xxs));
        const uint8_t* wb = (const uint8_t*) W;
        // the weight words of a step - d (its block's first word), the group's two words - a step ahead
        es::simd<uint32_t, EX> ndw[NB], nqx[NB], nqy[NB];
        for (int c = 0; c < NB; ++c) words(wb, rb, n0 + c * EX, 0, ndw[c], nqx[c], nqy[c]);
        for (int k = 0; k < K; k += 32) {
            es::simd<half, KS * EX> b[NB][2];   // VNNI: (k, n) at (k / 2) * 32 + n * 2 + k % 2
#pragma unroll
            for (int c = 0; c < NB; ++c) {
                es::simd<uint32_t, EX> dw = ndw[c], qx = nqx[c], qy = nqy[c];
                if (k + 32 < K) words(wb, rb, n0 + c * EX, k + 32, ndw[c], nqx[c], nqy[c]);
                es::simd<uint16_t, EX> dbits = dw & 0xffffu;
                es::simd<half, EX> dh = dbits.template bit_cast_view<half>();
                es::simd<float, EX> d = dh;
                d = d * (0.5f + es::simd<float, EX>(qy >> 28)) * 0.25f;
#pragma unroll
                for (int il = 0; il < 4; ++il) {
                    es::simd<uint32_t, EX> gi = (qx >> (8 * il)) & 0xffu;
                    es::simd<uint32_t, EX> glo = es::gather<uint32_t, EX>(grid32, gi * 8);
                    es::simd<uint32_t, EX> ghi = es::gather<uint32_t, EX>(grid32, gi * 8 + 4);
                    es::simd<uint32_t, EX> s7 = (qy >> (7 * il)) & 127u;
                    es::simd<uint32_t, EX> signs = s7 | ((es::cbit(s7) & 1u) << 7);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        es::simd<uint32_t, EX> byte = ((j < 4 ? glo : ghi) >> (8 * (j % 4))) & 0xffu;
                        es::simd<float, EX> v = d * es::simd<float, EX>(byte);
                        v.merge(-v, ((signs >> j) & 1u) != 0);
                        const int kk = (il % 2) * 8 + j;
                        b[c][il / 2].template select<EX, 2>((kk / 2) * 32 + (kk % 2)) = v;
                    }
                }
            }
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    es::simd<half, RC * KS> a = es::load_2d<half, KS, RC>(A, (unsigned) (K * 2 - 1), (unsigned) (M - 1), (unsigned) (K * 2 - 1),
                                                                          k + hh * 16, m0 + i * RC);
#pragma unroll
                    for (int c = 0; c < NB; ++c) acc[i][c] = xmx::dpas<DEPTH, RC, float, float, half, half>(acc[i][c], b[c][hh], a);
                }
            }
        }
        for (int i = 0; i < RM; ++i)
            for (int c = 0; c < NB; ++c)
#pragma unroll
                for (int r = 0; r < RC; ++r)
                    es::block_store<float, EX>(C + (size_t) (m0 + i * RC + r) * N + n0 + c * EX, acc[i][c].template select<EX, 1>(r * EX));
    });
}

// A work-group of T threads shares a 16-column slice of N: per 256-k block (one IQ2_XXS block a weight row) the
// threads decode its 8 groups between them (16 B tiles in all) into local memory, then each loads every tile for its
// own RM row tiles' dpas. The grid table in local memory; each row's group in one 8-byte gather (U64) or two 4-byte.
template <int RM, int T, bool U64>
void fused_wg(queue& q, const half* A, const block_iq2_xxs* W, float* C, int M, int N, int K, const uint32_t* grid32) {
    constexpr int GRID = 2048, TILE = KS * EX * 2, BUF = 16 * TILE;   // bytes: the table, a B tile, a block's tiles
    const int mblocks = M / (RC * RM * T);
    q.parallel_for(nd_range<1>(range<1>(mblocks * (N / EX) * T), range<1>(T)), [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<GRID + 2 * BUF>();
        const int t = it.get_local_id(0), g = it.get_group(0);
        const int nt = g % (N / EX), mb = g / (N / EX);
        const int n0 = nt * EX, m0 = (mb * T + t) * RC * RM;
        // the table into local memory, the threads in turn
        for (int o = t * 256; o < GRID; o += T * 256)
            es::slm_block_store<uint32_t, 64>(o, es::block_load<uint32_t, 64>(grid32 + o / 4));
        es::barrier();
        es::simd<float, RC * EX> acc[RM];
        for (int i = 0; i < RM; ++i) acc[i] = 0.f;
        const uint32_t rb = (uint32_t) ((K / 256) * sizeof(block_iq2_xxs));
        const uint8_t* wb = (const uint8_t*) W;
        es::simd<uint32_t, EX> rows(0, 1);
        rows = rows * rb + (uint32_t) (n0 * rb);
        for (int kb = 0; kb < K; kb += 256) {
            const uint32_t buf = GRID + ((kb / 256) % 2) * BUF;
            const es::simd<uint32_t, EX> off = rows + (uint32_t) ((kb / 256) * sizeof(block_iq2_xxs));
            es::simd<uint32_t, EX> dw = es::gather<uint32_t, EX>((const uint32_t*) wb, off);
            es::simd<uint16_t, EX> dbits = dw & 0xffffu;
            es::simd<half, EX> dh = dbits.template bit_cast_view<half>();
            const es::simd<float, EX> d0 = dh;
            for (int grp = t; grp < 8; grp += T) {
                es::simd<uint32_t, EX> qx, qy;
                if constexpr (U64) {
                    es::simd<uint64_t, EX> qq = es::gather<uint64_t, EX>((const uint64_t*) wb, off + (uint32_t) (2 + 8 * grp));
                    qx = es::simd<uint32_t, EX>(qq & 0xffffffffull);
                    qy = es::simd<uint32_t, EX>(qq >> 32);
                } else {
                    qx = es::gather<uint32_t, EX>((const uint32_t*) wb, off + (uint32_t) (2 + 8 * grp));
                    qy = es::gather<uint32_t, EX>((const uint32_t*) wb, off + (uint32_t) (6 + 8 * grp));
                }
                es::simd<float, EX> d = d0 * (0.5f + es::simd<float, EX>(qy >> 28)) * 0.25f;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    es::simd<half, KS * EX> b;   // VNNI: (k, n) at (k / 2) * 32 + n * 2 + k % 2
#pragma unroll
                    for (int g2 = 0; g2 < 2; ++g2) {
                        const int il = hh * 2 + g2;
                        es::simd<uint32_t, EX> gi = (qx >> (8 * il)) & 0xffu;
                        es::simd<uint32_t, EX> glo = es::slm_gather<uint32_t, EX>(gi * 8);
                        es::simd<uint32_t, EX> ghi = es::slm_gather<uint32_t, EX>(gi * 8 + 4);
                        es::simd<uint32_t, EX> s7 = (qy >> (7 * il)) & 127u;
                        es::simd<uint32_t, EX> signs = s7 | ((es::cbit(s7) & 1u) << 7);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            es::simd<uint32_t, EX> byte = ((j < 4 ? glo : ghi) >> (8 * (j % 4))) & 0xffu;
                            es::simd<float, EX> v = d * es::simd<float, EX>(byte);
                            v.merge(-v, ((signs >> j) & 1u) != 0);
                            const int kk = g2 * 8 + j;
                            b.template select<EX, 2>((kk / 2) * 32 + (kk % 2)) = v;
                        }
                    }
                    const uint32_t at = buf + (uint32_t) ((2 * grp + hh) * TILE);
                    es::slm_block_store<half, 128>(at, b.template select<128, 1>(0));
                    es::slm_block_store<half, 128>(at + 256, b.template select<128, 1>(128));
                }
            }
            es::barrier();
#pragma unroll 4
            for (int tile = 0; tile < 16; ++tile) {
                const uint32_t at = buf + (uint32_t) (tile * TILE);
                es::simd<half, KS * EX> b;
                b.template select<128, 1>(0) = es::slm_block_load<half, 128>(at);
                b.template select<128, 1>(128) = es::slm_block_load<half, 128>(at + 256);
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    es::simd<half, RC * KS> a = es::load_2d<half, KS, RC>(A, (unsigned) (K * 2 - 1), (unsigned) (M - 1), (unsigned) (K * 2 - 1),
                                                                          kb + tile * 16, m0 + i * RC);
                    acc[i] = xmx::dpas<DEPTH, RC, float, float, half, half>(acc[i], b, a);
                }
            }
        }
        for (int i = 0; i < RM; ++i)
#pragma unroll
            for (int r = 0; r < RC; ++r)
                es::block_store<float, EX>(C + (size_t) (m0 + i * RC + r) * N + n0, acc[i].template select<EX, 1>(r * EX));
    });
}

inline void dq8(const block_iq2_xxs* b, int ib, int il, float* v) {
    const uint16_t* q2 = b->qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) b->d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint32_t s7 = (aux32 >> 7 * il) & 127;
    const uint32_t signs = s7 | ((__builtin_popcount(s7) & 1u) << 7);
    const uint64_t g = iq2xxs_grid[aux8[il]];
    for (int j = 0; j < 8; ++j) v[j] = d * (float) ((g >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f);
}
void expand(queue& q, const block_iq2_xxs* W, half* Wh, size_t n) {
    q.parallel_for(range<1>(n / 8), [=](id<1> i) {
        const size_t v0 = i[0] * 8, blk = v0 / 256;
        float v[8];
        dq8(W + blk, (v0 % 256) / 32, (v0 % 32) / 8, v);
        for (int j = 0; j < 8; ++j) Wh[v0 + j] = half(v[j]);
    });
}

int main(int argc, char** argv) {
    queue q(gpu_selector_v, property::queue::in_order());
    const int M = argc > 1 ? atoi(argv[1]) : 128, N = 4096, K = 4096, NE = 32;
    printf("%s M=%d\n", q.get_device().get_info<info::device::name>().c_str(), M);
    std::mt19937 rng(7);
    std::vector<half> ha((size_t) M * K);
    for (auto& x : ha) x = half(std::uniform_real_distribution<float>(-1, 1)(rng));
    std::vector<block_iq2_xxs> hw((size_t) N * K / 256);
    for (auto& b : hw) { b.d = half(std::uniform_real_distribution<float>(0.001f, 0.01f)(rng)); for (auto& w : b.qs) w = rng() & 0xffff; }
    half* A = malloc_device<half>(ha.size(), q);
    block_iq2_xxs* W0 = malloc_device<block_iq2_xxs>(hw.size() * NE, q);
    for (int e = 0; e < NE; ++e) q.memcpy(W0 + hw.size() * e, hw.data(), hw.size() * sizeof(block_iq2_xxs)).wait();
    uint32_t* grid = malloc_device<uint32_t>(512, q);
    q.memcpy(grid, iq2xxs_grid, 256 * 8).wait();
    half* Wh = malloc_device<half>((size_t) N * K, q);
    float *C1 = malloc_device<float>((size_t) M * N, q), *C2 = malloc_device<float>((size_t) M * N, q);
    q.memcpy(A, ha.data(), ha.size() * 2).wait();
    const block_iq2_xxs* W = W0;
    auto time = [&](const char* name, auto f) {
        for (int i = 0; i < 3; ++i) f(); q.wait();
        const int R = 50; int rot = 0; auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < R; ++i) { W = W0 + hw.size() * (rot++ % NE); f(); } q.wait();
        W = W0;
        printf("  %-28s %8.1f us\n", name, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / R * 1e6);
    };
    using oneapi::mkl::transpose;
    auto mkl = [&] { oneapi::mkl::blas::row_major::gemm(q, transpose::nontrans, transpose::trans, M, N, K, 1.0f, A, K, Wh, K, 0.0f, C1, N); };
    time("expand + oneMKL", [&] { expand(q, W, Wh, (size_t) N * K); mkl(); });
    time("  of it, oneMKL alone", mkl);
    expand(q, W0, Wh, (size_t) N * K); mkl(); q.wait();
    std::vector<float> c1((size_t) M * N), c2((size_t) M * N);
    q.memcpy(c1.data(), C1, c1.size() * 4).wait();
    auto check = [&](const char* name, auto f) {
        q.memset(C2, 0, c2.size() * 4).wait();
        time(name, f);
        q.memcpy(c2.data(), C2, c2.size() * 4).wait();
        double md = 0, mx = 0; for (size_t i = 0; i < c1.size(); ++i) { md = std::max(md, (double) std::abs(c1[i] - c2[i])); mx = std::max(mx, (double) std::abs(c1[i])); }
        printf("  %28s   max |diff| %.1e of %.1e\n", "", md, mx);
    };
    // one work-group over all the rows, 16 a thread (RM = 2): every weight decoded once
    auto one = [&](int m) {
        switch (m / 16) {
            case 2: fused_wg<2, 2, false>(q, A, W, C2, M, N, K, grid); break;
            case 4: fused_wg<2, 4, false>(q, A, W, C2, M, N, K, grid); break;
            case 8: fused_wg<2, 8, false>(q, A, W, C2, M, N, K, grid); break;
            case 16: fused_wg<2, 16, false>(q, A, W, C2, M, N, K, grid); break;
            default: fused_wg<2, 32, false>(q, A, W, C2, M, N, K, grid); break;
        }
    };
    check("wg, one group (RM=2)", [&] { one(M); });
}
