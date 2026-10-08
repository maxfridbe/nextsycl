// Build in the oneAPI image beside ggml-common.h: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g31" esimd-down.cpp -lmkl_sycl ...
// An expert's down at prompt width in ESIMD: C[M,N] = X[M,K] (fp16) . W[N,K]^T with W in Q2_K (N = 4096, K = 2048), a
// work-group over all the rows sharing the decode through local memory (as esimd-fused.cpp's gate | up); against
// expanding to fp16 and oneMKL, and the joint_matrix kernel (down.cpp).
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
constexpr int EX = 16, DEPTH = 8, RC = 8, KS = 16, RM2 = 2;

template <int T>
void esimd_down(queue& q, const half* A, const block_q2_K* W, float* C, int M, int N, int K) {
    constexpr int TILE = KS * EX * 2, BUF = 16 * TILE;
    q.parallel_for(nd_range<1>(range<1>((N / EX) * T), range<1>(T)), [=](nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<2 * BUF>();
        const int t = it.get_local_id(0);
        const int n0 = it.get_group(0) * EX, m0 = t * RC * RM2;
        es::simd<float, RC * EX> acc[RM2];
        for (int i = 0; i < RM2; ++i) acc[i] = 0.f;
        const uint32_t rb = (uint32_t) ((K / 256) * sizeof(block_q2_K));
        const uint8_t* wb = (const uint8_t*) W;
        es::simd<uint32_t, EX> rows(0, 1);
        rows = rows * rb + (uint32_t) (n0 * rb);
        for (int kb = 0; kb < K; kb += 256) {
            const uint32_t buf = ((kb / 256) % 2) * BUF;
            const es::simd<uint32_t, EX> off = rows + (uint32_t) ((kb / 256) * sizeof(block_q2_K));
            // d | dmin (the block's last 4 bytes)
            es::simd<uint32_t, EX> dm = es::gather<uint32_t, EX>((const uint32_t*) wb, off + 80u);
            es::simd<uint16_t, EX> dlo = dm & 0xffffu, dhi = dm >> 16;
            es::simd<half, EX> dh = dlo.template bit_cast_view<half>(), mh = dhi.template bit_cast_view<half>();
            const es::simd<float, EX> d = dh, dmin = mh;
            for (int tile = t; tile < 16; tile += T) {
                const int n = tile / 8, j = (tile % 8) / 2, hf = tile % 2;
                const int si = 8 * n + 2 * j + hf;
                es::simd<uint32_t, EX> sw = es::gather<uint32_t, EX>((const uint32_t*) wb, off + (uint32_t) (si & ~3));
                es::simd<uint32_t, EX> sc = (sw >> (8 * (si & 3))) & 0xffu;
                es::simd<float, EX> dl = d * es::simd<float, EX>(sc & 15u), ml = dmin * es::simd<float, EX>(sc >> 4);
                es::simd<half, KS * EX> b;   // VNNI: (k, n) at (k / 2) * 32 + n * 2 + k % 2
#pragma unroll
                for (int w4 = 0; w4 < 4; ++w4) {
                    es::simd<uint32_t, EX> qw = es::gather<uint32_t, EX>((const uint32_t*) wb, off + (uint32_t) (16 + 32 * n + 16 * hf + 4 * w4));
#pragma unroll
                    for (int bb = 0; bb < 4; ++bb) {
                        const int kk = w4 * 4 + bb;
                        es::simd<uint32_t, EX> qv = (qw >> (8 * bb + 2 * j)) & 3u;
                        es::simd<float, EX> v = dl * es::simd<float, EX>(qv) - ml;
                        b.template select<EX, 2>((kk / 2) * 32 + (kk % 2)) = v;
                    }
                }
                const uint32_t at = buf + (uint32_t) (tile * TILE);
                es::slm_block_store<half, 128>(at, b.template select<128, 1>(0));
                es::slm_block_store<half, 128>(at + 256, b.template select<128, 1>(128));
            }
            es::barrier();
#pragma unroll 4
            for (int tile = 0; tile < 16; ++tile) {
                const uint32_t at = buf + (uint32_t) (tile * TILE);
                es::simd<half, KS * EX> b;
                b.template select<128, 1>(0) = es::slm_block_load<half, 128>(at);
                b.template select<128, 1>(128) = es::slm_block_load<half, 128>(at + 256);
#pragma unroll
                for (int i = 0; i < RM2; ++i) {
                    es::simd<half, RC * KS> a = es::load_2d<half, KS, RC>(A, (unsigned) (K * 2 - 1), (unsigned) (M - 1), (unsigned) (K * 2 - 1),
                                                                          kb + tile * 16, m0 + i * RC);
                    acc[i] = xmx::dpas<DEPTH, RC, float, float, half, half>(acc[i], b, a);
                }
            }
        }
        for (int i = 0; i < RM2; ++i)
#pragma unroll
            for (int r = 0; r < RC; ++r)
                es::block_store<float, EX>(C + (size_t) (m0 + i * RC + r) * N + n0, acc[i].template select<EX, 1>(r * EX));
    });
}

inline float q2v(const block_q2_K* b, int v) {
    const int n = v / 128, j = (v % 128) / 32, hf = (v % 32) / 16, l = v % 16;
    const uint8_t sc = b->scales[8 * n + 2 * j + hf];
    const int q = (b->qs[32 * n + 16 * hf + l] >> (2 * j)) & 3;
    return (float) b->dm[0] * (sc & 0xF) * q - (float) b->dm[1] * (sc >> 4);
}
void expand(queue& q, const block_q2_K* W, half* Wh, size_t n) {
    q.parallel_for(range<1>(n), [=](id<1> i) { Wh[i] = half(q2v(W + i / 256, i % 256)); });
}

int main(int argc, char** argv) {
    queue q(gpu_selector_v, property::queue::in_order());
    const int M = argc > 1 ? atoi(argv[1]) : 128, N = 4096, K = 2048, NE = 32;
    printf("%s M=%d\n", q.get_device().get_info<info::device::name>().c_str(), M);
    std::mt19937 rng(7);
    std::vector<half> ha((size_t) M * K);
    for (auto& x : ha) x = half(std::uniform_real_distribution<float>(-1, 1)(rng));
    std::vector<block_q2_K> hw((size_t) N * K / 256);
    for (auto& b : hw) {
        b.dm = half2(std::uniform_real_distribution<float>(0.001f, 0.01f)(rng), std::uniform_real_distribution<float>(0.001f, 0.01f)(rng));
        for (auto& s : b.scales) s = rng() & 0xff; for (auto& s : b.qs) s = rng() & 0xff;
    }
    half* A = malloc_device<half>(ha.size(), q);
    block_q2_K* W0 = malloc_device<block_q2_K>(hw.size() * NE, q);
    for (int e = 0; e < NE; ++e) q.memcpy(W0 + hw.size() * e, hw.data(), hw.size() * sizeof(block_q2_K)).wait();
    half* Wh = malloc_device<half>((size_t) N * K, q);
    float *C1 = malloc_device<float>((size_t) M * N, q), *C2 = malloc_device<float>((size_t) M * N, q);
    q.memcpy(A, ha.data(), ha.size() * 2).wait();
    const block_q2_K* W = W0;
    auto time = [&](const char* name, auto f) {
        for (int i = 0; i < 3; ++i) f(); q.wait();
        const int R = 50; int rot = 0; auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < R; ++i) { W = W0 + hw.size() * (rot++ % NE); f(); } q.wait();
        W = W0;
        printf("  %-28s %8.1f us\n", name, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / R * 1e6);
    };
    using oneapi::mkl::transpose;
    auto mkl = [&] { oneapi::mkl::blas::row_major::gemm(q, transpose::nontrans, transpose::trans, M, N, K, 1.0f, A, K, Wh, K, 0.0f, C1, N); };
    time("  oneMKL alone (expanded)", mkl);
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
    check("esimd down, one group", [&] {
        switch (M / 16) {
            case 2: esimd_down<2>(q, A, W, C2, M, N, K); break;
            case 4: esimd_down<4>(q, A, W, C2, M, N, K); break;
            case 8: esimd_down<8>(q, A, W, C2, M, N, K); break;
            case 16: esimd_down<16>(q, A, W, C2, M, N, K); break;
            default: esimd_down<32>(q, A, W, C2, M, N, K); break;
        }
    });
}
