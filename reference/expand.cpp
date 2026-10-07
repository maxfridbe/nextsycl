// Build in the oneAPI image beside kernels/strata/third_party/ggml/ggml-common.h: icpx -fsycl -O3 -fsycl-targets=spir64_gen -Xs "-device bmg-g31" expand.cpp
// The prompt path's expert expansion (compressed -> fp16 in VRAM) against the memory's ceiling: Strata's form (a
// work-group of 32 a block, 8 values a work-item) and variants, for IQ2_XXS (gate/up) and Q2_K (down).
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include <sycl/sycl.hpp>
#include "ggml-common.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
using namespace sycl;

// 8 values (v0 = 8 * tid of block b) of IQ2_XXS
inline void iq2_8(const block_iq2_xxs* x, int tid, float* v) {
    const int il = tid % 4, ib = tid / 4;
    const uint16_t* q2 = x->qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const sycl::uint2 g = ((const sycl::uint2*) iq2xxs_grid)[aux8[il]];
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x->d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint32_t s7 = (aux32 >> 7 * il) & 127;
    const uint32_t signs = s7 | ((sycl::popcount(s7) & 1u) << 7);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t b = ((j < 4 ? g.x() : g.y()) >> (8 * (j % 4))) & 0xff;
        v[j] = d * (float) b * ((signs >> j) & 1 ? -1.f : 1.f);
    }
}
inline void q2k_8(const block_q2_K* x, int tid, float* v) {
    const int v0 = 8 * tid;
    const int n = v0 / 128, j = (v0 % 128) / 32, l2 = v0 % 32, sub = l2 / 16, l = l2 % 16;
    const uint8_t sc = x->scales[8 * n + 2 * j + sub];
    const sycl::float2 dm = x->dm.convert<float, sycl::rounding_mode::automatic>();
    const float d1 = dm.x() * (float) (sc & 0xF), m1 = dm.y() * (float) (sc >> 4);
    const uint8_t* q = x->qs + 32 * n + 16 * sub + l;
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = d1 * (float) ((q[k] >> (2 * j)) & 3) - m1;
}
template <class B> inline void dq8(const B* x, int tid, float* v);
template <> inline void dq8<block_iq2_xxs>(const block_iq2_xxs* x, int tid, float* v) { iq2_8(x, tid, v); }
template <> inline void dq8<block_q2_K>(const block_q2_K* x, int tid, float* v) { q2k_8(x, tid, v); }

inline void st8(half* y, const float* v) {
    vec<half, 8> h;
#pragma unroll
    for (int j = 0; j < 8; ++j) h[j] = half(v[j]);
    *reinterpret_cast<vec<half, 8>*>(y) = h;
}

// BPG blocks a work-group of 32 * BPG, a work-item 8 values (BPG = 1: Strata's form)
template <class B, int BPG>
void expand_wg(queue& q, const B* x, half* y, int64_t n) {
    const int64_t nb = n / 256;
    q.parallel_for(nd_range<1>(nb / BPG * 32 * BPG, 32 * BPG), [=](nd_item<1> it) {
        const int64_t b = it.get_global_id(0) / 32;
        const int tid = it.get_global_id(0) % 32;
        float v[8];
        dq8<B>(x + b, tid, v);
        st8(y + b * 256 + 8 * tid, v);
    });
}
// a work-item K blocks in a row (grid-stride: work-item w takes blocks w / 32 + k * total / 32)
template <class B, int K>
void expand_multi(queue& q, const B* x, half* y, int64_t n) {
    const int64_t nb = n / 256, items = nb / K * 32;
    q.parallel_for(nd_range<1>(items, 256), [=](nd_item<1> it) {
        const int64_t w = it.get_global_id(0);
        const int tid = w % 32;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int64_t b = w / 32 + k * (items / 32);
            float v[8];
            dq8<B>(x + b, tid, v);
            st8(y + b * 256 + 8 * tid, v);
        }
    });
}
void fill(queue& q, half* y, int64_t n) {
    q.parallel_for(range<1>(n / 8), [=](id<1> i) { *reinterpret_cast<vec<half, 8>*>(y + i[0] * 8) = vec<half, 8>(half(1)); });
}

template <class B>
void run(queue& q, const char* tname, int64_t n, int NE) {
    std::mt19937 rng(7);
    std::vector<B> hb(n / 256);
    for (auto& b : hb) { uint8_t* p = (uint8_t*) &b; for (size_t i = 0; i < sizeof(B); ++i) p[i] = rng() & 0xff; }
    for (auto& b : hb) { if constexpr (std::is_same_v<B, block_iq2_xxs>) b.d = half(0.01f); else b.dm = half2(0.01f, 0.01f); }
    B* x0 = malloc_device<B>(hb.size() * NE, q);
    for (int e = 0; e < NE; ++e) q.memcpy(x0 + hb.size() * e, hb.data(), hb.size() * sizeof(B)).wait();
    half* y0 = malloc_device<half>(n * NE, q);
    half* ref = malloc_device<half>(n, q);
    expand_wg<B, 1>(q, x0, ref, n); q.wait();
    std::vector<half> hr(n), hy(n);
    q.memcpy(hr.data(), ref, n * 2).wait();
    const double bytes = n * 2.0 + hb.size() * sizeof(B);
    auto time = [&](const char* name, auto f, bool check) {
        int rot = 0;
        for (int i = 0; i < 3; ++i) f(x0, y0); q.wait();
        const int R = 60; auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < R; ++i) { const int e = rot++ % NE; f(x0 + hb.size() * e, y0 + n * e); } q.wait();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / R;
        bool ok = true;
        if (check) {
            q.memset(y0, 0, n * 2).wait(); f(x0, y0); q.wait();
            q.memcpy(hy.data(), y0, n * 2).wait();
            for (int64_t i = 0; i < n; ++i) if (*(uint16_t*)&hy[i] != *(uint16_t*)&hr[i]) { ok = false; break; }
        }
        printf("%-8s %-26s %7.1f us %6.0f GB/s %s\n", tname, name, s * 1e6, bytes / s / 1e9, check ? (ok ? "exact" : "MISMATCH") : "");
    };
    time("fp16 write (ceiling)", [&](const B*, half* y) { fill(q, y, n); }, false);
    time("WG 32 (Strata's)", [&](const B* x, half* y) { expand_wg<B, 1>(q, x, y, n); }, true);
    time("WG 256 (8 blocks)", [&](const B* x, half* y) { expand_wg<B, 8>(q, x, y, n); }, true);
    time("WG 1024 (32 blocks)", [&](const B* x, half* y) { expand_wg<B, 32>(q, x, y, n); }, true);
    time("2 blocks a work-item", [&](const B* x, half* y) { expand_multi<B, 2>(q, x, y, n); }, true);
    time("4 blocks a work-item", [&](const B* x, half* y) { expand_multi<B, 4>(q, x, y, n); }, true);
    free(x0, q); free(y0, q); free(ref, q);
}

int main() {
    queue q(gpu_selector_v, property::queue::in_order());
    printf("%s\n", q.get_device().get_info<info::device::name>().c_str());
    const int64_t n = 2048LL * 4096;   // one expert matrix
    run<block_iq2_xxs>(q, "IQ2_XXS", n, 32);
    run<block_q2_K>(q, "Q2_K", n, 32);
}
