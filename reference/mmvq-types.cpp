// Build in the oneAPI image: icpx -O2 -Ikernels/ns reference/mmvq-types.cpp -o mmvq_types dist/libnextsycl-llm.so
//   dist/silo/libnextsycl-qwen35.so -Wl,-rpath,dist:dist/silo
// The decode products (ns_mmvq: Strata's native and IQ kernels) by ggml type at a dense 27B's shapes, one column:
// GB/s of weights read, 8 matrices rotated (past the L2), random blocks (the arithmetic does not depend on them).
// `mmvq_types GPU NCOLS [check][own]`: check - random activations instead, and each product's output summarized (sum,
// sum of |y|, y[0], y[n_out-1]), to compare two runs; own - the qwen35 engine's own kernels (ns_q35_mmvq) where they
// cover the type (its silo), the shared ones (ns_mmvq) otherwise; peaks - the card's practical ceilings instead (a
// device copy's bandwidth, oneMKL's half GEMM at a prompt chunk's shapes).
#include "ns.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
extern "C" {
size_t ns_q8_1_bytes(int64_t n_in, int64_t ncols);
int ns_mmvq_supported(int type);
int ns_quantize_q8_1(ns_gpu* g, const float* x, void* q8_1, int64_t n_in, int64_t ncols);
int ns_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
int ns_q35_mmvq_supported(int type, int64_t n_in);
int ns_gemm_f16(ns_gpu* g, int64_t T, int64_t N, int64_t K, const uint16_t* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int acc);
int ns_q35_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
}
struct T { int id; const char* name; int elems; int bytes; };
int main(int argc, char** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    const int ncols = argc > 2 ? atoi(argv[2]) : 1;
    const std::string mode = argc > 3 ? argv[3] : "";
    const bool check = mode.find("check") != std::string::npos, own = mode.find("own") != std::string::npos;
    ns_gpu* g;
    if (ns_gpu_open(dev, &g)) { printf("open: %s\n", ns_last_error()); return 1; }
    const T types[] = {{8, "Q8_0", 32, 34}, {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210}, {23, "IQ4_XS", 256, 136},
                       {20, "IQ4_NL", 32, 18}, {11, "Q3_K", 256, 110}, {21, "IQ3_S", 256, 110}, {18, "IQ3_XXS", 256, 98}, {22, "IQ2_S", 256, 82},
                       {10, "Q2_K", 256, 84}, {2, "Q4_0", 32, 18}};
    const int64_t shapes[][2] = {{17408, 5120}, {5120, 17408}, {10240, 5120}};
    const int NE = 8;
    float* x; ns_alloc(g, 17408 * 8 * 4, (void**) &x);
    ns_fill(g, x, 0, 17408 * 8 * 4);
    if (check) {
        std::vector<float> hx(17408 * 8);
        unsigned long long r = 0x2545f4914f6cdd1dull;
        for (auto& v : hx) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; v = (float) ((int) (r % 2001) - 1000) / 500.0f; }
        ns_copy_to(g, x, hx.data(), hx.size() * 4);
    }
    void* q; ns_alloc(g, ns_q8_1_bytes(17408, 8), &q);
    float* y; ns_alloc(g, 17408 * 8 * 4, (void**) &y);
    if (mode.find("peaks") != std::string::npos) {   // the card's practical ceilings: a device copy, oneMKL's half GEMM
        const size_t n = size_t(1) << 30;
        void *a, *b; ns_alloc(g, n, &a); ns_alloc(g, n, &b);
        std::vector<unsigned char> h(n);
        unsigned long long r = 0x9e3779b97f4a7c15ull;
        for (auto& v : h) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; v = (unsigned char) r; }
        ns_copy_to(g, a, h.data(), n);
        ns_sync(g);
        for (int i = 0; i < 2; ++i) ns_copy_dev(g, b, a, n);
        ns_sync(g);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) ns_copy_dev(g, b, a, n);
        ns_sync(g);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 10;
        printf("device copy 1 GiB: %.0f GB/s (read + write)\n", 2.0 * n / s / 1e9);
        const int64_t shapes[][3] = {{4096, 4096, 4096}, {512, 17408, 5120}, {512, 5120, 17408}, {512, 10240, 5120}};
        for (auto& sh : shapes) {
            const int64_t T = sh[0], N = sh[1], K = sh[2];
            ns_fill(g, a, 0x11, T * K * 2); ns_fill(g, b, 0x11, N * K * 2);
            float* y; ns_alloc(g, T * N * 4, (void**) &y);
            for (int i = 0; i < 2; ++i) ns_gemm_f16(g, T, N, K, (uint16_t*) a, K, (uint16_t*) b, y, N, 0);
            ns_sync(g);
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i) ns_gemm_f16(g, T, N, K, (uint16_t*) a, K, (uint16_t*) b, y, N, 0);
            ns_sync(g);
            const double sg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() / 10;
            printf("gemm f16 %lldx%lldx%lld: %.1f TFLOP/s (%.2f ms)\n", (long long) T, (long long) N, (long long) K, 2.0 * T * N * K / sg / 1e12, sg * 1e3);
            ns_free(g, y);
        }
        ns_free(g, a); ns_free(g, b);
        return 0;
    }
    {   // the cost of a launch: 2,000 tiny products back to back (Q8_0, 64 x 256)
        void* w; ns_alloc(g, 64 * 8 * 34, &w);
        ns_fill(g, w, 0, 64 * 8 * 34);
        ns_quantize_q8_1(g, x, q, 256, 1);
        for (int i = 0; i < 100; ++i) ns_mmvq(g, 8, w, q, y, 256, 64, 1);
        ns_sync(g);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 2000; ++i) ns_mmvq(g, 8, w, q, y, 256, 64, 1);
        ns_sync(g);
        printf("a launch: %.2f us\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 2000 * 1e6);
        ns_free(g, w);
    }
    printf("GPU %d, %d column(s): GB/s of weights (us per matrix)\n%-8s", dev, ncols, "type");
    for (auto& s : shapes) printf("   %5lldx%-5lld     ", (long long) s[0], (long long) s[1]);
    printf("\n");
    for (const T& t : types) {
        if (!ns_mmvq_supported(t.id)) { printf("%-8s not supported\n", t.name); continue; }
        printf("%-8s", t.name);
        for (auto& s : shapes) {
            const int64_t n_out = s[0], n_in = s[1];
            const size_t wb = (size_t) n_out * n_in / t.elems * t.bytes;
            void* w; ns_alloc(g, wb * NE, &w);
            std::vector<unsigned char> h(wb);
            unsigned long long r = 0x9e3779b97f4a7c15ull;
            for (auto& b : h) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; b = (unsigned char) (r & 0x3f); }
            for (int e = 0; e < NE; ++e) ns_copy_to(g, (char*) w + wb * e, h.data(), wb);
            ns_sync(g);
            ns_quantize_q8_1(g, x, q, n_in, ncols);
            auto mm = own && ns_q35_mmvq_supported(t.id, n_in) ? ns_q35_mmvq : ns_mmvq;
            if (check) {
                mm(g, t.id, w, q, y, n_in, n_out, ncols);
                std::vector<float> hy((size_t) n_out * ncols);
                ns_copy_from(g, hy.data(), y, hy.size() * 4);
                ns_sync(g);
                double sum = 0, abs = 0;
                for (float v : hy) { sum += v; abs += v < 0 ? -v : v; }
                printf("  %.6g %.6g %.6g %.6g |", sum, abs, hy[0], hy[hy.size() - 1]);
                fflush(stdout);
                ns_free(g, w);
                continue;
            }
            for (int i = 0; i < 4; ++i) mm(g, t.id, (char*) w + wb * (i % NE), q, y, n_in, n_out, ncols);
            ns_sync(g);
            const int R = 100;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < R; ++i) mm(g, t.id, (char*) w + wb * (i % NE), q, y, n_in, n_out, ncols);
            ns_sync(g);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / R;
            printf("   %5.0f (%6.1f)    ", wb / sec / 1e9, sec * 1e6);
            fflush(stdout);
            ns_free(g, w);
        }
        printf("\n");
    }
    return 0;
}
