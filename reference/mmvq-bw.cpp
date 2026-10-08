// Build in the oneAPI image: icpx -O2 -Ikernels/ns reference/mmvq-bw.cpp -o mmvq_bw dist/libnextsycl.so -Wl,-rpath,dist; run with ONEAPI_DEVICE_SELECTOR=level_zero:*
// The decode-width Q8_0 product (Strata's wide32 kernel, through libnextsycl's ns_mmvq) on one GPU: GB/s of weights
// read, an 8192 x 4096 matrix (35.7 MB, KDA's q / k / v), 1-3 columns, 16 matrices rotated (cold, past the L2).
#include "ns.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    ns_gpu* g;
    if (ns_gpu_open(dev, &g)) { printf("open: %s\n", ns_last_error()); return 1; }
    const int64_t n_in = 4096, n_out = 8192, NE = 16;
    const size_t wb = (size_t) n_out * n_in / 32 * 34;
    void* w; ns_alloc(g, wb * NE, &w);
    {   // random bytes (a constant fill is compressed by the GPU's memory: an inflated bandwidth); valid fp16 scales
        std::vector<unsigned char> h(wb);
        unsigned long long r = 0x9e3779b97f4a7c15ull;
        for (auto& b : h) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; b = (unsigned char) r; }
        for (size_t k = 0; k + 34 <= wb; k += 34) { h[k] = 0x00; h[k + 1] = 0x20; }
        for (int e = 0; e < NE; ++e) ns_copy_to(g, (char*) w + wb * e, h.data(), wb);
        ns_sync(g);   // the copies are queued: h must outlive them
    }
    float* x; ns_alloc(g, n_in * 8 * 4, (void**) &x);
    ns_fill(g, x, 0, n_in * 8 * 4);
    void* q; ns_alloc(g, ns_q8_1_bytes(n_in, 8), &q);
    float* y; ns_alloc(g, n_out * 8 * 4, (void**) &y);
    for (int nc : {1, 2, 3}) {
        ns_quantize_q8_1(g, x, q, n_in, nc);
        for (int i = 0; i < 5; ++i) ns_mmvq(g, 8, (char*) w + wb * (i % NE), q, y, n_in, n_out, nc);
        ns_sync(g);
        const int R = 200;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < R; ++i) ns_mmvq(g, 8, (char*) w + wb * (i % NE), q, y, n_in, n_out, nc);
        ns_sync(g);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / R;
        printf("GPU %d, Q8_0 8192 x 4096, %d col(s): %6.1f us, %5.0f GB/s\n", dev, nc, s * 1e6, wb / s / 1e9);
    }
    // the ceiling: a plain read of the same bytes (a copy into one buffer)
    void* dst; ns_alloc(g, wb, &dst);
    for (int i = 0; i < 5; ++i) ns_copy_dev(g, dst, (char*) w + wb * (i % NE), wb);
    ns_sync(g);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i) ns_copy_dev(g, dst, (char*) w + wb * (i % NE), wb);
    ns_sync(g);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 100;
    printf("GPU %d, a device copy of the same 35.7 MB: %6.1f us, %5.0f GB/s read + %5.0f written\n", dev, s * 1e6, wb / s / 1e9, wb / s / 1e9);
    return 0;
}
