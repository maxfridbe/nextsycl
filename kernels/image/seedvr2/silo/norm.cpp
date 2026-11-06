// norm.cpp: the VAE's group norm (+ SiLU) for channels in eights. The engine's own (svr.cpp) applies the statistics
// with 64-bit divisions for every 8 channels (the frame and the channel from a flat index, the group from the channel)
// and reads the mean, the scale, the weight and the bias apart; here the statistics pass folds them into one scale and
// shift a (frame, channel) and the apply pass takes its frame, voxel and channels from a 3-D range: y = x a + b.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

#include <map>
#include <mutex>
#include <string>

namespace {
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
inline float silu(float v) { return v / (1.0f + sycl::exp(-v)); }
using v8 = sycl::vec<uint16_t, 8>;
inline float bf2f(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }
inline uint16_t f2bf(float f) { return sycl::bit_cast<uint16_t>((sycl::ext::oneapi::bfloat16) f); }

// a float32 scratch buffer per GPU, grown to the largest asked (the partial sums, the scales and shifts)
std::mutex mu;
struct Scratch {
    float* p = nullptr;
    size_t cap = 0;
};
std::map<ns_gpu*, Scratch> scratch;

float* grow(ns_gpu* g, size_t n) {
    std::lock_guard<std::mutex> l(mu);
    Scratch& s = scratch[g];
    if (n > s.cap) {
        g->q.wait();                               // nothing queued may still use the old one
        if (s.p) sycl::free(s.p, g->q);
        s.p = sycl::malloc_device<float>(n, g->q);
        s.cap = s.p ? n : 0;
    }
    return s.p;
}
}  // namespace

extern "C" {

int ns_image_seedvr2_silo_group_norm(ns_gpu* g, const void* x, int64_t F, int64_t N, int64_t C, int64_t G, const float* weight, const float* bias,
                                     float eps, int silu_, void* out) {
    NS_TRY
    if (F <= 0 || N <= 0 || C <= 0) return 0;
    if (G <= 0 || C % G || C % 8) return ns_fail("ns_image_seedvr2_silo_group_norm: channels in eights that split into the groups");
    constexpr int64_t P = 4096;                    // parts a frame: short loops, many work-items
    const int64_t C8 = C / 8, cg = C / G;
    float* part = grow(g, sz(F * P * C * 2 + F * C * 2));
    if (!part) return ns_fail("ns_image_seedvr2_silo_group_norm: no room for the partial sums");
    float* ab = part + F * P * C * 2;
    const v8* xv = (const v8*) x;
    // each part's sums of its voxels' 8 channels, in registers
    g->q.parallel_for(sycl::range<3>(sz(F), sz(P), sz(C8)), [=](sycl::id<3> id) {
        const int64_t f = id[0], p = id[1], c8 = id[2];
        float s[8] = {0, 0, 0, 0, 0, 0, 0, 0}, s2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const v8* base = xv + (size_t) f * N * C8 + c8;
        for (int64_t n = p; n < N; n += P) {
            const v8 v = base[(size_t) n * C8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float a = bf2f(v[i]);
                s[i] += a;
                s2[i] += a * a;
            }
        }
        float* o = part + ((size_t) (f * P + p) * C + c8 * 8) * 2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[2 * i] = s[i];
            o[2 * i + 1] = s2[i];
        }
    });
    // a work-group of 256 a (frame, group): the parts reduced, then each channel's scale and shift
    constexpr size_t RG = 256;
    g->q.parallel_for(sycl::nd_range<1>(sz(F * G) * RG, RG), [=](sycl::nd_item<1> it) {
        const int64_t fg = (int64_t) it.get_group(0), f = fg / G, gr = fg % G;
        const int64_t l = (int64_t) it.get_local_id(0);
        float a = 0.0f, a2 = 0.0f;
        for (int64_t p = l; p < P; p += RG)
            for (int64_t c = gr * cg; c < (gr + 1) * cg; ++c) {
                a += part[((size_t) (f * P + p) * C + c) * 2];
                a2 += part[((size_t) (f * P + p) * C + c) * 2 + 1];
            }
        const double sa = sycl::reduce_over_group(it.get_group(), (double) a, sycl::plus<double>());
        const double sa2 = sycl::reduce_over_group(it.get_group(), (double) a2, sycl::plus<double>());
        const double n = (double) N * (double) cg, mean = sa / n;
        const double var = sycl::fmax(sa2 / n - mean * mean, 0.0);
        const float rstd = (float) (1.0 / sycl::sqrt(var + (double) eps));
        for (int64_t c = gr * cg + l; c < (gr + 1) * cg; c += (int64_t) RG) {
            const float sc = rstd * weight[c];
            ab[(f * C + c) * 2] = sc;
            ab[(f * C + c) * 2 + 1] = bias[c] - (float) mean * sc;
        }
    });
    // y = x a + b (+ SiLU): (frame, voxel, 8 channels) a work-item
    v8* ov = (v8*) out;
    g->q.parallel_for(sycl::range<3>(sz(F), sz(N), sz(C8)), [=](sycl::id<3> id) {
        const int64_t f = id[0], c8 = id[2];
        const size_t i = ((size_t) f * N + id[1]) * C8 + c8;
        const float* s = ab + (f * C + c8 * 8) * 2;
        const v8 v = xv[i];
        v8 r;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float y = bf2f(v[k]) * s[2 * k] + s[2 * k + 1];
            r[k] = f2bf(silu_ ? silu(y) : y);
        }
        ov[i] = r;
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
