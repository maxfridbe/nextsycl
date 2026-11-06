// move.cpp: the upscalers' data movement in 16-byte pieces (8 halves a work-item) and without divisions by the
// channels: a channel slice copied into a wider buffer (a dense block's input into its concatenation) and the nearest
// upsampling by a whole factor (RRDB's two 2x steps). The engine's own kernels (up.cpp) move a half at a time and take
// every index apart with 64-bit divisions: 8-20 GB/s on the B70 for these.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

namespace {
using v8 = sycl::vec<uint16_t, 8>;
inline size_t sz(int64_t n) { return (size_t) (n > 0 ? n : 0); }
}  // namespace

extern "C" {

int ns_image_up_silo_copy_channels(ns_gpu* g, const void* srcv, int64_t P, int64_t Cs, void* dstv, int64_t Cd, int64_t off) {
    NS_TRY
    if (Cs % 8 || Cd % 8 || off % 8 || off + Cs > Cd) return ns_fail("ns_image_up_silo_copy_channels: channels and offset in eights, inside the destination");
    const v8* src = (const v8*) srcv;
    v8* dst = (v8*) dstv;
    const int64_t s8 = Cs / 8, d8 = Cd / 8, o8 = off / 8;
    if (P > 0) g->q.parallel_for(sycl::range<2>(sz(P), sz(s8)), [=](sycl::id<2> id) {
        dst[id[0] * d8 + o8 + id[1]] = src[id[0] * s8 + id[1]];
    });
    return 0;
    NS_CATCH
}

int ns_image_up_silo_upsample(ns_gpu* g, const void* xv, int64_t H, int64_t W, int64_t C, int64_t r, void* out) {
    NS_TRY
    if (r < 1) return ns_fail("ns_image_up_silo_upsample: a factor of at least 1");
    if (C % 8) {
        // few channels (the picture itself, 3): a pixel a work-item
        const uint16_t* x = (const uint16_t*) xv;
        uint16_t* o = (uint16_t*) out;
        const int64_t Wo = W * r;
        if (H > 0 && W > 0) g->q.parallel_for(sycl::range<2>(sz(H * r), sz(Wo)), [=](sycl::id<2> id) {
            const int64_t oy = id[0], ox = id[1];
            const uint16_t* s = x + ((oy / r) * W + ox / r) * C;
            uint16_t* d = o + (oy * Wo + ox) * C;
            for (int64_t c = 0; c < C; ++c) d[c] = s[c];
        });
        return 0;
    }
    const v8* x = (const v8*) xv;
    v8* o = (v8*) out;
    const int64_t c8 = C / 8, Wo = W * r;
    if (H > 0 && W > 0) g->q.parallel_for(sycl::range<3>(sz(H * r), sz(Wo), sz(c8)), [=](sycl::id<3> id) {
        const int64_t oy = id[0], ox = id[1], c = id[2];
        o[(oy * Wo + ox) * c8 + c] = x[((oy / r) * W + ox / r) * c8 + c];
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
