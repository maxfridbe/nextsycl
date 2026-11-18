// tla/gather.cpp: the H3 video silo's gather for MPEG-Diffusion's two-track attention (silo.h; crates/h3/src/mpeg.rs).
// One work-item copies 8 values (16 bytes) of one row; the rows' order is the index's. A copy, so exact: the bits move
// unchanged. Plain SYCL, but built with the silo's sycl-tla sources (kernels/build.sh: a silo of tla/ sources alone is
// one library, dist/silo/libnextsycl-h3.so) so the fast decoder's kernels and this one share a library and silo.h:
// it takes the GPU's SYCL queue (nextsycl-core's Gpu::queue, the queue the engine's device is created on) and says why
// it failed in ns_video_h3_silo_error().
#include <cstdint>
#include <exception>
#include <string>

#include <sycl/sycl.hpp>
#include "../silo.h"

namespace ns_video_h3_silo {
int fail(const std::string& m);   // tla/fastvae.cpp
}

extern "C" {

int ns_video_h3_silo_gather_supported(int64_t width, int64_t src_stride, int64_t dst_stride) {
    return width > 0 && src_stride >= width && dst_stride >= width && width % 8 == 0 && src_stride % 8 == 0 && dst_stride % 8 == 0;
}

int ns_video_h3_silo_gather(void* queue, void* dst, int64_t dst_stride, const void* src, int64_t src_stride, const int32_t* idx, int64_t rows,
                            int64_t width, int64_t src_rows) try {
    using ns_video_h3_silo::fail;
    if (!queue || !dst || !src || !idx) return fail("ns_video_h3_silo_gather: a null pointer");
    if (!ns_video_h3_silo_gather_supported(width, src_stride, dst_stride))
        return fail("ns_video_h3_silo_gather: width and strides must be positive multiples of 8 values");
    if (((uintptr_t) dst | (uintptr_t) src) % 16 != 0) return fail("ns_video_h3_silo_gather: buffers must be 16-byte aligned");
    if (rows <= 0) return rows == 0 ? 0 : fail("ns_video_h3_silo_gather: bad rows");
    if (src_rows <= 0 || rows > INT32_MAX || width / 8 > INT32_MAX) return fail("ns_video_h3_silo_gather: bad sizes");
    auto* qu = static_cast<sycl::queue*>(queue);
    using V = sycl::vec<uint32_t, 4>;   // 8 16-bit values
    const int64_t vw = width / 8, ss = src_stride / 8, ds = dst_stride / 8;
    const V* s = static_cast<const V*>(src);
    V* d = static_cast<V*>(dst);
    // a 2D range: rows x vectors of a row, the inner dimension rounded up to the work-group
    const size_t wg = 64;
    const size_t inner = (size_t) ((vw + wg - 1) / wg * wg);
    qu->parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) rows, inner), sycl::range<2>(1, wg)), [=](sycl::nd_item<2> it) {
        const int64_t r = (int64_t) it.get_global_id(0), c = (int64_t) it.get_global_id(1);
        if (c >= vw) return;
        const int64_t from = idx[r];
        if (from < 0 || from >= src_rows) return;
        d[r * ds + c] = s[from * ss + c];
    });
    return 0;
} catch (const std::exception& e) {
    return ns_video_h3_silo::fail(std::string("ns_video_h3_silo_gather: ") + e.what());
}

}  // extern "C"
