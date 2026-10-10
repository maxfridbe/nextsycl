// common.hpp: what the qwen35 silo's kernels share - loads of 2-byte aligned blocks, the K-quants' 6-bit scales
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace q35silo {

// 16 bytes at a 2-byte aligned address from aligned 16-byte loads (Strata's load16_a2): the second chunk only when
// the address is unaligned, so it never reads past an allocation's last chunk
inline sycl::int4 load16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;
    const bool half = (a & 2) != 0;
    const sycl::int4 lo = q[0], hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t) lo.x(), (uint32_t) lo.y(), (uint32_t) lo.z(), (uint32_t) lo.w(),
                           (uint32_t) hi.x(), (uint32_t) hi.y(), (uint32_t) hi.z(), (uint32_t) hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int) o[0], (int) o[1], (int) o[2], (int) o[3]);
}

// get_scale_min_k4 from the 12 scale bytes held in three words
inline int k4_byte(const uint32_t sc[3], int i) { return int((sc[i >> 2] >> (8 * (i & 3))) & 0xff); }
inline void k4_scale_min(const uint32_t sc[3], int j, float& s, float& m) {
    if (j < 4) { s = (float) (k4_byte(sc, j) & 63); m = (float) (k4_byte(sc, j + 4) & 63); }
    else {
        s = (float) ((k4_byte(sc, j + 4) & 0xF) | ((k4_byte(sc, j - 4) >> 6) << 4));
        m = (float) ((k4_byte(sc, j + 4) >> 4) | ((k4_byte(sc, j) >> 6) << 4));
    }
}
}  // namespace q35silo
