// ns_q8_1.hpp: a Q8_1 activation block's values, scale and sum (GGUF's Q8_1: 32 int8 values, then the scale
// d = amax / 127 and the sum of the 32 raw values, both fp16), with the scale and sum held finite.
//
// A sum past 65504 (one massive-activation dimension among 32 values is enough) would round to inf in fp16, and so
// would d past amax = 8.3M; the dot products then read inf * 0 = NaN and the residual turns to NaN. Clamped to the
// largest finite half, such a block gives an approximate product instead, its values held to the int8 range. A block
// that was finite before is stored bit for bit the same; a NaN stays NaN (the comparisons are false for it).
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace ns {

constexpr float kHalfMax = 65504.0f;

inline float q8_1_finite(float v) { return sycl::fabs(v) > kHalfMax ? sycl::copysign(kHalfMax, v) : v; }

// x's int8 value for the (finite) scale d; amax = 0 is an all-zero block
inline int8_t q8_1_quant(float x, float d, float amax) {
    if (amax == 0.0f) return 0;
    const float q = sycl::round(x / d);
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

// the block's {d, sum} as fp16 (rounded to nearest even)
inline sycl::half2 q8_1_ds(float d, float sum) {
    return sycl::half2((sycl::half) q8_1_finite(d), (sycl::half) q8_1_finite(sum));
}

}  // namespace ns
