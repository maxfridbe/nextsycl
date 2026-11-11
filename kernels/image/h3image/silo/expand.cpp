// expand.cpp: the MiniMax-H3-Image silo's k-quant expansion (silo.h): a GGUF block matrix into the bfloat16 buffer the
// denoiser's product reads, once a product - 200 matrices a step, 19.25 G values. The shared nsd_dequant gives a
// work-item four values and reads its block's header and bytes one at a time; here a work-item expands 16 consecutive
// values from one 16-byte load of the block's packed values and writes them as two 16-byte stores, a sub-group of 16
// covering a 256-value block. And Q3_K is read as it is stored (110 bytes a block, 3.44 bits a value): the engine no
// longer has to hold Q3_K_M's matrices as Q6_K (210 bytes a block) on the card.
//
// The arithmetic is the shared kernels' (and ggml's) operation for operation: Q4_K / Q5_K d * s and dmin * m, then
// (d s) q - dmin m; Q8_0 d q; Q3_K (d s) q (Q6_K's order for the Q6_K blocks the engine used to make of it). Every
// product is exact in float32 (at most 22 significant bits), so the one rounding is the subtraction's and then
// float32 -> bfloat16 (round to nearest even, inf / NaN truncated, as nsd.cpp's f32_to_bf16): the same bits.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

#include <cstring>
#include <string>

namespace {
constexpr int SG = 16;   // lanes a sub-group: one 256-value block, 16 values a lane

inline uint16_t to_bf16(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) (u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

inline float half_at(const uint8_t* p) {
    const uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
    return (float) sycl::bit_cast<sycl::half>(h);
}

// 16 bfloat16 values to out[0..16): two 16-byte stores (out is 32-byte aligned: 16 values a lane from a 16-aligned
// buffer)
inline void put16(uint16_t* out, const float (&v)[16]) {
    sycl::vec<uint16_t, 8> a, b;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        a[t] = to_bf16(v[t]);
        b[t] = to_bf16(v[t + 8]);
    }
    *reinterpret_cast<sycl::vec<uint16_t, 8>*>(out) = a;
    *reinterpret_cast<sycl::vec<uint16_t, 8>*>(out + 8) = b;
}

// Q4_K / Q5_K sub-block j's 6-bit scale and min (ggml's get_scale_min_k4)
inline void scale_min(const uint8_t* sc, int j, int& s, int& m) {
    if (j < 4) {
        s = sc[j] & 63;
        m = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

inline sycl::vec<uint8_t, 16> load16(const uint8_t* p) {
    return *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(p);
}
}  // namespace

extern "C" {

int ns_image_h3i_silo_expand_supported(int qtype) {
    return qtype == 8 || qtype == 11 || qtype == 12 || qtype == 13;
}

int ns_image_h3i_silo_expand(ns_gpu* g, const void* src, int qtype, int64_t n, void* out) try {
    if (!ns_image_h3i_silo_expand_supported(qtype)) return ns_fail("h3image silo expand: Q8_0 (8), Q3_K (11), Q4_K (12) or Q5_K (13)");
    const int64_t blk = qtype == 8 ? 32 : 256;
    if (n <= 0 || n % blk != 0) return ns_fail("h3image silo expand: a whole number of blocks");
    if ((uintptr_t) out % 32 != 0 || (uintptr_t) src % 16 != 0) return ns_fail("h3image silo expand: 32-byte aligned output, 16-byte aligned blocks");
    sycl::queue& q = g->q;
    const uint8_t* b = static_cast<const uint8_t*>(src);
    uint16_t* o = static_cast<uint16_t*>(out);
    const size_t lanes = (size_t) (n / 16);
    const size_t wg = 256;
    const sycl::nd_range<1> range(sycl::range<1>((lanes + wg - 1) / wg * wg), sycl::range<1>(wg));
    if (qtype == 12) {   // Q4_K: d, dmin (f16), 12 bytes of scales / mins, 128 bytes of nibbles (144)
        q.parallel_for(range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t lane = it.get_global_id(0);
            if (lane >= lanes) return;
            const size_t bi = lane / 16;
            const int l = (int) (lane % 16), j = l / 2, l0 = (l % 2) * 16;
            const uint8_t* x = b + bi * 144;
            int s, m;
            scale_min(x + 4, j, s, m);
            const float d1 = half_at(x) * (float) s, m1 = half_at(x + 2) * (float) m;
            const sycl::vec<uint8_t, 16> qv = load16(x + 16 + (j / 2) * 32 + l0);
            const bool high = j % 2 == 1;
            float v[16];
#pragma unroll
            for (int t = 0; t < 16; ++t) v[t] = d1 * (float) (high ? (qv[t] >> 4) : (qv[t] & 0xF)) - m1;
            put16(o + lane * 16, v);
        });
    } else if (qtype == 13) {   // Q5_K: Q4_K's header, 32 bytes of fifth bits, 128 of nibbles (176)
        q.parallel_for(range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t lane = it.get_global_id(0);
            if (lane >= lanes) return;
            const size_t bi = lane / 16;
            const int l = (int) (lane % 16), j = l / 2, l0 = (l % 2) * 16;
            const uint8_t* x = b + bi * 176;
            int s, m;
            scale_min(x + 4, j, s, m);
            const float d1 = half_at(x) * (float) s, m1 = half_at(x + 2) * (float) m;
            const sycl::vec<uint8_t, 16> hv = load16(x + 16 + l0);
            const sycl::vec<uint8_t, 16> qv = load16(x + 48 + (j / 2) * 32 + l0);
            const bool high = j % 2 == 1;
            float v[16];
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                const int val = (high ? (qv[t] >> 4) : (qv[t] & 0xF)) + (((hv[t] >> j) & 1) << 4);
                v[t] = d1 * (float) val - m1;
            }
            put16(o + lane * 16, v);
        });
    } else if (qtype == 11) {   // Q3_K: hmask[32], qs[64], 12 bytes of 6-bit scales, d (f16): 110 bytes, not aligned
        q.parallel_for(range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t lane = it.get_global_id(0);
            if (lane >= lanes) return;
            const size_t bi = lane / 16;
            const int l = (int) (lane % 16);
            const uint8_t* x = b + bi * 110;
            // values 16 l .. 16 l + 15: half c of 128, shift pair j, positions i0 .. i0 + 15 of 32; scale l
            const int c = l / 8, j = (l % 8) / 2, i0 = (l % 2) * 16;
            const uint8_t* sc = x + 96;
            auto word = [&](int i) { return (uint32_t) sc[4 * i] | ((uint32_t) sc[4 * i + 1] << 8) | ((uint32_t) sc[4 * i + 2] << 16) | ((uint32_t) sc[4 * i + 3] << 24); };
            const uint32_t a0 = word(0), a1 = word(1), tmp = word(2);
            constexpr uint32_t K1 = 0x03030303u, K2 = 0x0f0f0f0fu;
            const int wi = l / 4;
            const uint32_t aux = wi == 0 ? (a0 & K2) | ((tmp & K1) << 4)
                               : wi == 1 ? (a1 & K2) | (((tmp >> 2) & K1) << 4)
                               : wi == 2 ? ((a0 >> 4) & K2) | (((tmp >> 4) & K1) << 4)
                                         : ((a1 >> 4) & K2) | (((tmp >> 6) & K1) << 4);
            const int s = (int) ((aux >> (8 * (l % 4))) & 0xFF) - 32;
            const float ds = half_at(x + 108) * (float) s;
            const uint8_t bit = (uint8_t) (1u << (4 * c + j));
            float v[16];
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                const int i = i0 + t;
                const int low = (x[32 + 32 * c + i] >> (2 * j)) & 3;
                const int val = low - ((x[i] & bit) ? 0 : 4);
                v[t] = ds * (float) val;
            }
            put16(o + lane * 16, v);
        });
    } else {   // Q8_0: d (f16), 32 int8 (34 bytes); a lane half a block
        q.parallel_for(range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t lane = it.get_global_id(0);
            if (lane >= lanes) return;
            const uint8_t* x = b + (lane / 2) * 34;
            const float d = half_at(x);
            const uint8_t* qs = x + 2 + (lane % 2) * 16;
            float v[16];
#pragma unroll
            for (int t = 0; t < 16; ++t) v[t] = d * (float) (int8_t) qs[t];
            put16(o + lane * 16, v);
        });
    }
    return 0;
} catch (const std::exception& e) {
    return ns_fail(std::string("h3image silo expand: ") + e.what());
}

}  // extern "C"
