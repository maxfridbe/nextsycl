// vq.cpp: the routed experts' vector quantization (ds41.h) and the fp8_32x32 matrices.
//
// A DQV3 payload (YoungAi's vq_fmt.h, ds4-dfm-rs's vq.rs): a 32-byte header, an f16 gain a row, the row's indices as a
// little-endian 12-bit stream (cols / 8 of them, 1.5 bytes each), and at 13 bits a plane of the 13th bits (a byte a
// row's 8 indices). value[r][c] = codebook[idx(r, c / 8)][c % 8] x gain[r]; the codebook is the layer's, E4M3 [2^nbit][8].
// Eight indices are 12 aligned bytes (three words) and 64 columns: the unit every kernel here reads.
//
// An E4M3 byte as an fp16's bits - sign to bit 15, exponent and mantissa shifted 7 up - is the value x 2^-8, the
// subnormals too (fp16's exponent bias 15 against E4M3's 7): one shift, one convert; the 2^8 goes into the gain.
#include "ns.h"
#include "ns_internal.hpp"
#include "ds41.h"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

inline float e4m3_scaled(uint32_t b) {   // the value x 2^-8
    const uint16_t bits = (uint16_t) (((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return (float) sycl::bit_cast<sycl::half>(bits);
}

// the 8 indices of a chunk: 12 bytes at s (4-aligned), the plane's byte pb (13-bit layers; 0 otherwise)
inline void chunk_idx(const uint32_t* s, uint32_t pb, uint32_t idx[8]) {
    const uint64_t lo = (uint64_t) s[0] | ((uint64_t) s[1] << 32);
    const uint32_t hi = s[2];
#pragma unroll
    for (int i = 0; i < 5; ++i) idx[i] = (uint32_t) (lo >> (12 * i)) & 0xFFFu;
    idx[5] = (uint32_t) ((lo >> 60) | ((uint64_t) hi << 4)) & 0xFFFu;
    idx[6] = (hi >> 8) & 0xFFFu;
    idx[7] = (hi >> 20) & 0xFFFu;
#pragma unroll
    for (int i = 0; i < 8; ++i) idx[i] |= ((pb >> i) & 1u) << 12;
}

struct VqMat {
    const sycl::half* gains;
    const uint8_t* stream;
    const uint8_t* plane;
    int64_t mrow, prow;
};
inline VqMat vq_open(const uint8_t* p, int64_t rows, int64_t cols, int64_t nbit) {
    VqMat m;
    m.gains = reinterpret_cast<const sycl::half*>(p + 32);
    m.stream = p + 32 + rows * 2;
    m.mrow = cols / 8 * 12 / 8;
    m.prow = cols / 64;
    m.plane = nbit > 12 ? m.stream + rows * m.mrow : nullptr;
    return m;
}

}  // namespace

namespace {
// an index's 8 weights (x 2^-8): CB 0 from the E4M3 bytes a byte at a time, 1 from the bytes two at a time (a word's
// bytes 0 and 2, then 1 and 3, as fp16 pairs), 2 from the fp16 codebook (cbh)
template <int CB>
inline void vq_weights(const uint8_t* cb, const sycl::half* cbh, uint32_t ix, float wv[8]) {
    if constexpr (CB == 2) {
        const sycl::vec<sycl::half, 8> h = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(cbh + (size_t) ix * 8);
#pragma unroll
        for (int d = 0; d < 8; ++d) wv[d] = (float) h[d];
    } else if constexpr (CB == 1) {
        const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(cb + (size_t) ix * 8);
#pragma unroll
        for (int hw = 0; hw < 2; ++hw) {
            const uint32_t u = hw ? w.y() : w.x();
            const uint32_t e = ((u & 0x00800080u) << 8) | ((u & 0x007F007Fu) << 7);
            const uint32_t o = (u & 0x80008000u) | ((u & 0x7F007F00u) >> 1);
            const sycl::vec<sycl::half, 2> he = sycl::bit_cast<sycl::vec<sycl::half, 2>>(e), ho = sycl::bit_cast<sycl::vec<sycl::half, 2>>(o);
            wv[4 * hw + 0] = (float) he[0];
            wv[4 * hw + 2] = (float) he[1];
            wv[4 * hw + 1] = (float) ho[0];
            wv[4 * hw + 3] = (float) ho[1];
        }
    } else {
        const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(cb + (size_t) ix * 8);
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            wv[d] = e4m3_scaled((w.x() >> (8 * d)) & 0xFF);
            wv[4 + d] = e4m3_scaled((w.y() >> (8 * d)) & 0xFF);
        }
    }
}

// vq_mmv's variants 1, 2 and 4: a lane an output row (WG rows a work-group), the activations read alike by all lanes, an
// index's weights from the E4M3 bytes (8 bytes gathered; 4 converts them two at a time) or the fp16 codebook (2: 16
// bytes). 6 experts, one token, gate | up / down in us - B70 12-bit: variant 0 239 / 158, 1 216 / 121, 2 199 / 105,
// 3 230 / 149, 4 191 / 106; 13-bit: 374 / 220, 266 / 138, 264 / 132, 226 / 153, 240 / 122; B65 12-bit: 440 / 279,
// 507 / 245, 452 / 209, 418 / 264, 433 / 210. Which wins depends on the GPU and the shape: the engine times them all
// at load (vq_pick).
template <bool PLANE, int CB>
void vq_mmv_rows(sycl::queue& q, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const uint8_t* cb, const sycl::half* cbh, const float* x, float* y,
                 int64_t rows, int64_t cols, int WG) {
    const int64_t rgs = (rows + WG - 1) / WG, chunks = cols / 64;
    q.parallel_for(sycl::nd_range<1>(entries * rgs * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t gi = it.get_group(0), e = gi / rgs, r = (gi % rgs) * WG + it.get_local_id(0);
        if (r >= rows) return;
        const int32_t* me = meta + e * 10;
        const int nc = me[0];
        const VqMat m = vq_open(reinterpret_cast<const uint8_t*>(ptrs[e]), rows, cols, PLANE ? 13 : 12);
        const uint32_t* srow = reinterpret_cast<const uint32_t*>(m.stream + r * m.mrow);
        const uint8_t* prow = PLANE ? m.plane + r * m.prow : nullptr;
        float acc[8] = {};
        for (int64_t k = 0; k < chunks; ++k) {
            uint32_t idx[8];
            chunk_idx(srow + 3 * k, PLANE ? prow[k] : 0u, idx);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float wv[8];
                vq_weights<CB>(cb, cbh, idx[i], wv);
                for (int c = 0; c < nc; ++c) {
                    const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(x + (int64_t) me[2 + c] * cols + k * 64 + i * 8);
                    const sycl::float4 a = xv[0], b = xv[1];
                    acc[c] += wv[0] * a.x() + wv[1] * a.y() + wv[2] * a.z() + wv[3] * a.w() + wv[4] * b.x() + wv[5] * b.y() + wv[6] * b.z() + wv[7] * b.w();
                }
            }
        }
        const float gn = (float) m.gains[r] * 256.f;
        for (int c = 0; c < nc; ++c) y[(int64_t) (me[1] + c) * rows + r] = acc[c] * gn;
    });
}
}  // namespace

extern "C" {

int ns_ds41_vq_supported(int64_t rows, int64_t cols, int64_t nbit) {
    return rows > 0 && cols % 64 == 0 && (nbit == 12 || nbit == 13);
}

int ns_ds41_vq_codebook_f16(ns_gpu* g, const uint8_t* cb, int64_t n, void* outv) {
    NS_TRY
    sycl::half* out = static_cast<sycl::half*>(outv);
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] = (sycl::half) e4m3_scaled(cb[i]); });
    return 0;
    NS_CATCH
}

int ns_ds41_vq_codebook_f32(ns_gpu* g, const uint8_t* cb, int64_t n, float* out) {
    NS_TRY
    g->q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] = e4m3_scaled(cb[i]); });
    return 0;
    NS_CATCH
}

int ns_ds41_vq_dequant(ns_gpu* g, const void* payload, const uint8_t* cb, int64_t rows, int64_t cols, int64_t nbit, void* out, int f16) {
    NS_TRY
    if (!ns_ds41_vq_supported(rows, cols, nbit)) return ns_fail("ns_ds41_vq_dequant: cols a multiple of 64, 12 or 13 bits");
    const VqMat m = vq_open(static_cast<const uint8_t*>(payload), rows, cols, nbit);
    const int64_t chunks = cols / 64;
    float* of = static_cast<float*>(out);
    sycl::half* oh = static_cast<sycl::half*>(out);
    g->q.parallel_for(sycl::range<2>(rows, chunks), [=](sycl::id<2> id) {
        const int64_t r = id[0], k = id[1];
        uint32_t idx[8];
        chunk_idx(reinterpret_cast<const uint32_t*>(m.stream + r * m.mrow + 12 * k), m.plane ? m.plane[r * m.prow + k] : 0u, idx);
        const float gn = (float) m.gains[r] * 256.f;
        for (int i = 0; i < 8; ++i) {
            const uint8_t* c = cb + (size_t) idx[i] * 8;
            for (int d = 0; d < 8; ++d) {
                const float v = e4m3_scaled(c[d]) * gn;
                const int64_t at = r * cols + k * 64 + i * 8 + d;
                if (f16) oh[at] = (sycl::half) v;
                else of[at] = v;
            }
        }
    });
    return 0;
    NS_CATCH
}

// Variants 0 and 3: a sub-group of 16 an output row, RPG rows a work-group (RPS a sub-group in turn); each lane a run of 64-column chunks,
// every column at once. The codebook's gathers go through L1 (a lane an index): as float32 (cbf, widened once a layer:
// no conversion a weight), fp16 (3: cbf is the fp16 codebook) or the E4M3 bytes (cbf null). Measured on the B70 (6 experts of a 12-bit layer, one token): gate | up
// 208 us, down 121 us with cbf (218 / 144 from the bytes); kept local-memory codebooks (bytes, fp16; persistent or not;
// rows staged; 4 rows sharing the activations) all measured slower, 217-720 us.
int ns_ds41_vq_mmv(ns_gpu* g, const uint64_t* ptrs, const int32_t* meta, int64_t entries, const uint8_t* cb, const float* cbf, const float* x, float* y,
                   int64_t rows, int64_t cols, int64_t nbit, int variant) {
    NS_TRY
    if (!ns_ds41_vq_supported(rows, cols, nbit)) return ns_fail("ns_ds41_vq_mmv: cols a multiple of 64, 12 or 13 bits");
    const sycl::half* cbh = reinterpret_cast<const sycl::half*>(cbf);
    if (variant == 1 || variant == 2 || variant == 4) {
        const bool p = nbit > 12;
#define VQR(P, C) vq_mmv_rows<P, C>(g->q, ptrs, meta, entries, cb, cbh, x, y, rows, cols, 64)
        if (variant == 1) { if (p) VQR(true, 0); else VQR(false, 0); }
        else if (variant == 2) { if (p) VQR(true, 2); else VQR(false, 2); }
        else { if (p) VQR(true, 1); else VQR(false, 1); }
#undef VQR
        return 0;
    }
    constexpr int SG = 16, MC = 8;
    static const int rpg_s = [] { const char* v = std::getenv("NS_DS41_VQ_RPG"); return v ? std::atoi(v) : 8; }();
    static const int rps_s = [] { const char* v = std::getenv("NS_DS41_VQ_RPS"); return v ? std::atoi(v) : 1; }();
    const int RPG = rpg_s, RPS = rps_s;
    const int64_t rgs = (rows + RPG * RPS - 1) / (RPG * RPS), chunks = cols / 64;
    const bool plane = nbit > 12;
    g->q.parallel_for(sycl::nd_range<1>(entries * rgs * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t gi = it.get_group(0), e = gi / rgs, r0 = ((gi % rgs) * RPG + sg.get_group_id()[0]) * RPS;
        const int lane = sg.get_local_id()[0];
        for (int q = 0; q < RPS; ++q) {
        const int64_t r = r0 + q;
        if (r >= rows) return;
        const int32_t* me = meta + e * 10;
        const int nc = me[0];
        const VqMat m = vq_open(reinterpret_cast<const uint8_t*>(ptrs[e]), rows, cols, plane ? 13 : 12);
        const uint8_t* srow = m.stream + r * m.mrow;
        const uint8_t* prow = plane ? m.plane + r * m.prow : nullptr;
        float acc[MC] = {};
        for (int64_t k = lane; k < chunks; k += SG) {
            uint32_t idx[8];
            chunk_idx(reinterpret_cast<const uint32_t*>(srow + 12 * k), prow ? prow[k] : 0u, idx);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float wv[8];
                if (variant == 3) {
                    vq_weights<2>(cb, cbh, idx[i], wv);
                } else if (cbf) {
                    const sycl::float4* cw = reinterpret_cast<const sycl::float4*>(cbf + (size_t) idx[i] * 8);
                    const sycl::float4 c0 = cw[0], c1 = cw[1];
                    wv[0] = c0.x(); wv[1] = c0.y(); wv[2] = c0.z(); wv[3] = c0.w();
                    wv[4] = c1.x(); wv[5] = c1.y(); wv[6] = c1.z(); wv[7] = c1.w();
                } else {
                    const sycl::uint2 w = *reinterpret_cast<const sycl::uint2*>(cb + (size_t) idx[i] * 8);
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        wv[d] = e4m3_scaled((w.x() >> (8 * d)) & 0xFF);
                        wv[4 + d] = e4m3_scaled((w.y() >> (8 * d)) & 0xFF);
                    }
                }
                for (int c = 0; c < nc; ++c) {
                    const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(x + (int64_t) me[2 + c] * cols + k * 64 + i * 8);
                    const sycl::float4 a = xv[0], b = xv[1];
                    acc[c] += wv[0] * a.x() + wv[1] * a.y() + wv[2] * a.z() + wv[3] * a.w() + wv[4] * b.x() + wv[5] * b.y() + wv[6] * b.z() + wv[7] * b.w();
                }
            }
        }
        const float gn = (float) m.gains[r] * 256.f;
        for (int c = 0; c < nc; ++c) {
            const float s = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
            if (lane == 0) y[(int64_t) (me[1] + c) * rows + r] = s * gn;
        }
        }
    });
    return 0;
    NS_CATCH
}

int ns_ds41_fp8_dequant(ns_gpu* g, const uint8_t* w, int64_t rows, int64_t cols, int64_t r0, int64_t n, void* out, int f16) {
    NS_TRY
    if (r0 < 0 || r0 + n > rows) return ns_fail("ns_ds41_fp8_dequant: rows past the matrix");
    const int64_t tc = (cols + 31) / 32;
    const uint8_t* sc = w + rows * cols;
    float* of = static_cast<float*>(out);
    sycl::half* oh = static_cast<sycl::half*>(out);
    g->q.parallel_for(sycl::range<2>(n, cols), [=](sycl::id<2> id) {
        const int64_t r = r0 + id[0], c = id[1];
        const uint8_t s = sc[(r / 32) * tc + c / 32];
        const float scale = sycl::bit_cast<float>(s == 0 ? 0x00400000u : ((uint32_t) s << 23)) * 256.f;
        const float v = e4m3_scaled(w[r * cols + c]) * scale;
        if (f16) oh[id[0] * cols + c] = (sycl::half) v;
        else of[id[0] * cols + c] = v;
    });
    return 0;
    NS_CATCH
}

int ns_ds41_fp8_mmv(ns_gpu* g, const uint8_t* w, int64_t rows, int64_t cols, const float* x, float* y, int64_t t) {
    NS_TRY
    if (cols % 32 != 0 || t < 1 || t > 8) return ns_fail("ns_ds41_fp8_mmv: cols a multiple of 32, 1..8 rows of x");
    constexpr int SG = 16, RPG = 8;
    const int64_t tc = cols / 32;
    const uint8_t* sc = w + rows * cols;
    g->q.parallel_for(sycl::nd_range<1>((rows + RPG - 1) / RPG * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int64_t r = it.get_group(0) * RPG + sg.get_group_id()[0];
        const int lane = sg.get_local_id()[0];
        if (r >= rows) return;
        float acc[8] = {};
        for (int64_t k = lane; k < tc; k += SG) {
            const uint8_t s = sc[(r / 32) * tc + k];
            const float scale = sycl::bit_cast<float>(s == 0 ? 0x00400000u : ((uint32_t) s << 23)) * 256.f;
            const uint32_t* wr = reinterpret_cast<const uint32_t*>(w + r * cols + k * 32);
            float wv[32];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const uint32_t u = wr[q];
#pragma unroll
                for (int d = 0; d < 4; ++d) wv[q * 4 + d] = e4m3_scaled((u >> (8 * d)) & 0xFF);
            }
            for (int c = 0; c < t; ++c) {
                const float* xc = x + c * cols + k * 32;
                float s2 = 0.f;
#pragma unroll
                for (int d = 0; d < 32; ++d) s2 += wv[d] * xc[d];
                acc[c] += s2 * scale;
            }
        }
        for (int c = 0; c < t; ++c) {
            const float s = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
            if (lane == 0) y[c * rows + r] = s;
        }
    });
    return 0;
    NS_CATCH
}

}  // extern "C"
