// dequant.cpp: the qwen35 silo's weight expansion to half (silo.h: ns_q35_dequant_f16) - a prompt pass expands each
// matrix for the half GEMMs. The shared ones measured, at a 27B's shapes on the B70: IQ4_XS 190 GB/s, the K-quants
// ~300, Q6_K none (through float32: 13-15 GB/s - a work-item a block writing 256 floats in a row, neighbours 1 KB
// apart). Here a sub-group a 256-block, a lane 16 consecutive values (32-byte stores: the sub-group writes the block's
// 512 bytes in one go), its bytes through aligned 16-byte loads (load16_a2 for the 2-byte aligned blocks), the IQ
// codebook / grid from local memory.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"
#include "common.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

namespace {
using namespace q35silo;
using half = sycl::half;
using V16 = sycl::vec<half, 16>;
constexpr int QK = 256, SG = 16, BPW = 16, WG = SG * BPW;   // BPW: blocks (sub-groups) a work-group

inline float h2f(uint16_t v) { return (float) sycl::bit_cast<half>(v); }
inline uint32_t word(const sycl::int4& v, int i) { return uint32_t(i == 0 ? v.x() : i == 1 ? v.y() : i == 2 ? v.z() : v.w()); }
inline int byte_of(const sycl::int4& v, int i) { return int((word(v, i >> 2) >> (8 * (i & 3))) & 0xff); }

// Each type: a 256-value block's values 16 L .. 16 L + 15 (lane L), `tab` its local table (words: the IQ4 codebook as
// floats' bits, IQ3_S's grid and sign masks)
struct DQ4K {
    using Block = block_q4_K;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    static void get(const Block* b, int L, const uint32_t*, float* y) {
        const int j = L / 2, jp = L / 4;   // sub-block (32 values), its pair (a 32-byte qs run)
        const sycl::int4 hd = *reinterpret_cast<const sycl::int4*>(b);
        const uint32_t dmw = uint32_t(hd.x()), sc[3] = {uint32_t(hd.y()), uint32_t(hd.z()), uint32_t(hd.w())};
        float s, m;
        k4_scale_min(sc, j, s, m);
        const float d = h2f(uint16_t(dmw & 0xffff)) * s, mn = h2f(uint16_t(dmw >> 16)) * m;
        const sycl::int4 q = *reinterpret_cast<const sycl::int4*>(b->qs + 32 * jp + 16 * (L % 2));
        const int sh = 4 * (j % 2);
#pragma unroll
        for (int i = 0; i < 16; ++i) y[i] = d * (float) ((byte_of(q, i) >> sh) & 0xF) - mn;
    }
};
struct DQ5K {
    using Block = block_q5_K;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    static void get(const Block* b, int L, const uint32_t*, float* y) {
        const int j = L / 2, jp = L / 4;
        const sycl::int4 hd = *reinterpret_cast<const sycl::int4*>(b);
        const uint32_t dmw = uint32_t(hd.x()), sc[3] = {uint32_t(hd.y()), uint32_t(hd.z()), uint32_t(hd.w())};
        float s, m;
        k4_scale_min(sc, j, s, m);
        const float d = h2f(uint16_t(dmw & 0xffff)) * s, mn = h2f(uint16_t(dmw >> 16)) * m;
        const sycl::int4 q = *reinterpret_cast<const sycl::int4*>(b->qs + 32 * jp + 16 * (L % 2));
        const sycl::int4 h = *reinterpret_cast<const sycl::int4*>(b->qh + 16 * (L % 2));
        const int sh = 4 * (j % 2);
#pragma unroll
        for (int i = 0; i < 16; ++i) y[i] = d * (float) (((byte_of(q, i) >> sh) & 0xF) | (((byte_of(h, i) >> j) & 1) << 4)) - mn;
    }
};
// Q6_K (210-byte blocks): value e = 128 nh + 32 qq + l: ql[64 nh + 32 (qq % 2) + l] (nibble qq / 2), qh[32 nh + l]
// bits 2 qq, scale [8 nh + 2 qq + l / 16]
struct DQ6K {
    using Block = block_q6_K;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    static void get(const Block* b, int L, const uint32_t*, float* y) {
        const int nh = L / 8, qq = (L / 2) % 4, l0 = 16 * (L % 2);
        const sycl::int4 ql = load16_a2(b->ql + 64 * nh + 32 * (qq % 2) + l0), qh = load16_a2(b->qh + 32 * nh + l0);
        const float d = h2f(*reinterpret_cast<const uint16_t*>(&b->d)) * (float) b->scales[8 * nh + 2 * qq + L % 2];
        const int sl = 4 * (qq / 2), shh = 2 * qq;
#pragma unroll
        for (int i = 0; i < 16; ++i) y[i] = d * (float) (int(((byte_of(ql, i) >> sl) & 0xF) | (((byte_of(qh, i) >> shh) & 3) << 4)) - 32);
    }
};
// Q3_K (110-byte blocks): value e = 128 n + 32 j + l: (qs[32 n + l] >> 2 j) & 3, less 4 where hmask[l] bit 4 n + j is
// clear, scale [8 n + 2 j + l / 16] (6 bits, less 32)
struct DQ3K {
    using Block = block_q3_K;
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    static void get(const Block* b, int L, const uint32_t*, float* y) {
        const int n = L / 8, j = (L / 2) % 4, l0 = 16 * (L % 2), is = 8 * n + 2 * j + L % 2;
        const sycl::int4 qs = load16_a2(b->qs + 32 * n + l0), hm = load16_a2(b->hmask + l0), sc = load16_a2(b->scales);
        const int lo = is < 8 ? (byte_of(sc, is) & 0xF) : (byte_of(sc, is - 8) >> 4);
        const int hi = (byte_of(sc, 8 + (is & 3)) >> (2 * (is >> 2))) & 3;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(&b->d)) * (float) ((lo | (hi << 4)) - 32);
        const int hb = 4 * n + j;
#pragma unroll
        for (int i = 0; i < 16; ++i) y[i] = d * (float) (((byte_of(qs, i) >> (2 * j)) & 3) - (((byte_of(hm, i) >> hb) & 1) ? 0 : 4));
    }
};
// IQ4_XS (136-byte blocks, 8-byte aligned): sub-block L / 2's 16 bytes, low nibbles for values 0..15, high for 16..31
struct DIQ4XS {
    using Block = block_iq4_xs;
    static constexpr int LUT = 16;
    static uint32_t lut(int i) { return sycl::bit_cast<uint32_t>((float) kvalues_iq4nl[i]); }
    static void get(const Block* b, int L, const uint32_t* tab, float* y) {
        const int ib = L / 2;
        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * ib);
        const sycl::int2 qa = q2[0], qb = q2[1];
        const sycl::int4 q(qa.x(), qa.y(), qb.x(), qb.y());
        const sycl::int2 hd = *reinterpret_cast<const sycl::int2*>(b);
        const uint32_t h0 = uint32_t(hd.x()), sl = uint32_t(hd.y());
        const int ls = int((sl >> (4 * ib)) & 0x0f) | int(((h0 >> (16 + 2 * ib)) & 0x03) << 4);
        const float d = h2f(uint16_t(h0 & 0xffff)) * (float) (ls - 32);
        const int sh = 4 * (L % 2);
#pragma unroll
        for (int i = 0; i < 16; ++i) y[i] = d * sycl::bit_cast<float>(tab[(byte_of(q, i) >> sh) & 0xF]);
    }
};
// IQ4_NL (18-byte blocks of 32): block L / 2
struct DIQ4NL {
    using Block = block_iq4_nl;
    static constexpr int LUT = 16, PER = 8;   // blocks of 32 in 256 values
    static uint32_t lut(int i) { return sycl::bit_cast<uint32_t>((float) kvalues_iq4nl[i]); }
    static void get(const Block* b, int L, const uint32_t* tab, float* y) {
        const Block* c = b + L / 2;
        const sycl::int4 q = load16_a2(c->qs);
        const float d = h2f(*reinterpret_cast<const uint16_t*>(&c->d));
        const int sh = 4 * (L % 2);
#pragma unroll
        for (int i = 0; i < 16; ++i) y[i] = d * sycl::bit_cast<float>(tab[(byte_of(q, i) >> sh) & 0xF]);
    }
};
// Q8_0 (34-byte blocks of 32): block L / 2, half L % 2
struct DQ80 {
    using Block = block_q8_0;
    static constexpr int LUT = 0, PER = 8;
    static uint32_t lut(int) { return 0; }
    static void get(const Block* b, int L, const uint32_t*, float* y) {
        const Block* c = b + L / 2;
        const sycl::int4 q = load16_a2(c->qs + 16 * (L % 2));
        const float d = h2f(*reinterpret_cast<const uint16_t*>(&c->d));
#pragma unroll
        for (int i = 0; i < 16; ++i) y[i] = d * (float) (int8_t) byte_of(q, i);
    }
};
// IQ3_S (110-byte blocks): sub-block L / 2 (32 values: four grid pairs), half L % 2 its pairs 2h, 2h + 1; the grid
// and the sign masks from local memory
inline uint32_t sign_mask4(uint32_t b) { return ((b * 0x00204081u) & 0x01010101u) * 0xffu; }
struct DIQ3S {
    using Block = block_iq3_s;
    static constexpr int LUT = 512 + 16;
    static uint32_t lut(int i) { return i < 512 ? iq3s_grid[i] : sign_mask4(uint32_t(i - 512)); }
    static void get(const Block* b, int L, const uint32_t* tab, float* y) {
        const int ib = L / 2, h = L % 2;
        const sycl::int4 q4 = load16_a2(b->qs + 16 * (ib >> 1)), s4 = load16_a2(b->signs + 16 * (ib >> 2));
        const uint32_t qw = word(q4, 2 * (ib & 1) + h);   // qs bytes 8 ib + 4 h ..: the two pairs' four indices
        const uint32_t sg = word(s4, ib & 3);
        const uint32_t qh = (reinterpret_cast<const uint16_t*>(b->qh)[ib >> 1] >> (8 * (ib & 1))) & 0xff;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(&b->d)) * (float) (1 + 2 * ((b->scales[ib >> 1] >> (4 * (ib & 1))) & 0xf));
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int l = 2 * h + p;   // the pair (8 values)
            const uint32_t i0 = ((qw >> (16 * p)) & 0xff) | ((qh << (8 - 2 * l)) & 0x100), i1 = ((qw >> (16 * p + 8)) & 0xff) | ((qh << (7 - 2 * l)) & 0x100);
            const uint32_t sb = (sg >> (8 * l)) & 0xff, m0 = tab[512 + (sb & 0xf)], m1 = tab[512 + (sb >> 4)];
            const uint32_t g0 = tab[i0], g1 = tab[i1];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int v0 = int((g0 >> (8 * e)) & 0xff), v1 = int((g1 >> (8 * e)) & 0xff);
                y[8 * p + e] = d * (float) (((m0 >> (8 * e)) & 1) ? -v0 : v0);
                y[8 * p + 4 + e] = d * (float) (((m1 >> (8 * e)) & 1) ? -v1 : v1);
            }
        }
    }
};

template <typename F>
void expand(sycl::queue& q, const void* src, int64_t n, half* dst) {
    const int64_t blocks = n / QK;
    int64_t groups = (blocks + BPW - 1) / BPW;
    if (F::LUT > 0) groups = std::min<int64_t>(groups, 2048);   // each fills its table once, then takes blocks in a stride
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) groups * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int tid = (int) it.get_local_id(0), L = tid % SG;
            uint32_t* t = tab.get_multi_ptr<sycl::access::decorated::no>().get();
            if constexpr (F::LUT > 0) {
                for (int i = tid; i < F::LUT; i += WG) t[i] = F::lut(i);
                sycl::group_barrier(it.get_group());
            }
            for (int64_t blk = it.get_group(0) * BPW + tid / SG; blk < blocks; blk += it.get_group_range(0) * BPW) {
                float y[16];
                if constexpr (requires { F::PER; }) F::get(reinterpret_cast<const typename F::Block*>(src) + blk * F::PER, L, t, y);
                else F::get(reinterpret_cast<const typename F::Block*>(src) + blk, L, t, y);
                V16 o;
#pragma unroll
                for (int i = 0; i < 16; ++i) o[i] = (half) y[i];
                *reinterpret_cast<V16*>(dst + blk * QK + 16 * L) = o;
            }
        });
    });
}
}  // namespace

extern "C" {

int ns_q35_dequant_f16_supported(int type) {
    switch (type) {
        case 8: case 11: case 12: case 13: case 14: case 20: case 21: case 23: return 1;
        default: return 0;
    }
}

int ns_q35_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst) {
    NS_TRY
    if (!ns_q35_dequant_f16_supported(type) || n % QK != 0) return ns_fail("ns_q35_dequant_f16: ggml type " + std::to_string(type));
    half* d = reinterpret_cast<half*>(dst);
    switch (type) {
        case 8: expand<DQ80>(g->q, src, n, d); break;
        case 11: expand<DQ3K>(g->q, src, n, d); break;
        case 12: expand<DQ4K>(g->q, src, n, d); break;
        case 13: expand<DQ5K>(g->q, src, n, d); break;
        case 14: expand<DQ6K>(g->q, src, n, d); break;
        case 20: expand<DIQ4NL>(g->q, src, n, d); break;
        case 21: expand<DIQ3S>(g->q, src, n, d); break;
        case 23: expand<DIQ4XS>(g->q, src, n, d); break;
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
