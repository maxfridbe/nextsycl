// mmvq.cpp: the qwen35 silo's decode products (silo.h: ns_q35_mmvq) - 1..8 columns of Q8_1 activations against
// the stored blocks of the types a dense Qwen3.8-27B file holds (Q4_K, Q5_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S), tuned for
// its shapes (5,120 / 6,144 / 17,408) on the B70. Started from Strata's wide kernels (kernels/strata, native_mmvq:
// 8 lanes a 256-block, 16-byte weight loads, dp4a against Q8_1), which stay as they are for the engines tuned on them;
// what changed here and why (B70, one column, GB/s of weights; Strata's in brackets):
//
// - the scales and d from one aligned load of the block's head, not byte loads - small global loads are slow on Xe2:
//   Q4_K 507 (366), Q5_K 525 (415); the same for IQ4_XS
// - rows a sub-group sharing each lane's activations (ROWS, NS_Q35_MMVQ_ROWS): measured per type, one row wins now
// - IQ4_XS / IQ4_NL: a byte of two nibbles -> its two codebook bytes from a 256-entry table in local memory, about
//   ten operations for eight weights instead of ~80 selects: IQ4_XS ~490 (361), IQ4_NL 486 (326)
// - Q3_K and IQ3_S (Strata's are the generic kernels: 89 and 118): the same wide scheme, IQ3_S's grid and sign masks
//   in local memory, the weights through 16-byte loads: 380 and 439
//
// Work-groups of kernels with a table take rows in a stride (512 groups), so each fills its table once.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"
#include "common.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>

// ggml's IQ grids (iq3s_grid)
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

namespace {
using namespace q35silo;
constexpr int QK = 256, Q8K = 32, WARPS = 4, WARP = 32;

struct Q81Block { sycl::half2 ds; int8_t qs[32]; };
struct Q4KBlock { sycl::half2 dm; uint8_t scales[12]; uint8_t qs[128]; };
struct Q5KBlock { sycl::half2 dm; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct Q3KBlock { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; sycl::half d; };
struct IQ4XSBlock { sycl::half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };
struct IQ4NLBlock { sycl::half d; uint8_t qs[16]; };
static_assert(sizeof(Q81Block) == 36 && sizeof(Q4KBlock) == 144 && sizeof(Q5KBlock) == 176 && sizeof(Q3KBlock) == 110);
static_assert(sizeof(IQ4XSBlock) == 136 && sizeof(IQ4NLBlock) == 18 && sizeof(block_iq3_s) == 110);

inline int dp4a(int a, int b, int c) { return dpct::dp4a(a, b, c); }   // signed bytes, int32 accumulate (Strata's)
inline int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = dp4a(a.x(), b.x(), acc); acc = dp4a(a.y(), b.y(), acc); acc = dp4a(a.z(), b.z(), acc); return dp4a(a.w(), b.w(), acc);
}
// The activations in the silo's split Q8_1 layout (silo.h: ns_q35_q8_split): a column's quants contiguous and
// 16-byte aligned, its scales apart - one 16-byte load for 16 quants (Q8_1's 36-byte blocks leave them 4-byte
// aligned: four loads). `XA` is one column: block b's quants at qs + 32 b, its scale d[b].
struct XA {
    const int8_t* qs;
    const sycl::half* d;
    const int16_t* sm;
    sycl::int4 q16(int b, int half) const { return *reinterpret_cast<const sycl::int4*>(qs + 32 * b + 16 * half); }
    float dd(int b) const { return (float) d[b]; }
    int sum16(int b, int half) const { return sm[2 * b + half]; }   // the 16 quants' sum (the K-quants' mins)
};
inline sycl::int4 ld_q8_16(const Q81Block* b, int half) {   // standard Q8_1: qs is 4-byte aligned, four loads
    const int* q = reinterpret_cast<const int*>(b->qs) + 4 * half;
    return sycl::int4(q[0], q[1], q[2], q[3]);
}
// column j of `ncols` columns of n_in values in the split layout at xs: quants [ncols][n_in], scales [ncols][n_in /
// 32] (half), the 16-value groups' sums [ncols][n_in / 16] (int16)
inline XA xcol(const int8_t* xs, int n_in, int ncols, int j) {
    const sycl::half* d = reinterpret_cast<const sycl::half*>(xs + std::size_t(ncols) * n_in);
    const int16_t* sm = reinterpret_cast<const int16_t*>(d + std::size_t(ncols) * (n_in / 32));
    return XA{xs + std::size_t(j) * n_in, d + std::size_t(j) * (n_in / 32), sm + std::size_t(j) * (n_in / 16)};
}
// ---- Q4_K / Q5_K: lane g = sub-block pair 2 (g >> 1), half g & 1 - 32 quants of two sub-blocks
// a Q4_K / Q5_K block's first 16 bytes (16-byte aligned: 144 and 176 are multiples of 16): dm, then the scales
inline void k4_head(const void* b, int sb, float& dl, float& ml, float& dh, float& mh) {
    const sycl::int4 hd = *reinterpret_cast<const sycl::int4*>(b);
    const uint32_t dmw = uint32_t(hd.x()), sc[3] = {uint32_t(hd.y()), uint32_t(hd.z()), uint32_t(hd.w())};
    const float d = (float) sycl::bit_cast<sycl::half>(uint16_t(dmw & 0xffff)), dmin = (float) sycl::bit_cast<sycl::half>(uint16_t(dmw >> 16));
    float s0, m0, s1, m1; k4_scale_min(sc, sb, s0, m0); k4_scale_min(sc, sb + 1, s1, m1);
    dl = d * s0; ml = dmin * m0; dh = d * s1; mh = dmin * m1;
}
struct WideQ4K {
    using Block = Q4KBlock;
    static constexpr bool SPLIT = true;   // reads the split activations (B70: 1 column +4-8%, 4 columns +8%)
    static constexpr int ROWS = 1;   // with the head load: 1 row 507/470/488 GB/s, 2 rows 490/411/466 (before it 2 rows won)
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    struct W { sycl::int4 lo, hi; float dl, dh, ml, mh; };
    static W load(const Block* b, int g, const uint32_t*) {
        W r;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];
        r.lo = sycl::int4(q.x() & 0x0f0f0f0f, q.y() & 0x0f0f0f0f, q.z() & 0x0f0f0f0f, q.w() & 0x0f0f0f0f);
        r.hi = sycl::int4((q.x() >> 4) & 0x0f0f0f0f, (q.y() >> 4) & 0x0f0f0f0f, (q.z() >> 4) & 0x0f0f0f0f, (q.w() >> 4) & 0x0f0f0f0f);
        k4_head(b, 2 * (g >> 1), r.dl, r.ml, r.dh, r.mh);
        return r;
    }
    // the activations of lane g's quants (shared by the sub-group's rows) and their sums (the mins' term)
    struct X { sycl::int4 ua, uc; float da, dc; int sa, sc; };
    static X xload(int g, const XA& xa, int b0) {   // b0 = the 256-block's first 32-block
        X v; const int sb = 2 * (g >> 1), half = g & 1;
        const int a = b0 + sb, c = b0 + sb + 1;
        v.ua = xa.q16(a, half); v.uc = xa.q16(c, half);
        v.da = xa.dd(a); v.dc = xa.dd(c);
        v.sa = xa.sum16(a, half); v.sc = xa.sum16(c, half);   // precomputed (the quantization's): not 8 dp4a a column
        return v;
    }
    static float dot(const W& r, const X& v) {
        return v.da * (r.dl * (float) dp4a4(r.lo, v.ua, 0) - r.ml * (float) v.sa) +
               v.dc * (r.dh * (float) dp4a4(r.hi, v.uc, 0) - r.mh * (float) v.sc);
    }
};
struct WideQ5K {
    using Block = Q5KBlock;
    static constexpr bool SPLIT = true;
    static constexpr int ROWS = 1;   // 2 rows: 484 against 525 (the high bits' arithmetic twice)
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    using W = WideQ4K::W;
    static W load(const Block* b, int g, const uint32_t*) {
        W r; const int sb = 2 * (g >> 1), half = g & 1;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];
        const sycl::int4 h = reinterpret_cast<const sycl::int4*>(b->qh)[half];   // qh bytes of positions 16 half ..
        auto hb = [](int v, int bit) { return ((v >> bit) & 0x01010101) << 4; };
        r.lo = sycl::int4((q.x() & 0x0f0f0f0f) | hb(h.x(), sb), (q.y() & 0x0f0f0f0f) | hb(h.y(), sb),
                          (q.z() & 0x0f0f0f0f) | hb(h.z(), sb), (q.w() & 0x0f0f0f0f) | hb(h.w(), sb));
        r.hi = sycl::int4(((q.x() >> 4) & 0x0f0f0f0f) | hb(h.x(), sb + 1), ((q.y() >> 4) & 0x0f0f0f0f) | hb(h.y(), sb + 1),
                          ((q.z() >> 4) & 0x0f0f0f0f) | hb(h.z(), sb + 1), ((q.w() >> 4) & 0x0f0f0f0f) | hb(h.w(), sb + 1));
        k4_head(b, sb, r.dl, r.ml, r.dh, r.mh);
        return r;
    }
    using X = WideQ4K::X;
    static X xload(int g, const XA& xa, int b0) { return WideQ4K::xload(g, xa, b0); }
    static float dot(const W& r, const X& v) { return WideQ4K::dot(r, v); }
};

// ---- IQ4: a byte of two nibbles -> their codebook bytes (unsigned) at bits 0 and 16, from the work-group's table
inline uint32_t iq4_tab_entry(int b) {
    constexpr int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return uint32_t(uint8_t(kv[b & 15])) | uint32_t(uint8_t(kv[b >> 4])) << 16;
}
// four weight bytes -> the codebook bytes of their low nibbles (lo) and of their high nibbles (hi)
inline void iq4_tab4(const uint32_t* tab, uint32_t v, int& lo, int& hi) {
    const uint32_t a = tab[v & 0xff] | tab[(v >> 8) & 0xff] << 8, c = tab[(v >> 16) & 0xff] | tab[v >> 24] << 8;   // [l0 l1 h0 h1], [l2 l3 h2 h3]
    lo = int((a & 0xffff) | c << 16);
    hi = int(a >> 16 | (c & 0xffff0000u));
}
// lane g = sub-block g: 16 bytes, 32 quants
struct WideIQ4XS {
    using Block = IQ4XSBlock;
    static constexpr bool SPLIT = true;
    static constexpr int ROWS = 1;   // 2 rows: 454 against 481
    static constexpr int LUT = 256;
    static uint32_t lut(int i) { return iq4_tab_entry(i); }
    struct W { sycl::int4 lo, hi; float d; };
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        const sycl::int2* q2 = reinterpret_cast<const sycl::int2*>(b->qs + 16 * g);   // 8-byte aligned (136 = 17 x 8)
        const sycl::int2 qa = q2[0], qb = q2[1];
        const uint32_t v[4] = {(uint32_t) qa.x(), (uint32_t) qa.y(), (uint32_t) qb.x(), (uint32_t) qb.y()};
        int lo[4], hi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) iq4_tab4(tab, v[i], lo[i], hi[i]);
        r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]); r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
        // the block's first 8 bytes in one load: d, scales_h, scales_l
        const sycl::int2 hd = *reinterpret_cast<const sycl::int2*>(b);
        const uint32_t h0 = uint32_t(hd.x()), sl = uint32_t(hd.y());
        const int ls = int((sl >> (4 * g)) & 0x0f) | int(((h0 >> (16 + 2 * g)) & 0x03) << 4);
        r.d = (float) sycl::bit_cast<sycl::half>(uint16_t(h0 & 0xffff)) * (float) (ls - 32);
        return r;
    }
    struct X { sycl::int4 u0, u1; float d; };
    static X xload(int g, const XA& xa, int b0) { return X{xa.q16(b0 + g, 0), xa.q16(b0 + g, 1), xa.dd(b0 + g)}; }
    static float dot(const W& r, const X& v) { return r.d * v.d * (float) dp4a4(r.hi, v.u1, dp4a4(r.lo, v.u0, 0)); }
};

// ---- Q3_K (2-byte aligned blocks: load16_a2): lane g = (half h, l-group lg, j pair jp) takes qs[32h + 16lg ..] and
// hmask[16lg ..] and the shifts j = 2jp, 2jp + 1 - two sub-blocks of 16 (8h + 2j + lg). A quant = its 2 bits | its
// hmask bit << 2, less 4 (ggml's -4 where the bit is clear): 4 x the activations' sums taken off a sub-block
struct WideQ3K {
    using Block = Q3KBlock;
    static constexpr bool SPLIT = false;   // the standard blocks: the split layout measured 7-9% slower here
    static constexpr int ROWS = 1;   // 1 row 381/357/367 GB/s, 2 rows 384/310/363
    static constexpr int LUT = 0;
    static uint32_t lut(int) { return 0; }
    struct W { sycl::int4 q[2]; float d[2]; };
    static W load(const Block* b, int g, const uint32_t*) {
        W r;
        const int h = g >> 2, lg = (g >> 1) & 1, jp = g & 1;
        const sycl::int4 qs = load16_a2(b->qs + 32 * h + 16 * lg), hm = load16_a2(b->hmask + 16 * lg);
        const float d = (float) b->d;
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const int j = 2 * jp + jj, hb = 4 * h + j, i = 8 * h + 2 * j + lg;
            auto q3 = [&](int qv, int hv) { return ((qv >> (2 * j)) & 0x03030303) | (((hv >> hb) & 0x01010101) << 2); };
            r.q[jj] = sycl::int4(q3(qs.x(), hm.x()), q3(qs.y(), hm.y()), q3(qs.z(), hm.z()), q3(qs.w(), hm.w()));
            const int lo = i < 8 ? (b->scales[i] & 0xF) : (b->scales[i - 8] >> 4);
            const int hi = (b->scales[8 + (i & 3)] >> (2 * (i >> 2))) & 3;
            r.d[jj] = d * (float) ((lo | (hi << 4)) - 32);
        }
        return r;
    }
    struct X { sycl::int4 u[2]; float dx[2]; int sum[2]; };
    static X xload(int g, const Q81Block* xb) {
        X v;
        const int h = g >> 2, lg = (g >> 1) & 1, jp = g & 1;
        const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const Q81Block* a = xb + 4 * h + 2 * jp + jj;
            v.u[jj] = ld_q8_16(a, lg);
            v.dx[jj] = (float) a->ds[0];
            v.sum[jj] = dp4a4(ones, v.u[jj], 0);
        }
        return v;
    }
    static float dot(const W& r, const X& v) {
        return r.d[0] * v.dx[0] * (float) (dp4a4(r.q[0], v.u[0], 0) - 4 * v.sum[0]) +
               r.d[1] * v.dx[1] * (float) (dp4a4(r.q[1], v.u[1], 0) - 4 * v.sum[1]);
    }
};

// ---- IQ3_S (2-byte aligned blocks): lane g = sub-block g - eight grid entries (9-bit indices: qs | a qh bit << 8)
// from the work-group's copy of iq3s_grid, their signs (a byte a pair) applied bytewise (grid bytes are odd, so
// (v ^ m) + 1 never carries), the scale d (1 + 2 x its nibble)
inline uint32_t sign_mask4(uint32_t b) { return ((b * 0x00204081u) & 0x01010101u) * 0xffu; }   // bit i -> 0xff in byte i
struct WideIQ3S {
    using Block = block_iq3_s;
    static constexpr bool SPLIT = false;   // the standard blocks (split: 7% slower at one column)
    static constexpr int ROWS = 1;
    static constexpr int LUT = 512 + 16;   // the grid, then the 16 sign masks of four bits (32-bit multiplies are slow)
    static uint32_t lut(int i) { return i < 512 ? iq3s_grid[i] : sign_mask4(uint32_t(i - 512)); }
    using W = WideIQ4XS::W;
    static W load(const Block* b, int g, const uint32_t* tab) {
        W r;
        // 16-byte loads, each lane keeping its part: qs bytes 8g.., signs 4g.., qh byte g
        const sycl::int4 q4 = load16_a2(b->qs + 16 * (g >> 1)), s4 = load16_a2(b->signs + 16 * (g >> 2));
        const uint32_t qa = uint32_t(g & 1 ? q4.z() : q4.x()), qb = uint32_t(g & 1 ? q4.w() : q4.y());
        const int sw = g & 3;
        const uint32_t sg = uint32_t(sw == 0 ? s4.x() : sw == 1 ? s4.y() : sw == 2 ? s4.z() : s4.w());
        const uint32_t qh = (reinterpret_cast<const uint16_t*>(b->qh)[g >> 1] >> (8 * (g & 1))) & 0xff;
        int v[8];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qw = l < 2 ? qa : qb, sh = 16 * (l & 1);
            const uint32_t i0 = ((qw >> sh) & 0xff) | ((qh << (8 - 2 * l)) & 0x100), i1 = ((qw >> (sh + 8)) & 0xff) | ((qh << (7 - 2 * l)) & 0x100);
            const uint32_t sb = (sg >> (8 * l)) & 0xff, m0 = tab[512 + (sb & 0xf)], m1 = tab[512 + (sb >> 4)];
            v[2 * l] = int((tab[i0] ^ m0) + (m0 & 0x01010101u));
            v[2 * l + 1] = int((tab[i1] ^ m1) + (m1 & 0x01010101u));
        }
        r.lo = sycl::int4(v[0], v[1], v[2], v[3]); r.hi = sycl::int4(v[4], v[5], v[6], v[7]);
        r.d = (float) b->d * (float) (1 + 2 * ((b->scales[g >> 1] >> (4 * (g & 1))) & 0xf));
        return r;
    }
    using X = WideIQ4XS::X;
    static X xload(int g, const Q81Block* xb) {
        const Q81Block* a = xb + g;
        return X{ld_q8_16(a, 0), ld_q8_16(a, 1), (float) a->ds[0]};
    }
    static float dot(const W& r, const X& v) { return WideIQ4XS::dot(r, v); }
};

int lut_groups() {   // NS_Q35_MMVQ_GROUPS: the work-groups of a kernel with a table (default 512)
    static const int v = std::getenv("NS_Q35_MMVQ_GROUPS") ? std::max(1, std::atoi(std::getenv("NS_Q35_MMVQ_GROUPS"))) : 512;
    return v;
}
int rows_override() {   // NS_Q35_MMVQ_ROWS=1|2|4: every 256-block type's rows a sub-group (0: each type's own)
    static const int v = std::getenv("NS_Q35_MMVQ_ROWS") ? std::atoi(std::getenv("NS_Q35_MMVQ_ROWS")) : 0;
    return v;
}

// 256-blocks: 8 lanes a block, 4 blocks a sub-group step, R rows a sub-group sharing each lane's activations
template <typename F, int NCOLS, int R>
void wide_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ xq, const int8_t* __restrict__ x, float* __restrict__ y, int n_in, int n_out,
                 uint32_t* tab, const sycl::nd_item<3>& item) {
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    if constexpr (F::LUT > 0) {
        for (int i = warp * WARP + lane; i < F::LUT; i += WARPS * WARP) tab[i] = F::lut(i);
        sycl::group_barrier(item.get_group());
    }
    const int blocks_per_row = n_in / QK;
    const int g = lane & 7, sub = lane >> 3;
    auto sg = item.get_sub_group();
    XA xa[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) xa[j] = xcol(x, n_in, NCOLS, j);
    for (int row0 = (int(item.get_group(2)) * WARPS + warp) * R; row0 < n_out; row0 += int(item.get_group_range(2)) * WARPS * R) {
        const typename F::Block* wr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wr[r] = w + std::size_t(sycl::min(row0 + r, n_out - 1)) * blocks_per_row;
        float acc[R][NCOLS] = {};
        for (int kbx = sub; kbx < blocks_per_row; kbx += 4) {
            typename F::W wv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) wv[r] = F::load(wr[r] + kbx, g, tab);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                typename F::X xv;
                if constexpr (F::SPLIT) xv = F::xload(g, xa[j], kbx * (QK / Q8K));
                else xv = F::xload(g, xq + std::size_t(j) * (n_in / Q8K) + kbx * (QK / Q8K));
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r][j] += F::dot(wv[r], xv);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                float v = acc[r][j];
#pragma unroll
                for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
                if (lane == 0 && row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = v;
            }
    }
}
template <typename F, int NCOLS, int R>
void launch(sycl::queue& q, const void* weights, const void* xstd, const void* xs, float* y, int n_in, int n_out) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* xq = static_cast<const Q81Block*>(xstd);
    const auto* x = static_cast<const int8_t*>(xs);
    unsigned blocks = unsigned((std::size_t(n_out) + WARPS * R - 1) / (WARPS * R));
    if (F::LUT > 0) blocks = std::min(blocks, unsigned(lut_groups()));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(F::LUT > 0 ? F::LUT : 1), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                           wide_kernel<F, NCOLS, R>(w, xq, x, y, n_in, n_out, tab.template get_multi_ptr<sycl::access::decorated::no>().get(), it);
                       });
    });
}
template <typename F, int R>
void launch_cols(sycl::queue& q, const void* w, const void* xq, const void* x, float* y, int n_in, int n_out, int ncols) {
    switch (ncols) {
        case 1: launch<F, 1, R>(q, w, xq, x, y, n_in, n_out); return;
        case 2: launch<F, 2, R>(q, w, xq, x, y, n_in, n_out); return;
        case 3: launch<F, 3, R>(q, w, xq, x, y, n_in, n_out); return;
        case 4: launch<F, 4, R>(q, w, xq, x, y, n_in, n_out); return;
        case 5: launch<F, 5, R>(q, w, xq, x, y, n_in, n_out); return;
        case 6: launch<F, 6, R>(q, w, xq, x, y, n_in, n_out); return;
        case 7: launch<F, 7, R>(q, w, xq, x, y, n_in, n_out); return;
        default: launch<F, 8, R>(q, w, xq, x, y, n_in, n_out); return;
    }
}
template <typename F>
void wide(sycl::queue& q, const void* w, const void* xq, const void* x, float* y, int n_in, int n_out, int ncols) {
    // past 4 columns two rows would hold 2 x 8 accumulators and 8 activation sets: one row there; 3-4 columns (a
    // verify pass) two rows sharing each lane's activations (B70, 4 columns, GB/s: Q4_K 270 -> 344, IQ4_XS 331 -> 406,
    // Q5_K 316 -> 377; at 2 columns it is mixed). Rows never change a column's arithmetic: verify rows stay exact.
    const int r = ncols > 4 ? 1 : rows_override() ? rows_override() : ncols >= 3 ? 2 : F::ROWS;
    if (r >= 4) launch_cols<F, 4>(q, w, xq, x, y, n_in, n_out, ncols);
    else if (r == 2) launch_cols<F, 2>(q, w, xq, x, y, n_in, n_out, ncols);
    else launch_cols<F, 1>(q, w, xq, x, y, n_in, n_out, ncols);
}

// ---- IQ4_NL (18-byte blocks of 32, 2-byte aligned): one lane a block (its 16 bytes = 32 nibbles), 32 blocks a step
template <int NCOLS>
void iq4nl_kernel(const IQ4NLBlock* __restrict__ w, const int8_t* __restrict__ x, float* __restrict__ y, int n_in, int n_out, uint32_t* tab,
                  const sycl::nd_item<3>& item) {
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    for (int i = warp * WARP + lane; i < 256; i += WARPS * WARP) tab[i] = iq4_tab_entry(i);
    sycl::group_barrier(item.get_group());
    const int blocks_per_row = n_in / 32;
    auto sg = item.get_sub_group();
    XA xa[NCOLS];
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) xa[j] = xcol(x, n_in, NCOLS, j);
    for (int row = int(item.get_group(2)) * WARPS + warp; row < n_out; row += int(item.get_group_range(2)) * WARPS) {
        const IQ4NLBlock* wr = w + std::size_t(row) * blocks_per_row;
        float acc[NCOLS] = {};
        for (int kb = lane; kb < blocks_per_row; kb += WARP) {
            const sycl::int4 q = load16_a2(wr[kb].qs);
            const uint32_t v[4] = {(uint32_t) q.x(), (uint32_t) q.y(), (uint32_t) q.z(), (uint32_t) q.w()};
            int lo[4], hi[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) iq4_tab4(tab, v[i], lo[i], hi[i]);
            const sycl::int4 l4(lo[0], lo[1], lo[2], lo[3]), h4(hi[0], hi[1], hi[2], hi[3]);
            const float d = (float) wr[kb].d;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                acc[j] += d * xa[j].dd(kb) * (float) dp4a4(h4, xa[j].q16(kb, 1), dp4a4(l4, xa[j].q16(kb, 0), 0));
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[j];
#pragma unroll
            for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (lane == 0) y[std::size_t(j) * n_out + row] = v;
        }
    }
}
template <int NCOLS>
void launch_iq4nl(sycl::queue& q, const void* weights, const void* xq, float* y, int n_in, int n_out) {
    const auto* w = static_cast<const IQ4NLBlock*>(weights);
    const auto* x = static_cast<const int8_t*>(xq);
    const unsigned blocks = std::min(unsigned((std::size_t(n_out) + WARPS - 1) / WARPS), unsigned(lut_groups()));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> tab(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<3>(sycl::range(1, 1, blocks) * sycl::range(1, WARPS, WARP), sycl::range(1, WARPS, WARP)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
                           iq4nl_kernel<NCOLS>(w, x, y, n_in, n_out, tab.template get_multi_ptr<sycl::access::decorated::no>().get(), it);
                       });
    });
}
void iq4nl(sycl::queue& q, const void* w, const void* x, float* y, int n_in, int n_out, int ncols) {
    switch (ncols) {
        case 1: launch_iq4nl<1>(q, w, x, y, n_in, n_out); return;
        case 2: launch_iq4nl<2>(q, w, x, y, n_in, n_out); return;
        case 3: launch_iq4nl<3>(q, w, x, y, n_in, n_out); return;
        case 4: launch_iq4nl<4>(q, w, x, y, n_in, n_out); return;
        case 5: launch_iq4nl<5>(q, w, x, y, n_in, n_out); return;
        case 6: launch_iq4nl<6>(q, w, x, y, n_in, n_out); return;
        case 7: launch_iq4nl<7>(q, w, x, y, n_in, n_out); return;
        default: launch_iq4nl<8>(q, w, x, y, n_in, n_out); return;
    }
}
}  // namespace

extern "C" {

// standard Q8_1 blocks (column j's at j x n_in / 32) -> the split layout: quants [ncols][n_in], then scales
// [ncols][n_in / 32] (half)
int ns_q35_q8_split(ns_gpu* g, const void* q8_1, void* xs, int64_t n_in, int64_t ncols) {
    NS_TRY
    const int64_t nb = ncols * (n_in / 32);
    const auto* src = static_cast<const Q81Block*>(q8_1);
    int8_t* qs = static_cast<int8_t*>(xs);
    sycl::half* d = reinterpret_cast<sycl::half*>(qs + ncols * n_in);
    int16_t* sm = reinterpret_cast<int16_t*>(d + nb);
    g->q.parallel_for(sycl::range<1>((size_t) nb * 2), [=](sycl::id<1> i) {   // 2 work-items a block, 16 quants each
        const int64_t b = i[0] / 2, h = i[0] % 2;
        int t = 0;
        for (int k = 0; k < 16; ++k) {
            qs[b * 32 + 16 * h + k] = src[b].qs[16 * h + k];
            t += src[b].qs[16 * h + k];
        }
        sm[2 * b + h] = (int16_t) t;
        if (h == 0) d[b] = src[b].ds[0];
    });
    return 0;
    NS_CATCH
}

// x [ncols, n_in] float32 -> its Q8_1 blocks (q8_1, nullable: Strata's quantize_q8_1, the same bits) and the split
// layout (xs) in one launch: a sub-group of 32 a block
int ns_q35_quantize_q8(ns_gpu* g, const float* x, void* q8_1, void* xs, int64_t n_in, int64_t ncols) {
    NS_TRY
    const int64_t n = ncols * n_in;
    auto* q = static_cast<Q81Block*>(q8_1);
    int8_t* qs = static_cast<int8_t*>(xs);
    sycl::half* d = reinterpret_cast<sycl::half*>(qs + n);
    int16_t* sm = reinterpret_cast<int16_t*>(d + n / 32);
    g->q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((n + 255) / 256 * 256)), sycl::range<1>(256)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        auto sg = it.get_sub_group();
        const int64_t i = it.get_global_id(0);
        const float v = i < n ? x[i] : 0.0f;
        float amax = sycl::fabs(v), sum = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
            sum += sycl::permute_group_by_xor(sg, sum, o);
        }
        const float dd = strata::kernels::q8_1_finite(amax / 127.0f);
        const int8_t qv = strata::kernels::q8_1_quant(v, dd, amax);
        int t = qv;   // the 16-value group's sum (lanes 0-15, 16-31), every lane in the shuffles
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) t += sycl::permute_group_by_xor(sg, t, o);
        if (i >= n) return;
        const int64_t b = i / 32, l = i % 32;
        qs[i] = qv;
        const sycl::half2 ds = strata::kernels::q8_1_ds(dd, sum);
        if (l == 0) d[b] = ds[0];
        if (l % 16 == 0) sm[i / 16] = (int16_t) t;
        if (q) {
            q[b].qs[l] = qv;
            if (l == 0) q[b].ds = ds;
        }
    });
    return 0;
    NS_CATCH
}

size_t ns_q35_q8_split_bytes(int64_t n_in, int64_t ncols) { return (size_t) (ncols * n_in + ncols * (n_in / 32) * 2 + ncols * (n_in / 16) * 2); }

int ns_q35_mmvq_supported(int type, int64_t n_in) {
    switch (type) {
        case 11: case 12: case 13: case 21: case 23: return n_in % QK == 0;
        case 20: return n_in % 32 == 0;
        default: return 0;
    }
}

int ns_q35_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, const void* x_split, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (ncols < 1 || ncols > 8) return ns_fail("ns_q35_mmvq: 1..8 columns");
    if (!ns_q35_mmvq_supported(type, n_in)) return ns_fail("ns_q35_mmvq: ggml type " + std::to_string(type) + " at this width is not its own");
    const int ni = (int) n_in, no = (int) n_out, nc = (int) ncols;
    switch (type) {
        case 11: wide<WideQ3K>(g->q, w, x_q8_1, x_split, y, ni, no, nc); break;
        case 12: wide<WideQ4K>(g->q, w, x_q8_1, x_split, y, ni, no, nc); break;
        case 13: wide<WideQ5K>(g->q, w, x_q8_1, x_split, y, ni, no, nc); break;
        case 20: iq4nl(g->q, w, x_split, y, ni, no, nc); break;
        case 21: wide<WideIQ3S>(g->q, w, x_q8_1, x_split, y, ni, no, nc); break;
        case 23: wide<WideIQ4XS>(g->q, w, x_q8_1, x_split, y, ni, no, nc); break;
    }
    return 0;
    NS_CATCH
}

}  // extern "C"
