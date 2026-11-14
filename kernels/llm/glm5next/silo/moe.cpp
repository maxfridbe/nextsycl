// moe.cpp: a layer's routed experts at decode widths (1..8 rows) for the expert type pairs the shared grouped kernels
// (ns_moe_grouped, Strata's native_expert_grouped) do not take: the REAP-50 IQ3_M file's IQ3_S down projections and
// the Q4_K_M file's Q6_K ones (silo.h). The same contract as ns_moe_grouped - the same expert blob [gate | up | down],
// the same table (grp_ptr, grp_start, n_groups, ent_dst, ent_tok), q8_1 activations in, unweighted rows out, phases
// 0 / 1 / 2 - so the engine swaps it in by type alone.
//
// Three launches: gate | up for every (group, row) - each entry of the group against the row, the clamped SwiGLU
// written as float32 to the scratch; that quantized to q8_1 (a work-item a block); down for every (group, row).
// A row is a 16-lane sub-group: lane k takes the row's dot calls k, k + 16, ... (ggml's vec_dot_*_q8_1, the calls
// of its CUDA mmvq) and the sub-group sums them. The order of a row's sums is fixed by the lane count alone, so a
// verify pass's rows equal one-token passes bit for bit (spec-check). The dot products are llama.cpp's
// (ggml-cuda/vecdotq.cuh, MIT) as Strata's iq_kernels.dp.cpp transcribes them.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "ns_gguf.h"

#include <cstdint>
#include <cstdlib>
#include <string>

namespace {

constexpr int LANES = 16;          // lanes a row
constexpr int ROWS = 16;           // rows a work-group (256 work-items)
constexpr int T_Q4_K = 12, T_Q6_K = 14, T_IQ3_S = 21;

inline int dp4a(int a, int b, int c) { return dpct::dp4a(a, b, c); }
inline int get_int_b2(const void* x, const int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    return (int) ((uint32_t) x16[2 * i32 + 0] | ((uint32_t) x16[2 * i32 + 1] << 16));
}
inline int get_int_b4(const void* x, const int i32) { return ((const int*) x)[i32]; }
// 0xFF in every byte of x that is non-zero (at most one bit a byte here)
inline int swar_ne4(unsigned x) {
    x = (x | (x >> 4)) & 0x0F0F0F0Fu;
    x = (x | (x >> 2)) & 0x03030303u;
    x = (x | (x >> 1)) & 0x01010101u;
    return (int) (x * 0xFFu);
}
// per byte a - s for s in {0x00, 0xFF}
inline int swar_sub4(unsigned a, unsigned s) { return (int) (((a & 0x7F7F7F7Fu) + (s & 0x01010101u)) ^ (a & 0x80808080u)); }

// grid: iq3s_grid staged in local memory by the caller (a gather from the global table is a transaction a lane)
inline float dot_iq3_s(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs, const uint32_t* grid) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const sycl::int2 qs_packed(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int g0 = (int) grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const int g1 = (int) grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const int signs0 = swar_ne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21));
        const int signs1 = swar_ne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17));
        const int grid_l = swar_sub4((unsigned) (g0 ^ signs0), (unsigned) signs0);
        const int grid_h = swar_sub4((unsigned) (g1 ^ signs1), (unsigned) signs1);
        sumi = dp4a(grid_l, get_int_b4(bq8_1[iqs / 2].qs, l0 + 0), sumi);
        sumi = dp4a(grid_h, get_int_b4(bq8_1[iqs / 2].qs, l0 + 1), sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = (float) bq3->d * (float) bq8_1[iqs / 2].ds[0];
    return d * (float) sumi;
}

inline float dot_q6_K(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs) {
    const block_q6_K* b = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = get_int_b2(b->ql, iqs);
    const int vh = get_int_b2(b->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
    const int8_t* scales = b->scales + scale_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int u = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        const float d8 = (float) bq8_1[bq8_offset + 2 * i].ds[0];
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = dpct::vectorized_binary<sycl::char4>(vil | vih, 0x20202020, dpct::sub_sat());   // - 32
        sumf += d8 * (float) (dp4a(vi, u, 0) * sc);
    }
    return (float) b->d * sumf;
}

// the 6-bit scales and mins of the 32-value group pair bq8_offset / 2 (llama.cpp; Q4_K)
inline void k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}

inline float dot_q4_K(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs) {
    const block_q4_K* b = (const block_q4_K*) vbq + kbx;
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int v0 = q4[0], v1 = q4[4];
    uint16_t aux[2];
    k_scale_min(b->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        const float d8 = (float) bq8i->ds[0];
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F, v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = dp4a(v1i, q8[4], dp4a(v0i, q8[0], 0));
        const int dot2 = dp4a(0x01010101, q8[4], dp4a(0x01010101, q8[0], 0));
        sumf_d += d8 * (float) (dot1 * sc[i]);
        sumf_m += d8 * (float) (dot2 * m[i]);
    }
    const float d = (float) b->dm[0], dmin = (float) b->dm[1];
    return d * sumf_d - dmin * sumf_m;
}

// qk = values a block, ipb = dot calls a block, step = the iqs stride between calls, bytes = a block's
template <int T> struct Fmt;
template <> struct Fmt<T_IQ3_S> {
    static constexpr int qk = 256, ipb = 8, step = 2, bytes = 110;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* g) { return dot_iq3_s(v, y, kbx, iqs, g); }
};
template <> struct Fmt<T_Q6_K> {
    static constexpr int qk = 256, ipb = QI6_K, step = 1, bytes = 210;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t*) { return dot_q6_K(v, y, kbx, iqs); }
};
template <> struct Fmt<T_Q4_K> {
    static constexpr int qk = 256, ipb = QI4_K / 2, step = 2, bytes = 144;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t*) { return dot_q4_K(v, y, kbx, iqs); }
};

template <int T>
inline float row_dot(const sycl::sub_group& sg, const uint8_t* row, const block_q8_1* x, int nb, int lane, const uint32_t* grid) {
    using F = Fmt<T>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs, grid);
    }
    return sycl::reduce_over_group(sg, s, sycl::plus<float>());
}

int row_bytes(int type, int64_t n) {
    switch (type) {
        case T_IQ3_S: return (int) (n / 256 * Fmt<T_IQ3_S>::bytes);
        case T_Q6_K: return (int) (n / 256 * Fmt<T_Q6_K>::bytes);
        case T_Q4_K: return (int) (n / 256 * Fmt<T_Q4_K>::bytes);
        default: return 0;
    }
}

// the pairs taken: gate | up IQ3_S or Q4_K, down IQ3_S or Q6_K
bool pair_ok(int gu, int d) { return (gu == T_IQ3_S || gu == T_Q4_K) && (d == T_IQ3_S || d == T_Q6_K); }

size_t align256(size_t b) { return (b + 255) & ~(size_t) 255; }

template <int TG>
void launch_gu(sycl::queue& q, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
               const int32_t* ent_tok, int64_t groups, const block_q8_1* x, float* h, float limit) {
    const int64_t tiles = n_ff / ROWS;
    const int nb = (int) (n_embd / 256);
    const size_t gu_row = (size_t) row_bytes(TG, n_embd);
    const size_t up_off = (size_t) n_ff * gu_row;
    const int64_t xb = n_embd / 32;   // q8_1 blocks a token
    q.submit([&](sycl::handler& hd) {
    sycl::local_accessor<uint32_t, 1> lut(sycl::range<1>(TG == T_IQ3_S ? 512 : 1), hd);
    hd.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * tiles * ROWS * LANES), sycl::range<1>(ROWS * LANES)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(LANES)]] {
#pragma clang fp contract(off)
        const int64_t wg = it.get_group(0);
        const int64_t gi = wg / tiles;
        if (gi >= *n_groups) return;   // uniform in the work-group
        if constexpr (TG == T_IQ3_S) {
            for (int i = (int) it.get_local_linear_id(); i < 512; i += ROWS * LANES) lut[i] = iq3s_grid[i];
            sycl::group_barrier(it.get_group());
        }
        const uint32_t* grid = lut.template get_multi_ptr<sycl::access::decorated::no>().get();
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_id()[0];
        const int64_t r = (wg % tiles) * ROWS + sg.get_group_id()[0];
        const uint8_t* blob = (const uint8_t*) grp_ptr[gi];
        const uint8_t* gate = blob + r * gu_row;
        const uint8_t* up = blob + up_off + r * gu_row;
        for (int e = grp_start[gi]; e < grp_start[gi + 1]; ++e) {
            const block_q8_1* xe = x + (int64_t) ent_tok[e] * xb;
            const float gs = row_dot<TG>(sg, gate, xe, nb, lane, grid);
            const float us = row_dot<TG>(sg, up, xe, nb, lane, grid);
            if (lane == 0) {
                // GLM's clamped SwiGLU: silu(min(g, limit)) * clamp(u, -limit, limit) (limit > 0), as the shared kernels
                const float g = limit > 0.0f ? sycl::fmin(gs, limit) : gs;
                const float u = limit > 0.0f ? sycl::fmin(sycl::fmax(us, -limit), limit) : us;
                h[(int64_t) e * n_ff + r] = (g / (1.0f + sycl::native::exp(-g))) * u;
            }
        }
    });
    });
}

// h [entries, n_ff] -> q8_1 blocks (ggml's quantize_q8_1: d = amax / 127, q = round(x / d), ds = (d, sum)), a
// work-item a block; the entries of grp_start[0] .. grp_start[*n_groups]
void launch_q8(sycl::queue& q, const int32_t* grp_start, const int32_t* n_groups, int64_t entries, int64_t n_ff, const float* h, block_q8_1* hq) {
    const int64_t per = n_ff / 32;
    q.parallel_for(sycl::range<1>(entries * per), [=](sycl::id<1> id) {
        const int64_t b = id[0], e = b / per;
        if (e < grp_start[0] || e >= grp_start[*n_groups]) return;
        const float* v = h + b * 32;
        float amax = 0.0f, sum = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = sycl::fmax(amax, sycl::fabs(v[j]));
            sum += v[j];
        }
        constexpr float hmax = 65504.0f;   // the scale and sum kept finite in half (Strata's q8_1_finite)
        const float d = sycl::fabs(amax / 127.0f) > hmax ? hmax : amax / 127.0f;
        block_q8_1& y = hq[b];
        for (int j = 0; j < 32; ++j) {
            float qv = amax == 0.0f ? 0.0f : sycl::round(v[j] / d);
            qv = qv > 127.0f ? 127.0f : qv < -127.0f ? -127.0f : qv;
            y.qs[j] = (int8_t) qv;
        }
        const float s = sycl::fabs(sum) > hmax ? sycl::copysign(hmax, sum) : sum;
        y.ds = sycl::half2((sycl::half) d, (sycl::half) s);
    });
}

template <int TD>
void launch_down(sycl::queue& q, int gu_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                 const int32_t* ent_dst, int64_t groups, const block_q8_1* hq, float* out) {
    const int64_t tiles = n_embd / ROWS;
    const int nb = (int) (n_ff / 256);
    const size_t d_row = (size_t) row_bytes(TD, n_ff);
    const size_t down_off = 2 * (size_t) n_ff * (size_t) row_bytes(gu_type, n_embd);
    const int64_t hb = n_ff / 32;
    q.submit([&](sycl::handler& hd) {
    sycl::local_accessor<uint32_t, 1> lut(sycl::range<1>(TD == T_IQ3_S ? 512 : 1), hd);
    hd.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * tiles * ROWS * LANES), sycl::range<1>(ROWS * LANES)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(LANES)]] {
        const int64_t wg = it.get_group(0);
        const int64_t gi = wg / tiles;
        if (gi >= *n_groups) return;   // uniform in the work-group
        if constexpr (TD == T_IQ3_S) {
            for (int i = (int) it.get_local_linear_id(); i < 512; i += ROWS * LANES) lut[i] = iq3s_grid[i];
            sycl::group_barrier(it.get_group());
        }
        const uint32_t* grid = lut.template get_multi_ptr<sycl::access::decorated::no>().get();
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_id()[0];
        const int64_t r = (wg % tiles) * ROWS + sg.get_group_id()[0];
        const uint8_t* row = (const uint8_t*) grp_ptr[gi] + down_off + r * d_row;
        for (int e = grp_start[gi]; e < grp_start[gi + 1]; ++e) {
            const float s = row_dot<TD>(sg, row, hq + (int64_t) e * hb, nb, lane, grid);
            if (lane == 0) out[(int64_t) ent_dst[e] * n_embd + r] = s;
        }
    });
    });
}

// IQ3_S in mmv.cpp's layout (the 16-lane kernels above read 145 GB/s of a token's experts): a row a 32-lane sub-group,
// lane g decoding group g (8 values) of each block once into int8 and dotting it with every entry's q8_1 values of
// that group (q8_1 block g / 4, values (g % 4) * 8 ..), the grid in local memory. A lane's sums run block by block in
// the same order whatever the entry count, so rows do not depend on how many entries a launch has.
constexpr int SG3 = 32, RPG3 = 4;   // lanes a row, rows a work-group
constexpr int MAXE = 8;             // entries a pass of a row (an expert appears once a token at decode widths)

inline float h2f(const uint8_t* p) { return (float) sycl::bit_cast<sycl::half>((uint16_t) (p[0] | (p[1] << 8))); }

// group g of an IQ3_S block as signed integers w[8] and its scale (ggml's dequantize_row_iq3_s: value = db * w)
inline float iq3s_igroup(const uint8_t* b, int g, int* w, const uint32_t* grid) {
    const int ib32 = g >> 2, l = g & 3;
    const uint8_t* qs = b + 2 + 8 * ib32;
    const uint32_t qh = b[66 + ib32];
    const uint32_t signs = b[74 + 4 * ib32 + l];
    const uint8_t* g1 = (const uint8_t*) (grid + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
    const uint8_t* g2 = (const uint8_t*) (grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = ((signs >> j) & 1u) ? -(int) g1[j] : (int) g1[j];
        w[j + 4] = ((signs >> (j + 4)) & 1u) ? -(int) g2[j] : (int) g2[j];
    }
    return h2f(b) * (float) (1 + 2 * ((b[106 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xf));
}

// lane's share of a row against one entry's q8_1 group: (db * d8) * sum(w * q)
inline float iq3s_term(const int* w, float db, const block_q8_1* xb, int lane) {
    const block_q8_1& b = xb[lane >> 2];
    const int8_t* q = b.qs + (lane & 3) * 8;
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += w[j] * (int) q[j];
    return (db * (float) b.ds[0]) * (float) s;
}

bool iq3s_old() {
    static const bool old = [] { const char* v = std::getenv("NS_GLM_MOE_IQ3S"); return v && std::string(v) == "old"; }();
    return old;
}

// NE entries a pass of a row (1: a one-token pass, every group one entry; MAXE otherwise - fewer registers held at 1),
// KU blocks decoded before they are multiplied (their loads in flight together). Neither changes a lane's order of sums
template <int NE, int KU>
void gu_iq3s(sycl::queue& q, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
             const int32_t* ent_tok, int64_t groups, const block_q8_1* x, float* h, float limit) {
    const int64_t tiles = n_ff / RPG3;
    const int64_t nb = n_embd / 256, gu_row = nb * 110, up_off = n_ff * gu_row, xb = n_embd / 32;
    q.submit([&](sycl::handler& hd) {
    sycl::local_accessor<uint32_t, 1> lut(sycl::range<1>(512), hd);
    hd.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * tiles * RPG3 * SG3), sycl::range<1>(RPG3 * SG3)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG3)]] {
#pragma clang fp contract(off)
        const int64_t wg = it.get_group(0);
        const int64_t gi = wg / tiles;
        if (gi >= *n_groups) return;   // uniform in the work-group
        for (int i = (int) it.get_local_linear_id(); i < 512; i += RPG3 * SG3) lut[i] = iq3s_grid[i];
        sycl::group_barrier(it.get_group());
        const uint32_t* grid = lut.template get_multi_ptr<sycl::access::decorated::no>().get();
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_id()[0];
        const int64_t r = (wg % tiles) * RPG3 + sg.get_group_id()[0];
        const uint8_t* gate = (const uint8_t*) grp_ptr[gi] + r * gu_row;
        const uint8_t* up = gate + up_off;
        const int e1 = grp_start[gi + 1];
        for (int c0 = grp_start[gi]; c0 < e1; c0 += NE) {
            const int ne = e1 - c0 < NE ? e1 - c0 : NE;
            const block_q8_1* xe[NE];
#pragma unroll
            for (int c = 0; c < NE; ++c) xe[c] = x + (int64_t) ent_tok[c < ne ? c0 + c : c0] * xb;
            float ag[NE] = {}, au[NE] = {};
            for (int64_t k0 = 0; k0 < nb; k0 += KU) {
                int wg8[KU][8], wu8[KU][8];
                float dg[KU], du[KU];
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    const int64_t k = k0 + u < nb ? k0 + u : nb - 1;
                    dg[u] = iq3s_igroup(gate + k * 110, lane, wg8[u], grid);
                    du[u] = iq3s_igroup(up + k * 110, lane, wu8[u], grid);
                }
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    if (k0 + u >= nb) break;
#pragma unroll
                    for (int c = 0; c < NE; ++c) {
                        if (c >= ne) break;
                        ag[c] = ag[c] + iq3s_term(wg8[u], dg[u], xe[c] + (k0 + u) * 8, lane);
                        au[c] = au[c] + iq3s_term(wu8[u], du[u], xe[c] + (k0 + u) * 8, lane);
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < NE; ++c) {
                if (c >= ne) break;
                const float gs = sycl::reduce_over_group(sg, ag[c], sycl::plus<float>());
                const float us = sycl::reduce_over_group(sg, au[c], sycl::plus<float>());
                if (lane == 0) {
                    // GLM's clamped SwiGLU, as launch_gu
                    const float g = limit > 0.0f ? sycl::fmin(gs, limit) : gs;
                    const float u = limit > 0.0f ? sycl::fmin(sycl::fmax(us, -limit), limit) : us;
                    h[(int64_t) (c0 + c) * n_ff + r] = (g / (1.0f + sycl::native::exp(-g))) * u;
                }
            }
        }
    });
    });
}

template <int NE, int KU>
void down_iq3s(sycl::queue& q, int gu_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
               const int32_t* ent_dst, int64_t groups, const block_q8_1* hq, float* out) {
    const int64_t tiles = n_embd / RPG3;
    const int64_t nb = n_ff / 256, d_row = nb * 110, hb = n_ff / 32;
    const size_t down_off = 2 * (size_t) n_ff * (size_t) row_bytes(gu_type, n_embd);
    q.submit([&](sycl::handler& hd) {
    sycl::local_accessor<uint32_t, 1> lut(sycl::range<1>(512), hd);
    hd.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * tiles * RPG3 * SG3), sycl::range<1>(RPG3 * SG3)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG3)]] {
#pragma clang fp contract(off)
        const int64_t wg = it.get_group(0);
        const int64_t gi = wg / tiles;
        if (gi >= *n_groups) return;   // uniform in the work-group
        for (int i = (int) it.get_local_linear_id(); i < 512; i += RPG3 * SG3) lut[i] = iq3s_grid[i];
        sycl::group_barrier(it.get_group());
        const uint32_t* grid = lut.template get_multi_ptr<sycl::access::decorated::no>().get();
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = (int) sg.get_local_id()[0];
        const int64_t r = (wg % tiles) * RPG3 + sg.get_group_id()[0];
        const uint8_t* row = (const uint8_t*) grp_ptr[gi] + down_off + r * d_row;
        const int e1 = grp_start[gi + 1];
        for (int c0 = grp_start[gi]; c0 < e1; c0 += NE) {
            const int ne = e1 - c0 < NE ? e1 - c0 : NE;
            float a[NE] = {};
            for (int64_t k0 = 0; k0 < nb; k0 += KU) {
                int w8[KU][8];
                float d[KU];
#pragma unroll
                for (int u = 0; u < KU; ++u) d[u] = iq3s_igroup(row + (k0 + u < nb ? k0 + u : nb - 1) * 110, lane, w8[u], grid);
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    if (k0 + u >= nb) break;
#pragma unroll
                    for (int c = 0; c < NE; ++c) {
                        if (c >= ne) break;
                        a[c] = a[c] + iq3s_term(w8[u], d[u], hq + (int64_t) (c0 + c) * hb + (k0 + u) * 8, lane);
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < NE; ++c) {
                if (c >= ne) break;
                const float s = sycl::reduce_over_group(sg, a[c], sycl::plus<float>());
                if (lane == 0) out[(int64_t) ent_dst[c0 + c] * n_embd + r] = s;
            }
        }
    });
    });
}

// blocks decoded a step (NS_GLM_MOE_KU: 1, 2 or 4; for measuring)
int moe_ku() {
    static const int ku = [] { const char* v = std::getenv("NS_GLM_MOE_KU"); const int n = v ? std::atoi(v) : 2; return n == 1 || n == 4 ? n : 2; }();
    return ku;
}

// one entry a group (entries == groups: a one-token pass) takes the NE = 1 kernels
void launch_gu_iq3s(sycl::queue& q, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_tok, int64_t groups, int64_t entries, const block_q8_1* x, float* h, float limit) {
#define NS_GU(NE, KU) if (moe_ku() == KU) return gu_iq3s<NE, KU>(q, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_tok, groups, x, h, limit);
    if (entries == groups) { NS_GU(1, 1) NS_GU(1, 2) NS_GU(1, 4) }
    NS_GU(MAXE, 1) NS_GU(MAXE, 2) NS_GU(MAXE, 4)
#undef NS_GU
}

void launch_down_iq3s(sycl::queue& q, int gu_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                      const int32_t* ent_dst, int64_t groups, int64_t entries, const block_q8_1* hq, float* out) {
#define NS_DN(NE, KU) if (moe_ku() == KU) return down_iq3s<NE, KU>(q, gu_type, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_dst, groups, hq, out);
    if (entries == groups) { NS_DN(1, 1) NS_DN(1, 2) NS_DN(1, 4) }
    NS_DN(MAXE, 1) NS_DN(MAXE, 2) NS_DN(MAXE, 4)
#undef NS_DN
}

}  // namespace

extern "C" {

int ns_llm_glm5next_moe_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    return pair_ok(gu_type, d_type) && n_embd % 256 == 0 && n_ff % 256 == 0 && n_embd % ROWS == 0 && n_ff % ROWS == 0 ? 1 : 0;
}

size_t ns_llm_glm5next_moe_scratch_bytes(int64_t entries, int64_t n_ff) {
    return align256((size_t) entries * (size_t) n_ff * sizeof(float)) + align256((size_t) entries * (size_t) (n_ff / 32) * sizeof(block_q8_1));
}

int ns_llm_glm5next_moe(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t groups, int64_t entries, const void* x_q8_1,
                        void* scratch, float* out, float limit, int phase) {
    NS_TRY
    if (!ns_llm_glm5next_moe_supported(gu_type, d_type, n_embd, n_ff))
        return ns_fail("ns_llm_glm5next_moe: types " + std::to_string(gu_type) + "/" + std::to_string(d_type) + " not supported");
    if (phase < 0 || phase > 2) return ns_fail("ns_llm_glm5next_moe: phase " + std::to_string(phase));
    if (groups <= 0 || entries <= 0) return 0;
    auto& q = g->q;
    float* h = (float*) scratch;
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + align256((size_t) entries * (size_t) n_ff * sizeof(float)));
    const block_q8_1* x = (const block_q8_1*) x_q8_1;
    if (phase != 2) {
        if (gu_type == T_IQ3_S && !iq3s_old()) launch_gu_iq3s(q, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_tok, groups, entries, x, h, limit);
        else if (gu_type == T_IQ3_S) launch_gu<T_IQ3_S>(q, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_tok, groups, x, h, limit);
        else launch_gu<T_Q4_K>(q, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_tok, groups, x, h, limit);
        launch_q8(q, grp_start, n_groups, entries, n_ff, h, hq);
    }
    if (phase != 1) {
        if (d_type == T_IQ3_S && !iq3s_old()) launch_down_iq3s(q, gu_type, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_dst, groups, entries, hq, out);
        else if (d_type == T_IQ3_S) launch_down<T_IQ3_S>(q, gu_type, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_dst, groups, hq, out);
        else launch_down<T_Q6_K>(q, gu_type, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_dst, groups, hq, out);
    }
    return 0;
    NS_CATCH
}

// down alone, from q8_1 activations the caller placed (hq [entries, n_ff / 32] blocks): the shared grouped kernels'
// phase 1 (gate | up and the activation, faster than this file's) leaves them in its scratch, then this takes the
// down projection those kernels do not
int ns_llm_glm5next_moe_down(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                             const int32_t* n_groups, const int32_t* ent_dst, int64_t groups, int64_t entries, const void* hq, float* out) {
    NS_TRY
    if (!ns_llm_glm5next_moe_supported(gu_type, d_type, n_embd, n_ff))
        return ns_fail("ns_llm_glm5next_moe_down: types " + std::to_string(gu_type) + "/" + std::to_string(d_type) + " not supported");
    if (groups <= 0 || entries <= 0) return 0;
    const block_q8_1* h = (const block_q8_1*) hq;
    if (d_type == T_IQ3_S && !iq3s_old()) launch_down_iq3s(g->q, gu_type, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_dst, groups, entries, h, out);
    else if (d_type == T_IQ3_S) launch_down<T_IQ3_S>(g->q, gu_type, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_dst, groups, h, out);
    else launch_down<T_Q6_K>(g->q, gu_type, n_embd, n_ff, grp_ptr, grp_start, n_groups, ent_dst, groups, h, out);
    return 0;
    NS_CATCH
}

}  // extern "C"
