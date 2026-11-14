// dequant.cpp: the glm5next silo's weight expansion (silo.h) for the types the REAP-50 GGUFs store and the shared
// ns_dequant / ns_dequant_f16 do not read: Q4_0, Q5_0, IQ4_NL, IQ3_S, IQ4_XS into float32, Q6_K into half. Plain and
// exact, as glm.cpp's: one work-item a block, the order and arithmetic of ggml-quants.c's dequantize_row_*. It runs
// at load (the MLA's k_b / v_b, the MTP block's eh_proj), per token for the embedding's rows, and for the prompt
// products the half path does not take (a KDA gate's 128-wide f_b / g_b, a chunk below NS_PROMPT_F16_MIN).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>

#include "ns_gguf.h"

#include <cstdint>
#include <string>

namespace {

inline float h2f(sycl::half h) { return (float) h; }

void deq_q4_0(const block_q4_0* x, float* y) {
    const float d = h2f(x->d);
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const int x0 = (x->qs[j] & 0x0F) - 8;
        const int x1 = (x->qs[j] >> 4) - 8;
        y[j] = x0 * d;
        y[j + QK4_0 / 2] = x1 * d;
    }
}

void deq_q5_0(const block_q5_0* x, float* y) {
    const float d = h2f(x->d);
    const uint32_t qh = (uint32_t) x->qh[0] | ((uint32_t) x->qh[1] << 8) | ((uint32_t) x->qh[2] << 16) | ((uint32_t) x->qh[3] << 24);
    for (int j = 0; j < QK5_0 / 2; ++j) {
        const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
        const int32_t x0 = ((x->qs[j] & 0x0F) | xh_0) - 16;
        const int32_t x1 = ((x->qs[j] >> 4) | xh_1) - 16;
        y[j] = x0 * d;
        y[j + QK5_0 / 2] = x1 * d;
    }
}

void deq_iq4_nl(const block_iq4_nl* x, float* y) {
    const float d = h2f(x->d);
    for (int j = 0; j < QK4_NL / 2; ++j) {
        y[j] = d * kvalues_iq4nl[x->qs[j] & 0xf];
        y[j + QK4_NL / 2] = d * kvalues_iq4nl[x->qs[j] >> 4];
    }
}

void deq_iq4_xs(const block_iq4_xs* x, float* y) {
    const uint8_t* qs = x->qs;
    const float d = h2f(x->d);
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j] = dl * kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
}

// one 32-value sub-block: four pairs of 4-value grid entries (9-bit indices: qs, and a bit of qh), a sign bit each
inline void iq3_s_32(const uint8_t* qs, uint8_t qh, const uint8_t* signs, float db, float* y) {
    for (int l = 0; l < 4; ++l) {
        const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)));
        const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
        for (int j = 0; j < 4; ++j) {
            y[j + 0] = db * grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
            y[j + 4] = db * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
        }
        y += 8;
    }
}

void deq_iq3_s(const block_iq3_s* x, float* y) {
    const float d = h2f(x->d);
    const uint8_t* qs = x->qs;
    const uint8_t* qh = x->qh;
    const uint8_t* signs = x->signs;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        const float db1 = d * (1 + 2 * (x->scales[ib32 / 2] & 0xf));
        const float db2 = d * (1 + 2 * (x->scales[ib32 / 2] >> 4));
        iq3_s_32(qs, qh[0], signs, db1, y);
        iq3_s_32(qs + 8, qh[1], signs + 4, db2, y + 32);
        y += 64;
        qs += 16;
        qh += 2;
        signs += 8;
    }
}

// Q6_K as glm.cpp's deq_q6_K (ggml's), each value then rounded to half
void deq_q6_K_h(const block_q6_K* x, sycl::half* y) {
    const float d = h2f(x->d);
    const uint8_t* ql = x->ql;
    const uint8_t* qh = x->qh;
    const int8_t* sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = (sycl::half) (d * sc[is + 0] * q1);
            y[l + 32] = (sycl::half) (d * sc[is + 2] * q2);
            y[l + 64] = (sycl::half) (d * sc[is + 4] * q3);
            y[l + 96] = (sycl::half) (d * sc[is + 6] * q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

template <class Block, int QK, void (*F)(const Block*, float*)>
void dequant_blocks(sycl::queue& q, const void* src, size_t n, float* dst) {
    const size_t nb = n / QK;
    const Block* b = (const Block*) src;
    q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> i) { F(b + i[0], dst + i[0] * QK); });
}

int block_of(int type) {
    switch (type) {
        case 2: case 6: case 20: return 32;
        case 21: case 23: return QK_K;
        default: return 0;
    }
}

}  // namespace

extern "C" {

int ns_llm_glm5next_dequant_supported(int type) { return block_of(type) > 0 ? 1 : 0; }

int ns_llm_glm5next_dequant(ns_gpu* g, int type, const void* src, size_t n, float* dst) {
    NS_TRY
    const int qk = block_of(type);
    if (qk == 0) return ns_fail("ns_llm_glm5next_dequant: ggml type " + std::to_string(type) + " is not one the silo reads");
    if (n % (size_t) qk != 0) return ns_fail("ns_llm_glm5next_dequant: " + std::to_string(n) + " values are not whole blocks of " + std::to_string(qk));
    if (n == 0) return 0;
    auto& q = g->q;
    switch (type) {
        case 2: dequant_blocks<block_q4_0, QK4_0, deq_q4_0>(q, src, n, dst); break;
        case 6: dequant_blocks<block_q5_0, QK5_0, deq_q5_0>(q, src, n, dst); break;
        case 20: dequant_blocks<block_iq4_nl, QK4_NL, deq_iq4_nl>(q, src, n, dst); break;
        case 21: dequant_blocks<block_iq3_s, QK_K, deq_iq3_s>(q, src, n, dst); break;
        case 23: dequant_blocks<block_iq4_xs, QK_K, deq_iq4_xs>(q, src, n, dst); break;
    }
    return 0;
    NS_CATCH
}

int ns_llm_glm5next_dequant_f16_supported(int type) { return type == 14 ? 1 : 0; }

int ns_llm_glm5next_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst) {
    NS_TRY
    if (type != 14 || n < 0 || n % QK_K != 0) return ns_fail("ns_llm_glm5next_dequant_f16: ggml type " + std::to_string(type) + ", " + std::to_string(n) + " values");
    if (n == 0) return 0;
    const size_t nb = (size_t) n / QK_K;
    const block_q6_K* b = (const block_q6_K*) src;
    sycl::half* d = reinterpret_cast<sycl::half*>(dst);
    g->q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> i) { deq_q6_K_h(b + i[0], d + i[0] * QK_K); });
    return 0;
    NS_CATCH
}

}  // extern "C"
