// mmv.cpp: decode-width products (1..8 tokens) of the dense matrices the REAP-50 IQ3_M file stores in IQ3_S - its
// attention, shared expert, dense FFN and MTP. The shared decode kernel for IQ3_S (Strata's iq_mmvq, ggml's CUDA
// vec_dot) gathers each 4-value group's grid entry from the global table: 118 GB/s on a B70 against 432 for Q4_K, and
// those matrices are most of what a token reads (`nextsycl kernels`).
//
// deepseek41's scheme (its silo's std.cpp): a 256-value block is 32 groups of 8; a sub-group of 32 lanes works a row, lane
// g decoding group g of each block (a lane's 8 weights meet 8 contiguous activations: coalesced), RPS rows at once so
// their loads are in flight together, the grid staged in local memory by each work-group. The activations are float32
// (no Q8_1 rounding). A column's sum runs in an order fixed by the kernel alone (the same for 1 or 8 columns), so a
// verify pass's rows equal one-token passes bit for bit. The IQ3_S format (ns_gguf.h) gives the weights.
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "ns_gguf.h"

namespace {

constexpr int T_IQ3_S = 21;
constexpr int SB = 110;   // bytes of an IQ3_S block (256 values)

inline uint32_t ld16(const uint8_t* p) { return (uint32_t) p[0] | ((uint32_t) p[1] << 8); }
inline float h2f(const uint8_t* p) { return (float) sycl::bit_cast<sycl::half>((uint16_t) ld16(p)); }
inline float sgn(uint32_t signs, int j) { return ((signs >> j) & 1u) ? -1.f : 1.f; }

// group g (8 values) of an IQ3_S block: d; qs[64]; qh[8]; signs[32]; scales[4] (ggml's dequantize_row_iq3_s)
inline void iq3s_group(const uint8_t* b, int g, float* w, const uint32_t* grid) {
    const int ib32 = g >> 2, l = g & 3;
    const uint8_t* qs = b + 2 + 8 * ib32;
    const uint32_t qh = b[66 + ib32];
    const float db = h2f(b) * (1 + 2 * ((b[106 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xf));
    const uint32_t signs = b[74 + 4 * ib32 + l];
    const uint8_t* g1 = (const uint8_t*) (grid + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
    const uint8_t* g2 = (const uint8_t*) (grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = db * g1[j] * sgn(signs, j);
        w[j + 4] = db * g2[j] * sgn(signs, j + 4);
    }
}

// y[c * rows + r] = W[r, :] . x[c * cols ..] for c < nc (nc <= NC); W rows * (cols / 256) blocks
template <int RPS, int KU, int NC>
void mmv_iq3s(sycl::queue& q, const uint8_t* w, const float* x, float* y, int64_t rows, int64_t cols, int nc) {
    constexpr int SG = 32, RPG = 4;
    const int64_t rgs = (rows + RPG * RPS - 1) / (RPG * RPS), nb = cols / 256, rowb = nb * SB;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> lut(sycl::range<1>(512), h);
        h.parallel_for(sycl::nd_range<1>(rgs * RPG * SG, RPG * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
#pragma clang fp contract(off)
            const auto sg = it.get_sub_group();
            for (int i = (int) it.get_local_linear_id(); i < 512; i += RPG * SG) lut[i] = iq3s_grid[i];
            sycl::group_barrier(it.get_group());
            const uint32_t* grid = lut.template get_multi_ptr<sycl::access::decorated::no>().get();
            const int64_t r0 = ((int64_t) it.get_group(0) * RPG + sg.get_group_id()[0]) * RPS;
            const int lane = (int) sg.get_local_id()[0];
            if (r0 >= rows) return;
            float acc[RPS][NC] = {};
            for (int64_t k0 = 0; k0 < nb; k0 += KU) {
                // KU blocks of RPS rows decoded first (their loads in flight together), then multiplied
                float wv[KU][RPS][8];
#pragma unroll
                for (int u = 0; u < KU; ++u) {
                    const int64_t k = k0 + u < nb ? k0 + u : nb - 1;
#pragma unroll
                    for (int j = 0; j < RPS; ++j) {
                        const int64_t r = r0 + j < rows ? r0 + j : rows - 1;
                        iq3s_group(w + r * rowb + k * SB, lane, wv[u][j], grid);
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    if (c >= nc) break;
#pragma unroll
                    for (int u = 0; u < KU; ++u) {
                        if (k0 + u >= nb) break;
                        const sycl::float4* xv = reinterpret_cast<const sycl::float4*>(x + (int64_t) c * cols + (k0 + u) * 256 + lane * 8);
                        const sycl::float4 a = xv[0], b = xv[1];
                        // explicit fused steps: the same arithmetic in every compiled width (no contraction choices)
#pragma unroll
                        for (int j = 0; j < RPS; ++j) {
                            float s = acc[j][c];
                            s = sycl::fma(wv[u][j][0], a.x(), s);
                            s = sycl::fma(wv[u][j][1], a.y(), s);
                            s = sycl::fma(wv[u][j][2], a.z(), s);
                            s = sycl::fma(wv[u][j][3], a.w(), s);
                            s = sycl::fma(wv[u][j][4], b.x(), s);
                            s = sycl::fma(wv[u][j][5], b.y(), s);
                            s = sycl::fma(wv[u][j][6], b.z(), s);
                            s = sycl::fma(wv[u][j][7], b.w(), s);
                            acc[j][c] = s;
                        }
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                if (c >= nc) break;
#pragma unroll
                for (int j = 0; j < RPS; ++j) {
                    const float s = sycl::reduce_over_group(sg, acc[j][c], sycl::plus<float>());
                    if (lane == 0 && r0 + j < rows) y[(int64_t) c * rows + r0 + j] = s;
                }
            }
        });
    });
}

// rows a sub-group and blocks a step (NS_GLM_MMV_RPS / NS_GLM_MMV_KU: 1, 2 or 4; for measuring)
int knob(const char* name, int def) {
    const char* v = std::getenv(name);
    const int n = v ? std::atoi(v) : def;
    return n == 1 || n == 2 || n == 4 ? n : def;
}

template <int NC>
void dispatch(sycl::queue& q, const uint8_t* w, const float* x, float* y, int64_t rows, int64_t cols, int nc) {
    static const int rps = knob("NS_GLM_MMV_RPS", 1), ku = knob("NS_GLM_MMV_KU", 2);   // 1 / 2: 371 GB/s on a B70 (2 / 2: 301, 2 / 4: 363, 4 / 4: 317; the shared kernel 101-150)
#define NS_MMV(R, K) if (rps == R && ku == K) return mmv_iq3s<R, K, NC>(q, w, x, y, rows, cols, nc);
    NS_MMV(1, 1) NS_MMV(1, 2) NS_MMV(1, 4) NS_MMV(2, 1) NS_MMV(2, 2) NS_MMV(2, 4) NS_MMV(4, 1) NS_MMV(4, 2) NS_MMV(4, 4)
#undef NS_MMV
}

}  // namespace

extern "C" {

int ns_llm_glm5next_mmv_supported(int type) { return type == T_IQ3_S ? 1 : 0; }

int ns_llm_glm5next_mmv(ns_gpu* g, int type, const void* w, const float* x, float* y, int64_t n_in, int64_t n_out, int64_t ncols) {
    NS_TRY
    if (!ns_llm_glm5next_mmv_supported(type)) return ns_fail("ns_llm_glm5next_mmv: type " + std::to_string(type) + " not supported");
    if (ncols < 1 || ncols > 8) return ns_fail("ns_llm_glm5next_mmv: 1..8 columns");
    if (n_in % 256 != 0) return ns_fail("ns_llm_glm5next_mmv: n_in a multiple of 256");
    const auto* W = (const uint8_t*) w;
    const int nc = (int) ncols;
    // the column count rounded up to a compiled width; a column's arithmetic is the same in each
    if (nc == 1) dispatch<1>(g->q, W, x, y, n_out, n_in, nc);
    else if (nc <= 2) dispatch<2>(g->q, W, x, y, n_out, n_in, nc);
    else if (nc <= 4) dispatch<4>(g->q, W, x, y, n_out, n_in, nc);
    else dispatch<8>(g->q, W, x, y, n_out, n_in, nc);
    return 0;
    NS_CATCH
}

}  // extern "C"
