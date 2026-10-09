// fused.cpp: an expert's gate | up or down for a prompt chunk's tokens, its 2-bit weights decoded straight into the
// matrix engine (joint_matrix on the XMX units) - no fp16 copy of the expert in memory. C[M, N] (f32) = X[M, K] (f16)
// . W[N, K]^T, W in IQ2_XXS (gate | up: gate's rows then up's, a slot's first two matrices) or Q2_K (down). A sub-group's lane n holds column n
// of B (16 consecutive k of weight row n) - the layout probed on Xe2 - so each lane decodes its own 16 weights.
// Measured alone, the weights cold (gate | up 4096 x 4096): B70 117 us vs 202 for expanding to fp16 + oneMKL at 64
// tokens, 135 / 214 at 128; down (4096 x 2048), 8 row tiles: B70 39 us at 64 tokens, 40 at 128, 55 at 256, B65 47 /
// 66 / 119 (oneMKL alone on the expanded weights: 18 / 25 / 90, 31 / 62 / 61).
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ns.h"
#include "glm.h"
#include "ns_internal.hpp"
#include "ggml-common.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <map>
#include <mutex>

namespace jm = sycl::ext::oneapi::experimental::matrix;

namespace {
constexpr int TM = 8, TN = 16, TK = 16, SG = 16, NSG = 8;

// The 32 k from k of a weight row (k a multiple of 32), as two B columns of 16: the dequantizers' arithmetic
struct Iq2xxs {  // gate | up: a 32-group's 8 bytes - its 4 grid indices, the scale and the signs (iq_kernels' dq_iq2_xxs)
    static constexpr int BYTES = sizeof(block_iq2_xxs);
    static void decode(const void* row, int64_t k, sycl::half (&v)[2][16]) {
        const block_iq2_xxs* b = (const block_iq2_xxs*) row + k / 256;
        const sycl::uint2 q8 = *reinterpret_cast<const sycl::uint2*>(b->qs + 4 * ((k % 256) / 32));
        const uint32_t aux32 = q8.y();
        const float d = (float) b->d * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t s7 = (aux32 >> 7 * il) & 127;
            const uint32_t signs = s7 | ((sycl::popcount(s7) & 1u) << 7);
            const uint64_t g = iq2xxs_grid[(q8.x() >> 8 * il) & 0xff];
#pragma unroll
            for (int j = 0; j < 8; ++j) v[il / 2][(il % 2) * 8 + j] = sycl::half(d * (float) ((g >> (8 * j)) & 0xff) * ((signs >> j) & 1 ? -1.f : 1.f));
        }
    }
};
struct Q2k {  // down: 16 k share a scale byte and a 2-bit shift of 16 consecutive quant bytes (ggml's dequantize_row_q2_K)
    static constexpr int BYTES = sizeof(block_q2_K);
    static void decode(const void* row, int64_t k, sycl::half (&v)[2][16]) {
        const block_q2_K* b = (const block_q2_K*) row + k / 256;
        const int n = (k % 256) / 128, j = (k % 128) / 32;
        const float d = (float) b->dm[0], dm = (float) b->dm[1];
        const sycl::uint4* qp = reinterpret_cast<const sycl::uint4*>(b->qs + 32 * n);
        const uint16_t sc2 = *reinterpret_cast<const uint16_t*>(b->scales + 8 * n + 2 * j);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const sycl::uint4 qq = qp[hf];
            const uint32_t sc = (sc2 >> (8 * hf)) & 0xff;
            const float dl = d * (sc & 0xF), ml = dm * (sc >> 4);
#pragma unroll
            for (int l = 0; l < 16; ++l) v[hf][l] = sycl::half(dl * (float) ((qq[l / 4] >> (8 * (l % 4) + 2 * j)) & 3) - ml);
        }
    }
};

// RM row tiles (8 tokens each) a sub-group, 32 k a step: lane n decodes its weight row's 32 k into two B tiles
template <int RM, class Dec>
void fused(sycl::queue& q, const sycl::half* A, const uint8_t* W, float* out, int64_t M, int64_t N, int64_t K) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<(RM > 4 ? 256 : 128)>};
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>(M / (RM * TM), N / (TN * NSG) * NSG * SG), sycl::range<2>(1, NSG * SG)), props,
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int lane = sg.get_local_id()[0];
        const int64_t m0 = it.get_group(0) * RM * TM;
        const int64_t n0 = (it.get_group(1) * NSG + it.get_local_id(1) / SG) * TN;
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, TM, TN> acc[RM];
        for (int i = 0; i < RM; ++i) jm::joint_matrix_fill(sg, acc[i], 0.0f);
        auto gA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(A);
        const uint8_t* wr = W + (n0 + lane) * (K / 256) * Dec::BYTES;
        for (int64_t k = 0; k < K; k += 32) {
            sycl::half v[2][16];
            Dec::decode(wr, k, v);
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                // lane n holds column n of B (16 consecutive k of weight row n): the layout probed on Xe2
                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, TK, TN, jm::layout::ext_intel_packed> bm;
                jm::joint_matrix_fill(sg, bm, sycl::half(0));
                int e = 0;
                jm::joint_matrix_apply(sg, bm, [&](sycl::half& h) { h = v[hh][e++]; });
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, TM, TK, jm::layout::row_major> a;
                    jm::joint_matrix_load(sg, a, gA + (m0 + i * TM) * K + k + hh * 16, K);
                    jm::joint_matrix_mad(sg, acc[i], a, bm, acc[i]);
                }
            }
        }
        for (int i = 0; i < RM; ++i)
            jm::joint_matrix_store(sg, acc[i],
                                   sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(out + (m0 + i * TM) * N + n0),
                                   N, jm::layout::row_major);
    });
}

// M a multiple of 32 (of 64 above 64: those take 8 row tiles a sub-group, in the large register file)
template <class Dec>
int run(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K, const char* name) {
    const bool wide = M > 64;
    if (M % (wide ? 64 : 32) != 0 || N % (TN * NSG) != 0 || K % 256 != 0) return ns_fail(name);
    const sycl::half* A = (const sycl::half*) x;
    if (wide) fused<8, Dec>(g->q, A, (const uint8_t*) w, out, M, N, K);
    else fused<4, Dec>(g->q, A, (const uint8_t*) w, out, M, N, K);
    return 0;
}
}  // namespace

// ---- gate | up in ESIMD (xmx::dpas): one work-group over all of an expert's rows (16 a thread: two 8-row tiles),
// its threads sharing a 16-column slice of N. Per 256-k block (one IQ2_XXS block a weight row) the threads decode
// its 8 groups between them - into dpas's B layout (VNNI, built with vector selects: no joint_matrix fill) in local
// memory - then each loads every tile for its own rows: every weight decoded once a call. The grid table in local
// memory. Measured alone (reference/esimd-fused.cpp, 4096 x 4096, cold): B70 103 / 117 / 157 us at 64 / 128 / 256
// tokens (joint_matrix's 117 / 135 / 200), B65 165 / 213 / 327 (206 / 224 / -); past 256 (B70) or 128 (B65)
// expanding + oneMKL wins (ns_moe_esimd_gu_max).
namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace {
constexpr int EX = 16, DEPTH = 8, RC = 8, KS = 16, RM2 = 2;

template <int T>
void esimd_gu(sycl::queue& q, const sycl::half* A, const block_iq2_xxs* W, float* C, int M, int N, int K, const uint32_t* grid32) {
    constexpr int GRID = 2048, TILE = KS * EX * 2, BUF = 16 * TILE;   // bytes: the table, a B tile, a block's tiles
    using half = sycl::half;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((N / EX) * T), sycl::range<1>(T)), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<GRID + 2 * BUF>();
        const int t = it.get_local_id(0);
        const int n0 = it.get_group(0) * EX, m0 = t * RC * RM2;
        for (int o = t * 256; o < GRID; o += T * 256) es::slm_block_store<uint32_t, 64>(o, es::block_load<uint32_t, 64>(grid32 + o / 4));
        es::barrier();
        es::simd<float, RC * EX> acc[RM2];
        for (int i = 0; i < RM2; ++i) acc[i] = 0.f;
        const uint32_t rb = (uint32_t) ((K / 256) * sizeof(block_iq2_xxs));
        const uint8_t* wb = (const uint8_t*) W;
        es::simd<uint32_t, EX> rows(0, 1);
        rows = rows * rb + (uint32_t) (n0 * rb);
        for (int kb = 0; kb < K; kb += 256) {
            const uint32_t buf = GRID + ((kb / 256) % 2) * BUF;
            const es::simd<uint32_t, EX> off = rows + (uint32_t) ((kb / 256) * sizeof(block_iq2_xxs));
            es::simd<uint32_t, EX> dw = es::gather<uint32_t, EX>((const uint32_t*) wb, off);
            es::simd<uint16_t, EX> dbits = dw & 0xffffu;
            es::simd<half, EX> dh = dbits.template bit_cast_view<half>();
            const es::simd<float, EX> d0 = dh;
            for (int grp = t; grp < 8; grp += T) {
                es::simd<uint32_t, EX> qx = es::gather<uint32_t, EX>((const uint32_t*) wb, off + (uint32_t) (2 + 8 * grp));
                es::simd<uint32_t, EX> qy = es::gather<uint32_t, EX>((const uint32_t*) wb, off + (uint32_t) (6 + 8 * grp));
                es::simd<float, EX> d = d0 * (0.5f + es::simd<float, EX>(qy >> 28)) * 0.25f;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    es::simd<half, KS * EX> b;   // VNNI: (k, n) at (k / 2) * 32 + n * 2 + k % 2
#pragma unroll
                    for (int g2 = 0; g2 < 2; ++g2) {
                        const int il = hh * 2 + g2;
                        es::simd<uint32_t, EX> gi = (qx >> (8 * il)) & 0xffu;
                        es::simd<uint32_t, EX> glo = es::slm_gather<uint32_t, EX>(gi * 8);
                        es::simd<uint32_t, EX> ghi = es::slm_gather<uint32_t, EX>(gi * 8 + 4);
                        es::simd<uint32_t, EX> s7 = (qy >> (7 * il)) & 127u;
                        es::simd<uint32_t, EX> signs = s7 | ((es::cbit(s7) & 1u) << 7);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            es::simd<uint32_t, EX> byte = ((j < 4 ? glo : ghi) >> (8 * (j % 4))) & 0xffu;
                            es::simd<float, EX> v = d * es::simd<float, EX>(byte);
                            v.merge(-v, ((signs >> j) & 1u) != 0);
                            const int kk = g2 * 8 + j;
                            b.template select<EX, 2>((kk / 2) * 32 + (kk % 2)) = v;
                        }
                    }
                    const uint32_t at = buf + (uint32_t) ((2 * grp + hh) * TILE);
                    es::slm_block_store<half, 128>(at, b.template select<128, 1>(0));
                    es::slm_block_store<half, 128>(at + 256, b.template select<128, 1>(128));
                }
            }
            es::barrier();
#pragma unroll 4
            for (int tile = 0; tile < 16; ++tile) {
                const uint32_t at = buf + (uint32_t) (tile * TILE);
                es::simd<half, KS * EX> b;
                b.template select<128, 1>(0) = es::slm_block_load<half, 128>(at);
                b.template select<128, 1>(128) = es::slm_block_load<half, 128>(at + 256);
#pragma unroll
                for (int i = 0; i < RM2; ++i) {
                    es::simd<half, RC * KS> a = es::load_2d<half, KS, RC>(A, (unsigned) (K * 2 - 1), (unsigned) (M - 1), (unsigned) (K * 2 - 1),
                                                                          kb + tile * 16, m0 + i * RC);
                    acc[i] = xmx::dpas<DEPTH, RC, float, float, half, half>(acc[i], b, a);
                }
            }
        }
        for (int i = 0; i < RM2; ++i)
#pragma unroll
            for (int r = 0; r < RC; ++r)
                es::block_store<float, EX>(C + (size_t) (m0 + i * RC + r) * N + n0, acc[i].template select<EX, 1>(r * EX));
    });
}

// the grid table on each GPU (as 32-bit words: the kernel's block load)
const uint32_t* grid_on(ns_gpu* g) {
    static std::mutex m;
    static std::map<ns_gpu*, uint32_t*> on;
    std::lock_guard<std::mutex> l(m);
    auto it = on.find(g);
    if (it != on.end()) return it->second;
    uint32_t* p = sycl::malloc_device<uint32_t>(512, g->dev, g->ctx);
    g->q.memcpy(p, iq2xxs_grid, 256 * sizeof(uint64_t)).wait();
    on[g] = p;
    return p;
}
}  // namespace

extern "C" {

int ns_moe_fused_gu(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K) {
    NS_TRY
    return run<Iq2xxs>(g, x, w, out, M, N, K, "ns_moe_fused_gu: shapes");
    NS_CATCH
}

// the most rows the ESIMD gate | up takes on this GPU (0: none): 256 on a B70-sized GPU, 128 on smaller ones
int ns_moe_esimd_gu_max(ns_gpu* g) {
    NS_TRY
    return g->dev.get_info<sycl::info::device::max_compute_units>() >= 200 ? 256 : 128;
    NS_CATCH
}

// gate | up in ESIMD (above): M one of 32, 64, 128, 256 (the expert's rows padded)
int ns_moe_fused_gu_esimd(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K) {
    NS_TRY
    if (N % EX != 0 || K % 256 != 0) return ns_fail("ns_moe_fused_gu_esimd: shapes");
    const sycl::half* A = (const sycl::half*) x;
    const block_iq2_xxs* W = (const block_iq2_xxs*) w;
    const uint32_t* grid = grid_on(g);
    switch (M) {
        case 32: esimd_gu<2>(g->q, A, W, out, (int) M, (int) N, (int) K, grid); break;
        case 64: esimd_gu<4>(g->q, A, W, out, (int) M, (int) N, (int) K, grid); break;
        case 128: esimd_gu<8>(g->q, A, W, out, (int) M, (int) N, (int) K, grid); break;
        case 256: esimd_gu<16>(g->q, A, W, out, (int) M, (int) N, (int) K, grid); break;
        default: return ns_fail("ns_moe_fused_gu_esimd: M " + std::to_string(M) + " (32, 64, 128 or 256)");
    }
    return 0;
    NS_CATCH
}

int ns_moe_fused_down(ns_gpu* g, const uint16_t* x, const void* w, float* out, int64_t M, int64_t N, int64_t K) {
    NS_TRY
    return run<Q2k>(g, x, w, out, M, N, K, "ns_moe_fused_down: shapes");
    NS_CATCH
}

}  // extern "C"
