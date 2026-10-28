// attn.cpp: SkinTokens' attention in float32 (silo.h): a copy of UniRig's one-pass kernel (a silo is its engine's own), with
// grouped-query heads. The engine's own kernel (skintokens.cpp) gives a work-group to every query row and re-reads all keys for
// each - slow on the skin VAE's decoder, which answers for 54,000 samples with 12 heads of 64 over 388 keys, once a joint.
// Here heads of 16 to 64 over up to MAX_KEYS keys run in one pass, the scores in registers (flash attention's online softmax).
#include "ns.h"
#include "ns_internal.hpp"
#include "silo.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

// float32 attention in one pass (flash attention's online softmax), for heads of D = Dv = 16, 32, 48 or 64: a sub-group
// of 16 takes 16 QR query rows (a lane QR of them, each row's q and output in registers) and walks the keys 16 at a
// time. Lane j loads key j of the tile (all D of it), and every lane reads it from there: a broadcast of lane j's
// register, which the FMA takes as its operand - no load, no shuffle - so a 16 x 16 tile of scores is D FMAs a
// row and the weighted values D more, the scores never leave the registers.
namespace flash {
constexpr int SG = 16;
// the most keys one pass takes: its running sums over very many drift (UniRig measured 3x the reference's error at 65,536;
// here the 54,000 of the shape and condition encoders moved from 4.4e-5 to 4.5e-5 and 5.0e-5 to 5.04e-5 relative RMS, 3x faster)
constexpr int64_t MAX_KEYS = 65536;

template <int D, int QR>
void run(sycl::queue& q, const float* Q, int64_t q_row, int64_t q_head, const float* K, int64_t k_row, int64_t k_head, const float* V,
         int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv, float scale, bool causal, float* out, int64_t out_row) {
    constexpr int NSG = 4, ROWS = SG * QR;
    const int64_t tiles = (Sq + ROWS - 1) / ROWS;
    const int64_t groups = (tiles + NSG - 1) / NSG;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) (B * H), (size_t) (groups * NSG * SG)), sycl::range<2>(1, NSG * SG)), props,
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int lane = (int) sg.get_local_linear_id();
        const int64_t bh = it.get_group(0), b = bh / H, hh = bh % H;
        const int64_t tile = it.get_group(1) * NSG + sg.get_group_linear_id();
        if (tile >= tiles) return;
        const int64_t r0 = tile * ROWS;
        float qv[QR][D], acc[QR][D], m[QR], l[QR];
        int64_t row[QR];
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            row[i] = r0 + i * SG + lane;
            const int64_t rr = row[i] < Sq ? row[i] : Sq - 1;
            const float* qp = Q + (b * Sq + rr) * q_row + hh * q_head;
#pragma unroll
            for (int d = 0; d < D; d += 4) {
                const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(qp + d);
                qv[i][d] = v.x(); qv[i][d + 1] = v.y(); qv[i][d + 2] = v.z(); qv[i][d + 3] = v.w();
            }
#pragma unroll
            for (int d = 0; d < D; ++d) acc[i][d] = 0.0f;
            m[i] = -std::numeric_limits<float>::infinity();
            l[i] = 0.0f;
        }
        // the keys this tile's rows see (a causal mask: up to its last row)
        const int64_t last = r0 + ROWS - 1 < Sq ? r0 + ROWS - 1 : Sq - 1;
        const int64_t nk = causal ? last + 1 : Skv;
        const float* kb = K + b * Skv * k_row + (hh / (H / Hkv)) * k_head;
        const float* vb = V + b * Skv * v_row + (hh / (H / Hkv)) * v_head;
        for (int64_t j0 = 0; j0 < nk; j0 += SG) {
            const int64_t jl = j0 + lane < Skv ? j0 + lane : Skv - 1;
            float t[D];
            {
                const float* kp = kb + jl * k_row;
#pragma unroll
                for (int d = 0; d < D; d += 4) {
                    const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(kp + d);
                    t[d] = v.x(); t[d + 1] = v.y(); t[d + 2] = v.z(); t[d + 3] = v.w();
                }
            }
            float s[QR][SG];
#pragma unroll
            for (int i = 0; i < QR; ++i)
#pragma unroll
                for (int j = 0; j < SG; ++j) s[i][j] = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
#pragma unroll
                for (int j = 0; j < SG; ++j) {
                    const float kd = sycl::select_from_group(sg, t[d], j);
#pragma unroll
                    for (int i = 0; i < QR; ++i) s[i][j] = sycl::fma(qv[i][d], kd, s[i][j]);
                }
            }
            {
                const float* vp = vb + jl * v_row;
#pragma unroll
                for (int d = 0; d < D; d += 4) {
                    const sycl::float4 v = *reinterpret_cast<const sycl::float4*>(vp + d);
                    t[d] = v.x(); t[d + 1] = v.y(); t[d + 2] = v.z(); t[d + 3] = v.w();
                }
            }
#pragma unroll
            for (int i = 0; i < QR; ++i) {
                float mt = m[i];
#pragma unroll
                for (int j = 0; j < SG; ++j) {
                    const bool ok = j0 + j < Skv && (!causal || j0 + j <= row[i]);
                    s[i][j] = ok ? s[i][j] * scale : -std::numeric_limits<float>::infinity();
                    mt = sycl::fmax(mt, s[i][j]);
                }
                // nothing seen yet (all masked so far): no rescale, no weights
                const float corr = mt == -std::numeric_limits<float>::infinity() ? 1.0f : sycl::exp(m[i] - mt);
                float ps = 0.0f;
#pragma unroll
                for (int j = 0; j < SG; ++j) {
                    s[i][j] = mt == -std::numeric_limits<float>::infinity() ? 0.0f : sycl::exp(s[i][j] - mt);
                    ps += s[i][j];
                }
                l[i] = l[i] * corr + ps;
                m[i] = mt;
#pragma unroll
                for (int d = 0; d < D; ++d) acc[i][d] *= corr;
            }
#pragma unroll
            for (int d = 0; d < D; ++d) {
#pragma unroll
                for (int j = 0; j < SG; ++j) {
                    const float vd = sycl::select_from_group(sg, t[d], j);
#pragma unroll
                    for (int i = 0; i < QR; ++i) acc[i][d] = sycl::fma(s[i][j], vd, acc[i][d]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < QR; ++i) {
            if (row[i] >= Sq) continue;
            const float inv = 1.0f / l[i];
            float* op = out + (b * Sq + row[i]) * out_row + hh * D;
#pragma unroll
            for (int d = 0; d < D; d += 4) *reinterpret_cast<sycl::float4*>(op + d) = sycl::float4(acc[i][d], acc[i][d + 1], acc[i][d + 2], acc[i][d + 3]) * inv;
        }
    });
}

// whether run() takes these: heads of 16 to 64 (a multiple of 16), q / k / v / out rows and heads float4-aligned
inline bool takes(int64_t D, int64_t Dv, const void* q, int64_t q_row, int64_t q_head, const void* k, int64_t k_row, int64_t k_head, const void* v,
                  int64_t v_row, int64_t v_head, const void* out, int64_t out_row) {
    auto al = [](const void* p) { return ((uintptr_t) p & 15) == 0; };
    return D == Dv && (D == 16 || D == 32 || D == 48 || D == 64) && al(q) && al(k) && al(v) && al(out) && q_row % 4 == 0 && q_head % 4 == 0 &&
           k_row % 4 == 0 && k_head % 4 == 0 && v_row % 4 == 0 && v_head % 4 == 0 && out_row % 4 == 0;
}

inline void dispatch(sycl::queue& q, const float* Q, int64_t q_row, int64_t q_head, const float* K, int64_t k_row, int64_t k_head, const float* V,
                     int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv, int64_t D, float scale, bool causal, float* out,
                     int64_t out_row) {
    switch (D) {
        case 16: run<16, 4>(q, Q, q_row, q_head, K, k_row, k_head, V, v_row, v_head, B, Sq, Skv, H, Hkv, scale, causal, out, out_row); break;
        case 32: run<32, 2>(q, Q, q_row, q_head, K, k_row, k_head, V, v_row, v_head, B, Sq, Skv, H, Hkv, scale, causal, out, out_row); break;
        case 48: run<48, 1>(q, Q, q_row, q_head, K, k_row, k_head, V, v_row, v_head, B, Sq, Skv, H, Hkv, scale, causal, out, out_row); break;
        default: run<64, 1>(q, Q, q_row, q_head, K, k_row, k_head, V, v_row, v_head, B, Sq, Skv, H, Hkv, scale, causal, out, out_row); break;
    }
}
}  // namespace flash

extern "C" {

int ns_model3d_skintokens_silo_attention_supported(int64_t D, int64_t Dv, int64_t H, int64_t Hkv, int64_t Skv) {
    return D == Dv && (D == 16 || D == 32 || D == 48 || D == 64) && Hkv > 0 && H % Hkv == 0 && Skv > 0 && Skv <= flash::MAX_KEYS ? 1 : 0;
}

int ns_model3d_skintokens_silo_attention(ns_gpu* g, const float* q, int64_t q_row, int64_t q_head, const float* k, int64_t k_row, int64_t k_head,
                                         const float* v, int64_t v_row, int64_t v_head, int64_t B, int64_t Sq, int64_t Skv, int64_t H, int64_t Hkv,
                                         int64_t D, int64_t Dv, float scale, int causal, float* out, int64_t out_row) {
    NS_TRY
    if (B <= 0 || Sq <= 0 || Skv <= 0 || H <= 0) return 0;
    if (!ns_model3d_skintokens_silo_attention_supported(D, Dv, H, Hkv, Skv)) return ns_fail("ns_model3d_skintokens_silo_attention: shapes it does not take");
    if (causal && Sq != Skv) return ns_fail("ns_model3d_skintokens_silo_attention: a causal mask needs Sq = Skv");
    if (!flash::takes(D, Dv, q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, out, out_row)) return ns_fail("ns_model3d_skintokens_silo_attention: rows or heads not float4-aligned");
    flash::dispatch(g->q, q, q_row, q_head, k, k_row, k_head, v, v_row, v_head, B, Sq, Skv, H, Hkv, D, scale, causal != 0, out, out_row);
    return 0;
    NS_CATCH
}

}  // extern "C"
