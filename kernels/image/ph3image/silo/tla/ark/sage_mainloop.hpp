/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// nextsycl: Intel ARK's SAGEV1FwdMainloop (auto_round_kernel/wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp, the
// kernel libnextsycl-flash.so's nsflash_sage_square runs) copied for the P-H3-Image silo (from the MiniMax-H3-Image silo's copy) and changed:
//   - only what the denoiser uses: no causal mask, no mask tensor, no cached / paged KV, v in half (no int8 P V);
//   - the scores' dequantization scale folded into the exponent: the block maximum is taken on the raw int32 products
//     (times the block's positive scale once a row), and P = exp2(fma(s, scale, -max)) - one fused multiply-add an
//     element where ARK multiplies every element by the scale and then subtracts the maximum (one rounding less);
//   - the result (the kernel's FMHAFwdEpilogue) is written by the caller's choice of output tensor: the silo hands it
//     bfloat16 [S, H * D], the activations' own layout, so no conversion pass follows.
// Everything else (tiles, 2-D block loads and prefetches, the split barrier, the SIMD32 exponent) is ARK's.
#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include <cute/util/xe_split_barrier.hpp>

namespace nsph3 {
using namespace cute;

template <class TiledMMAQK_, class TiledMMAPV_, int VTiles_, class TensorQ_, class TensorK_, class TensorV_>
struct SageMainloop {
    using TiledMMAQK = TiledMMAQK_;
    using TiledMMAPV = TiledMMAPV_;
    using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
    using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
    static constexpr int VTiles = VTiles_;
    using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
    using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

    using TensorQ = TensorQ_;
    using TensorK = TensorK_;
    using TensorV = TensorV_;
    using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
    using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
    using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));
    // the kernel (ARK's XeSageFwdKernel) passes cache tensors of these types; never used here
    using TensorK_cache = TensorK_;
    using TensorV_cache = TensorV_;
    using TensorK_cache2D = TensorK2D;
    using TensorV_cache2D = TensorV2D;

    using TiledCopyQ = decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{}));
    using TiledCopyK = decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{}));
    using TiledCopyV = decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{}));

    template <typename TiledMMA>
    using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));
    using FragS = FragC<TiledMMAQK>;
    using ElementS = typename TiledMMAQK::ValTypeD;
    using ElementP = float;
    using FragP = decltype(make_subgroup_tensor(make_fragment_like<ElementP>(typename FragS::Base{}), decltype(FragS{}.tv_layout()){}));
    using FragPCol = decltype(reduce<0>(FragP{}, sycl::plus<void>{}));
    using SingleFragA = FragC<TiledMMAPV>;
    using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;
    using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
    using ElementA = typename SingleFragA::value_type;

    static constexpr bool CausalMask = false;
    static constexpr bool CachedKV = false;
    static constexpr bool PagedKV = false;
    static constexpr bool UseInt8PV = false;

    struct Arguments {
        float const scale;
        float const* mask = nullptr;
        int scale_block_size = 0;
        float const* qscale = nullptr;
        float const* kscale = nullptr;
        float const* vscale = nullptr;
        int const* ptr_page_table = nullptr;
        int page_size = 0;
        int const* num_pages_per_seq = nullptr;
    };
    using Params = Arguments;
    struct SharedStorage {};
    Params params;

    SageMainloop(Params const& p, SharedStorage&) : params(p) {}

    static constexpr Params to_underlying_arguments(Arguments const& args, void*) {
        constexpr double kLog2e = 1.4426950408889634074;   // log_2(e)
        return Params{float(args.scale * kLog2e), args.mask, args.scale_block_size, args.qscale, args.kscale, args.vscale,
                      args.ptr_page_table, args.page_size, args.num_pages_per_seq};
    }
    CUTLASS_HOST_DEVICE static bool can_implement(Arguments const& args) { return args.scale_block_size > 0; }

    // DPAS kernels run as SIMD16, so pack two SIMD16 vectors into one temporary SIMD32 vector before issuing a single
    // vISA SIMD32 instruction (ARK's)
    CUTLASS_DEVICE static void exp2_pair_simd32_asm(ElementP x0, ElementP x1, ElementP& y0, ElementP& y1) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
        asm("{\n"
            ".decl OUT0 v_type=G type=F num_elts=16 alias=<%0,0>\n"
            ".decl OUT1 v_type=G type=F num_elts=16 alias=<%1,0>\n"
            ".decl IN0  v_type=G type=F num_elts=16 alias=<%2,0>\n"
            ".decl IN1  v_type=G type=F num_elts=16 alias=<%3,0>\n"
            ".decl TMP_IN  v_type=G type=F num_elts=32 align=64\n"
            ".decl TMP_OUT v_type=G type=F num_elts=32 align=64\n"
            "mov (M1_NM, 16) TMP_IN(0,0)<1> IN0(0,0)<1;1,0>\n"
            "mov (M1_NM, 16) TMP_IN(1,0)<1> IN1(0,0)<1;1,0>\n"
            "exp (M1_NM, 32) TMP_OUT(0,0)<1> TMP_IN(0,0)<1;1,0>\n"
            "mov (M1_NM, 16) OUT0(0,0)<1> TMP_OUT(0,0)<1;1,0>\n"
            "mov (M1_NM, 16) OUT1(0,0)<1> TMP_OUT(1,0)<1;1,0>\n"
            "}\n"
            : "=rw"(y0), "=rw"(y1)
            : "rw"(x0), "rw"(x1));
#else
        y0 = sycl::native::exp2(x0);
        y1 = sycl::native::exp2(x1);
#endif
    }

    template <typename QVCoord>
    CUTLASS_DEVICE void operator()(TensorQ2D const& Q_2D, TensorK2D const& K_2D, TensorV2D const& V_2D, FragA& tArA, FragARow& tA_max,
                                   FragARow& tA_sum, QVCoord blk_qv, int blk_k0, int blk_k1, int total_blk, int thr_id, int seq_len,
                                   int /*seq_len_kv_cache*/, int /*l_coord*/, float* scaleQ, float* scaleK, float* /*scaleV*/,
                                   int /*full_tile_offset*/, int /*discard_seq_coord*/, TensorK_cache2D const& = TensorK_cache2D{},
                                   TensorV_cache2D const& = TensorV_cache2D{}) {
        using namespace sycl::ext::oneapi::this_work_item;
        auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

        /* Proxy coordinate tensors for Q / K / P / V, and their work-group tiles */
        Tensor cQ = make_identity_tensor(Q_2D.shape());
        Tensor cK = make_identity_tensor(K_2D.shape());
        Tensor cV = make_identity_tensor(V_2D.shape());
        Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));
        Tensor gQ = local_tile(cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});            // (q,d,D)
        Tensor gK = local_tile(cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});          // (k,d,K,D)
        Tensor gV = local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));                    // (v,k,K)
        Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});    // (v,k,VV,K)

        /* Global -> register copies, MMAs, and their work-item slices */
        TiledCopyQ copy_q{Q_2D};
        TiledCopyK copy_k{K_2D};
        TiledCopyV copy_v{V_2D};
        TiledMMAQK mma_qk{};
        TiledMMAPV mma_pv{};
        auto thr_copy_q = copy_q.get_slice(thr_id);
        auto thr_copy_k = copy_k.get_slice(thr_id);
        auto thr_copy_v = copy_v.get_slice(thr_id);
        auto thr_mma_qk = mma_qk.get_slice(thr_id);
        auto thr_mma_pv = mma_pv.get_slice(thr_id);

        auto tQgQ = thr_copy_q.partition_S(gQ);
        auto tKgK = thr_copy_k.partition_S(gK);
        auto tVgV = thr_copy_v.partition_S(gV_split);
        auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
        auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
        auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
        auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));
        auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
        FragP tPrS = make_subgroup_tensor(make_fragment_like<ElementP>(tSrS.tensor()), tSrS.tv_layout());
        auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
        auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
        auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

        /* Prefetches */
        auto prefetch_q = make_block_2d_prefetch(copy_q);
        auto prefetch_k = make_block_2d_prefetch(copy_k);
        auto prefetch_v = make_block_2d_prefetch(copy_v);
        auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
        auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
        auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);

        constexpr int Stages = 2;
        for (int D = 0; D < size<3>(pQgQ); D++) prefetch(prefetch_q, pQgQ(_, _, _, D));
        for (int D = 0; D < size<4>(pKgK); D++) {
            CUTLASS_PRAGMA_UNROLL
            for (int K = 0; K < Stages; K++) prefetch(prefetch_k, pKgK(_, _, _, K, D));
        }
        clear(tArA);
        fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
        clear(tA_sum);

        const bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);
        const int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));
        // this sub-group's rows share one q scale (its rows sit in one scale block); times the softmax scale, log2 units
        const float dq_scale = scaleQ[(get<0>(blk_qv) * get<0>(TileShapeQK{}) + cutlass::get_sub_group_id() * q_sg_tile) / params.scale_block_size] * params.scale;

        for (int K = blk_k0; K < blk_k1; K++) {
            /* Split barrier to keep threads together */
            barrier_arrive(ScopeWorkgroup);
            const int scalek_idx = K * get<1>(TileShapeQK{}) / params.scale_block_size;

            /* GEMM 1: S = Q K^T, int8 -> int32 */
            clear(tSrS);
            CUTLASS_PRAGMA_UNROLL
            for (int D = 0; D < size<4>(tKgK); D++) {
                copy(copy_q, tQgQ(_, _, _, D), tQrQ);
                copy(copy_k, tKgK(_, _, _, K, D), tKrK);
                reorder(tQrQ, tSrQ);
                reorder(tKrK, tSrK);
                cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
            }

            /* V prefetch for GEMM 2 */
            CUTLASS_PRAGMA_UNROLL
            for (int VV = 0; VV < VTiles; VV++) prefetch(prefetch_v, pVgV(_, _, _, VV, K));
            reorder(tSrS, tPrS);
            const float sc = dq_scale * scaleK[scalek_idx];   // > 0

            /* k masking for the remainder block */
            if (check_remainder_k && K == total_blk - 1) {
                FragPCol k_rem_mask;
                int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) k_rem_mask(i) = (k < seq_len) ? ElementP(sycl::nan(0u)) : ElementP(-INFINITY);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < tPrS.size(); i++) tPrS(i) = sycl::fmin(tPrS(i), broadcast<1>(k_rem_mask, tPrS, i));
            }

            /* Softmax step: the block maximum of the raw products times the scale, then exp2(s sc - max) */
            auto tS_bmax = reduce<1>(tPrS, sycl::maximum{});
            FragARow rescale;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tA_max.size(); i++) {
                const ElementA new_max = sycl::max(tA_max(i), ElementA(tS_bmax(i)) * sc);
                rescale(i) = sycl::native::exp2(tA_max(i) - new_max);
                tA_max(i) = new_max;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tPrS.size(); i += 2) {
                const ElementP x0 = sycl::fma(tPrS(i), sc, -broadcast<0>(tA_max, tPrS, i));
                const ElementP x1 = sycl::fma(tPrS(i + 1), sc, -broadcast<0>(tA_max, tPrS, i + 1));
                exp2_pair_simd32_asm(x0, x1, tPrS(i), tPrS(i + 1));
            }
            if (K != blk_k0) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < tA_sum.size(); i++) tA_sum(i) *= rescale(i);
            }
            auto tS_bsum = reduce<1>(tPrS, sycl::plus<void>{});
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tA_sum.size(); i++) tA_sum(i) += tS_bsum(i);
            reorder(tPrS, tArP);

            /* GEMM 2: A += P V, split in v; the accumulator's rescale fused per v tile */
            CUTLASS_PRAGMA_UNROLL
            for (int VV = 0; VV < VTiles; VV++) {
                copy(copy_v, tVgV(_, _, _, VV, K), tVrV);
                reorder(tVrV, tArV);
                if (K != blk_k0) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < tArA.size() / VTiles; i++) tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
                }
                cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
            }

            /* K prefetch */
            for (int D = 0; D < size<4>(pKgK); D++) prefetch(prefetch_k, pKgK(_, _, _, K + Stages, D));
            barrier_wait(ScopeWorkgroup);
        }
    }
};
}  // namespace nsph3
