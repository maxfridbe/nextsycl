// tla/attn.cpp: HunyuanVideo-1.5's attention kernels on sycl-tla (dist/silo/libnextsycl-hunyuanvideo15-tla.so, opened
// by the silo's attn.cpp beside itself). The DiT's joint attention is one long full attention (the 5 s 480p clip: 49,290
// picture rows + the prompt's, 16 heads of 128) - 80% of a step's arithmetic.
//
//   half   ARK's half prefill kernel (FMHAConfig: XeFMHAFwdKernel, SDPAFwdMainloop, FMHAFwdEpilogue), the tiles this
//          silo's choice (`tile` below; ARK's own for head 128 is 0) - q, k, v read in place from the qkv rows, the
//          result written as half into the activations. The same arithmetic in every tile: float32 scores and
//          accumulators, half P - the tiles change the order of the running maximum's updates, so a result can differ
//          from another tile's in the last bits.
//   sage   SageAttention v1 (ARK's int8 prefill kernel, its mainloop copied from the MiniMax-H3-Image silo:
//          ark/sage_mainloop.hpp, the scale folded into the exponent): q and k int8 [H, S, D] (one scale per head per
//          `block` rows, k less its mean; the silo's attn.cpp makes them), v half read in place from the qkv rows, the
//          result half into the activations. Approximate - an opt-in.
#include <cmath>
#include <cstdint>
#include <exception>
#include <string>

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa.hpp"
#include "ark/sage_mainloop.hpp"

namespace {
const sycl::queue* g_default = nullptr;   // the queue last made sycl-tla's default (setting it waits on the old one)
std::string g_err;

using namespace cute;
using half = cute::half_t;

void make_default(void* queue) {
    auto* qu = static_cast<sycl::queue*>(queue);
    if (g_default != qu) {
        compat::set_default_queue(*qu);
        g_default = qu;
    }
}

// the half kernels: (q rows a work-group, keys a block, the PV tile's v width), sub-groups, prefetch stages
template <int Q, int KB, int VW, int SG, int ST>
int half_kernel(const ark::detail::Options& o) {
    using QK = Shape<Int<Q>, Int<KB>, _32>;
    using PV = Shape<Int<Q>, Int<VW>, Int<KB>>;
    using Out = Shape<Int<Q>, _128>;
    using SGL = Layout<Shape<Int<SG>, _1, _1>>;
    return ark::detail::FMHAConfig<false, QK, PV, Out, SGL, void, ST, false, half, half, half>::template run<false, false>(o);
}

// the sage kernel: ARK's tiles (256 q rows, 64 keys, 16 sub-groups), the result half
using TileQK = Shape<_256, _64, _32>;
using TilePV = Shape<_256, _32, _64>;
using TileO = Shape<_256, _128>;
using SGLayoutQK = Layout<Shape<_16, _1, _1>>;
using SGLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SGLayoutQK{}));
using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, int32_t, int8_t>>, Layout<TileQK>, SGLayoutQK>::TiledMMA;
using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, half>>, Layout<TilePV>, SGLayoutPV>::TiledMMA;
constexpr int VTiles = 128 / 32;

template <class T, class Stride>
auto dummy_tensor(Stride) {
    T v{};
    return make_tensor(make_gmem_ptr(&v), make_layout(repeat<rank_v<Stride>>(1), Stride{}));
}
using TensorQ = decltype(dummy_tensor<cute::int8_t>(Stride<int, _1, int, int>{}));
using TensorK = decltype(dummy_tensor<cute::int8_t>(Stride<int, _1, int, int>{}));
using TensorV = decltype(dummy_tensor<half>(Stride<_1, int, int, int>{}));
using TensorO = decltype(dummy_tensor<half>(Stride<int, _1, int, int>{}));

using Mainloop = nshv15::SageMainloop<TiledMMAQK, TiledMMAPV, VTiles, TensorQ, TensorK, TensorV>;
using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<Mainloop, TileO, TensorO, void>;
using SageKernel = cutlass::fmha::kernel::XeSageFwdKernel<cutlass::fmha::kernel::SageProblemShape<false>, Mainloop, Epilogue,
                                                          cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>;
}  // namespace

extern "C" {

const char* ns_video_hv15_tla_error() { return g_err.c_str(); }

// The tiles `half` can run (0 .. count - 1)
int ns_video_hv15_tla_tiles() { return 5; }

// softmax(q k^T scale) v per head: Sq query rows q_rs apart, Skv key / value rows kv_rs apart (q, k, v may be the
// qkv rows), out half rows o_rs apart; H heads of D = 128. Queued on `queue`; 0, or -1 with the reason in
// ns_video_hv15_tla_error().
int ns_video_hv15_tla_half(void* queue, const void* q, const void* k, const void* v, void* out, int64_t Sq, int64_t Skv, int64_t H, int64_t D,
                           int64_t q_rs, int64_t kv_rs, int64_t o_rs, float scale, int tile) try {
    if (D != 128) { g_err = "hv15 tla half: head size " + std::to_string(D) + " (128 only)"; return -1; }
    if (Sq <= 0 || Skv <= 0 || H <= 0 || Sq * q_rs > INT32_MAX || Skv * kv_rs > INT32_MAX || Sq * o_rs > INT32_MAX) {
        g_err = "hv15 tla half: sizes out of range"; return -1;
    }
    make_default(queue);
    ark::detail::Options o;
    o.q = q; o.k = k; o.v = v; o.o = out;
    o.batch = 1;
    o.num_heads_q = o.num_heads_kv = (int) H;
    o.seq_len_qo = (int) Sq;
    o.seq_len_kv = (int) Skv;
    o.head_size_qk = o.head_size_vo = (int) D;
    o.softmax_scale = scale;
    o.is_causal = false;
    o.use_tensor_strides = true;
    o.q_stride_s = (int) q_rs; o.q_stride_h = (int) D; o.q_stride_b = (int) (Sq * q_rs);
    o.k_stride_s = (int) kv_rs; o.k_stride_h = (int) D; o.k_stride_b = (int) (Skv * kv_rs);
    o.v_stride_s = (int) kv_rs; o.v_stride_h = (int) D; o.v_stride_b = (int) (Skv * kv_rs);
    o.o_stride_s = (int) o_rs; o.o_stride_h = (int) D; o.o_stride_b = (int) (Sq * o_rs);
    int rc;
    switch (tile) {
        case 0: rc = half_kernel<256, 32, 32, 16, 2>(o); break;   // ARK's own for head 128
        case 1: rc = half_kernel<256, 64, 32, 16, 2>(o); break;   // the sage kernel's: 64 keys a block
        case 2: rc = half_kernel<128, 64, 32, 8, 2>(o); break;    // half the rows: twice the work-groups
        case 3: rc = half_kernel<256, 32, 32, 16, 3>(o); break;   // a third prefetch stage
        case 4: rc = half_kernel<128, 32, 32, 8, 2>(o); break;
        default: g_err = "hv15 tla half: no tile " + std::to_string(tile); return -1;
    }
    if (rc != 0) { g_err = "hv15 tla half: the kernel refused the problem"; return -1; }
    return 0;
} catch (const std::exception& e) {
    g_err = std::string("hv15 tla half: ") + e.what();
    return -1;
}

// q int8 [H, Sq, D], k int8 [H, Skv, D] (qscale [H, ceil(Sq / block)], kscale [H, ceil(Skv / block)], k less its mean),
// v half Skv rows v_rs apart (the qkv rows); out half Sq rows o_rs apart. D 128, `block` a multiple of 64.
int ns_video_hv15_tla_sage(void* queue, const int8_t* q, const int8_t* k, const void* v, void* out, const float* qscale,
                           const float* kscale, int block, int64_t Sq, int64_t Skv, int64_t H, int64_t D, int64_t v_rs, int64_t o_rs, float scale) try {
    if (D != 128) { g_err = "hv15 tla sage: head size " + std::to_string(D) + " (128 only)"; return -1; }
    if (block <= 0 || block % 64 != 0) { g_err = "hv15 tla sage: the scale block must be a multiple of 64"; return -1; }
    if (Sq <= 0 || Skv <= 0 || H <= 0 || Sq > INT32_MAX / D / H || Skv > INT32_MAX / D / H || Skv * v_rs > INT32_MAX || Sq * o_rs > INT32_MAX) {
        g_err = "hv15 tla sage: sequence out of range"; return -1;
    }
    make_default(queue);
    ark::detail::Options o;
    o.q = q; o.k = k; o.v = v; o.o = out;
    o.scale_block_size = block;
    o.qscale = qscale; o.kscale = kscale;
    o.batch = 1;
    o.num_heads_q = o.num_heads_kv = (int) H;
    o.seq_len_qo = (int) Sq;
    o.seq_len_kv = (int) Skv;
    o.head_size_qk = o.head_size_vo = (int) D;
    o.softmax_scale = scale;
    o.is_causal = false;
    o.use_tensor_strides = true;
    o.q_stride_s = o.k_stride_s = (int) D;
    o.q_stride_h = (int) (Sq * D);
    o.k_stride_h = (int) (Skv * D);
    o.q_stride_b = (int) (H * Sq * D);
    o.k_stride_b = (int) (H * Skv * D);
    o.v_stride_s = (int) v_rs; o.v_stride_h = (int) D; o.v_stride_b = (int) (Skv * v_rs);
    o.o_stride_s = (int) o_rs; o.o_stride_h = (int) D; o.o_stride_b = (int) (Sq * o_rs);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    ark::detail::SageKernelRunner<SageKernel, false> runner;
    if (runner.run(o, hw_info) != cutlass::Status::kSuccess) { g_err = "hv15 tla sage: the kernel refused the problem"; return -1; }
    return 0;
} catch (const std::exception& e) {
    g_err = std::string("hv15 tla sage: ") + e.what();
    return -1;
}

}  // extern "C"
