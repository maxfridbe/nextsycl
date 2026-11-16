// tla/sage.cpp: the MiniMax-H3-Image silo's SageAttention kernel on sycl-tla (dist/silo/libnextsycl-h3image-tla.so,
// opened by the silo's sage.cpp beside itself). ARK's int8 prefill kernel (its XeSageFwdKernel, FMHAFwdEpilogue and
// runner, from the build image's ARK headers) around this silo's copy of its mainloop (ark/sage_mainloop.hpp: the scale
// folded into the exponent's fused multiply-add) and with the result written as bfloat16 straight into the activations
// [S, H * D] - so the half [H, S, D] result and the pass converting it are gone.
//
// Tiles as ARK's: 256 query rows a work-group (16 sub-groups of 16 rows), keys 64 a block, two prefetch stages. B70,
// 56 heads of 128 (sagebench, the kernel alone, median of 20): 16,434 tokens 50.8 ms (152.5 TFLOPS counted as 4 S^2 D H)
// against ARK's 52.9 (146.3); 1,074 tokens 0.288 against 0.301 ms.
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

using TileQK = Shape<_256, _64, _32>;    // (q, k, d)
using TilePV = Shape<_256, _32, _64>;    // (q, v, k)
using TileO = Shape<_256, _128>;
using SGLayoutQK = Layout<Shape<_16, _1, _1>>;
using SGLayoutPV = decltype(cutlass::fmha::collective::get_sg_layout_pv(SGLayoutQK{}));
using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, int32_t, int8_t>>, Layout<TileQK>, SGLayoutQK>::TiledMMA;
using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::half_t>>, Layout<TilePV>, SGLayoutPV>::TiledMMA;
constexpr int VTiles = 128 / 32;

template <class T, class Stride>
auto dummy_tensor(Stride) {
    T v{};
    return make_tensor(make_gmem_ptr(&v), make_layout(repeat<rank_v<Stride>>(1), Stride{}));
}
using TensorQ = decltype(dummy_tensor<cute::int8_t>(Stride<int, _1, int, int>{}));
using TensorK = decltype(dummy_tensor<cute::int8_t>(Stride<int, _1, int, int>{}));
using TensorV = decltype(dummy_tensor<cute::half_t>(Stride<_1, int, int, int>{}));
using TensorO = decltype(dummy_tensor<cute::bfloat16_t>(Stride<int, _1, int, int>{}));

using Mainloop = nsh3i::SageMainloop<TiledMMAQK, TiledMMAPV, VTiles, TensorQ, TensorK, TensorV>;
using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<Mainloop, TileO, TensorO, void>;
using Kernel = cutlass::fmha::kernel::XeSageFwdKernel<cutlass::fmha::kernel::SageProblemShape<false>, Mainloop, Epilogue,
                                                      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>;
}  // namespace

extern "C" {

const char* ns_image_h3i_tla_error() { return g_err.c_str(); }

// q, k int8 [H, S, D] (one scale per head per `block` rows: qscale / kscale [H, ceil(S / block)], k less its mean), v
// half [H, S, D]; out bfloat16 [S, H * D]: softmax(q k^T scale) v for each head. Queued on `queue`; 0, or -1 with the
// reason in ns_image_h3i_tla_error(). D 128, `block` a multiple of 64.
int ns_image_h3i_tla_sage(void* queue, const int8_t* q, const int8_t* k, const void* v, void* out, const float* qscale,
                          const float* kscale, int block, int64_t S, int64_t H, int64_t D, float scale) try {
    if (D != 128) { g_err = "h3image tla sage: head size " + std::to_string(D) + " (128 only)"; return -1; }
    if (block <= 0 || block % 64 != 0) { g_err = "h3image tla sage: the scale block must be a multiple of 64"; return -1; }
    if (S <= 0 || H <= 0 || S > INT32_MAX / D / H) { g_err = "h3image tla sage: sequence out of range"; return -1; }
    auto* qu = static_cast<sycl::queue*>(queue);
    if (g_default != qu) {
        compat::set_default_queue(*qu);
        g_default = qu;
    }
    ark::detail::Options o;
    o.q = q; o.k = k; o.v = v; o.o = out;
    o.scale_block_size = block;
    o.qscale = qscale; o.kscale = kscale;
    o.batch = 1;
    o.num_heads_q = o.num_heads_kv = (int) H;
    o.seq_len_qo = o.seq_len_kv = (int) S;
    o.head_size_qk = o.head_size_vo = (int) D;
    o.softmax_scale = scale;
    o.is_causal = false;
    // q, k, v packed [H, S, D]; the result's rows H * D apart
    o.use_tensor_strides = true;
    o.q_stride_s = o.k_stride_s = o.v_stride_s = (int) D;
    o.q_stride_h = o.k_stride_h = o.v_stride_h = (int) (S * D);
    o.q_stride_b = o.k_stride_b = o.v_stride_b = (int) (H * S * D);
    o.o_stride_s = (int) (H * D);
    o.o_stride_h = (int) D;
    o.o_stride_b = (int) (S * H * D);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    ark::detail::SageKernelRunner<Kernel, false> runner;
    if (runner.run(o, hw_info) != cutlass::Status::kSuccess) { g_err = "h3image tla sage: the kernel refused the problem"; return -1; }
    return 0;
} catch (const std::exception& e) {
    g_err = std::string("h3image tla sage: ") + e.what();
    return -1;
}

}  // extern "C"
