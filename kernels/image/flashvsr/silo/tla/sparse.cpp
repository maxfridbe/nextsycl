// tla/sparse.cpp: the FlashVSR silo's block-sparse self-attention (silo.h: ns_image_flashvsr_silo_sparse) - Intel ARK's
// native-precision sparse SDPA (auto_round_kernel's sycl_tla_sdpa_sparse.hpp, MIT; the build image's /opt/ark) on
// sycl-tla, half in and out, heads of 128. Its routing is a LUT row a query block: FlashVSR's lists of 128-key blocks,
// delta-encoded (block_lists with delta), the keys' route blocks = Skv / num_k_blocks = 128 tokens.
//
// sycl-tla launches on compat's default queue of the calling thread's device: `use_queue` points the thread at the
// card of the queue passed and makes that queue its default (a change waits for the old default once - each card's
// queue stays its default after its first call). A problem it cannot take ends the process (CUTLASS_CHECK), so every
// shape is checked here first.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>

#include <sycl/sycl.hpp>
// ARK's wrapper and its sparse mainloop copied into ark/ (MIT / BSD-3-Clause, notices kept): the mainloop's prefetch
// changed (ark/stla/xe_sparse_sdpa_fwd_mainloop.hpp, "nextsycl")
#include "ark/sycl_tla_sdpa_sparse.hpp"
#include "../silo.h"

namespace {
thread_local std::string g_err;
std::mutex g_mu;

int fail(const std::string& m) {
    g_err = m;
    return -1;
}
}  // namespace

// shared with dense.cpp: the calling thread's sycl-tla device and default queue made the given queue's
void ns_flashvsr_tla_use_queue(sycl::queue* q) {
    std::lock_guard<std::mutex> l(g_mu);
    const sycl::device dev = q->get_device();
    const unsigned n = compat::device_count();
    for (unsigned i = 0; i < n; ++i) {
        if (static_cast<const sycl::device&>(compat::get_device(i)) == dev) {
            compat::select_device(i);
            if (compat::get_default_queue() != *q) compat::set_default_queue(*q);
            return;
        }
    }
    throw std::runtime_error("the queue's GPU is not one sycl-tla sees");
}

void ns_flashvsr_tla_error(const std::string& m) { g_err = m; }

namespace {
// A query tile of 128 rows - FlashVSR's query block, so each work-group walks exactly one LUT row and every key block it
// loads serves all its 128 queries (ARK ships tiles of 64, which load each key block twice, and 256, which walks the
// union of two rows' lists with half its sub-groups idle on a block only one row chose). ARK's 64-row configuration
// with twice the sub-groups (8 of 16 rows); KT: the keys a step.
// (KT: keys a step; DT: the features of q k^T a step; ST: the pipeline's stages; SG: sub-groups, 128 / SG query rows each)
template <int KT, int DT = 32, int ST = 2, int SG = 8>
int launch_q128(const ark::sparse_detail::Options& o) {
    using namespace cute;
    using half = cute::half_t;
    using ShapeQK = Shape<_128, Int<KT>, Int<DT>>;
    using ShapePV = Shape<_128, _32, Int<KT>>;
    using ShapeOut = Shape<_128, _128>;
    using SubgroupLayoutQK = Layout<Shape<Int<SG>, _1, _1>>;
    return ark::sparse_detail::SparseSDPAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, ST, false, half, half, half, half>::template run<false>(o);
}
}  // namespace

extern "C" {

const char* ns_image_flashvsr_silo_error(void) { return g_err.c_str(); }

int ns_image_flashvsr_silo_supported(void* queue) try {
    auto* q = static_cast<sycl::queue*>(queue);
    // the kernels are compiled ahead for Xe2 (bmg-g31): the B70 / B65 / B580's matrix engines and 2-D block loads
    return q && q->get_device().is_gpu() ? 1 : 0;
} catch (...) {
    return 0;
}

int ns_image_flashvsr_silo_sparse(void* queue, const void* q, const void* k, const void* v, void* out, int64_t Sq, int64_t Skv, int64_t H,
                                  int64_t kv_rs, const int32_t* lut, const int32_t* counts, int64_t qrow, int64_t q_tile) try {
    if (Sq <= 0 || Skv <= 0 || H <= 0) return fail("silo sparse: an empty problem");
    if (Sq % qrow || Skv % 128) return fail("silo sparse: rows in whole blocks (queries of " + std::to_string(qrow) + ", keys of 128)");
    if (!((q_tile == 64 && qrow == 64) || (q_tile == 256 && qrow == 128) || (q_tile >= 128 && q_tile <= 133 && qrow == 128)))
        return fail("silo sparse: q_tile 64 with LUT rows of 64, or 128 - 133 / 256 with 128");
    const int64_t ld = H * 128;
    if (kv_rs < ld) return fail("silo sparse: key rows shorter than their heads");
    if (Sq * ld > INT32_MAX || Skv * kv_rs > INT32_MAX) return fail("silo sparse: sizes out of range");
    auto* qu = static_cast<sycl::queue*>(queue);
    ns_flashvsr_tla_use_queue(qu);
    ark::sparse_detail::Options o;
    o.q = q;
    o.k = k;
    o.v = v;
    o.o = out;
    o.batch = 1;
    o.num_heads_q = o.num_heads_kv = (int) H;
    o.seq_len_qo = (int) Sq;
    o.seq_len_kv = (int) Skv;
    o.head_size_qk = o.head_size_vo = 128;
    o.softmax_scale = 1.0f / std::sqrt(128.0f);   // the mainloop multiplies by log2(e) itself
    o.use_tensor_strides = true;
    o.q_stride_s = (int) ld; o.q_stride_h = 128; o.q_stride_b = (int) (Sq * ld);
    o.k_stride_s = (int) kv_rs; o.k_stride_h = 128; o.k_stride_b = (int) (Skv * kv_rs);
    o.v_stride_s = (int) kv_rs; o.v_stride_h = 128; o.v_stride_b = (int) (Skv * kv_rs);
    o.o_stride_s = (int) ld; o.o_stride_h = 128; o.o_stride_b = (int) (Sq * ld);
    o.scale_block_size = 0;
    o.sparse_q_block_size = (int) qrow;
    o.q_tile_override = (int) q_tile;
    o.lut = lut;
    o.valid_block_num = counts;
    o.num_q_blocks = (int) (Sq / qrow);
    o.num_k_blocks = (int) (Skv / 128);
    using half = cute::half_t;
    const int rc = q_tile == 64 ? ark::sparse_detail::launch_sparse_sdpa_prefill_kernel_128_qtile64<false, false, half, half, half>(o)
                 : q_tile == 128 ? launch_q128<64>(o)
                 : q_tile == 129 ? launch_q128<32>(o)
                 : q_tile == 130 ? launch_q128<32, 64>(o)
                 : q_tile == 131 ? launch_q128<32, 32, 3>(o)
                 : q_tile == 132 ? launch_q128<32, 32, 2, 16>(o)
                 : q_tile == 133 ? launch_q128<16, 32, 2, 8>(o)
                                 : ark::sparse_detail::launch_sparse_sdpa_prefill_kernel_128<false, false, half, half, half>(o);
    if (rc != 0) return fail("silo sparse: the kernel refused the problem");
    return 0;
} catch (const std::exception& e) {
    return fail(std::string("silo sparse: ") + e.what());
}

}  // extern "C"
