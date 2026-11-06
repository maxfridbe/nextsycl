// tla/dense.cpp: the FlashVSR silo's cross-attention to the prompt (silo.h: ns_image_flashvsr_silo_attention) - ARK's
// half prefill kernel on sycl-tla (as kernels/diffusion/flash/flash.cpp runs it for the shared kernels), heads of 128,
// no mask: every token's query over the prompt's 512 keys. Its own translation unit: the dense and the sparse headers
// each define their kernels' launch options.
#include <cstdint>
#include <exception>
#include <string>

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa.hpp"
#include "../silo.h"

void ns_flashvsr_tla_use_queue(sycl::queue* q);
void ns_flashvsr_tla_error(const std::string& m);

extern "C" int ns_image_flashvsr_silo_attention(void* queue, const void* q, const void* k, const void* v, void* out, int64_t Sq, int64_t Skv, int64_t H,
                                                int64_t q_rs, int64_t kv_rs, int64_t o_rs) try {
    if (Sq <= 0 || Skv <= 0 || H <= 0 || Sq * q_rs > INT32_MAX || Skv * kv_rs > INT32_MAX || Sq * o_rs > INT32_MAX) {
        ns_flashvsr_tla_error("silo attention: sizes out of range");
        return -1;
    }
    ns_flashvsr_tla_use_queue(static_cast<sycl::queue*>(queue));
    ark::detail::Options o;
    o.q = q; o.k = k; o.v = v; o.o = out;
    o.batch = 1;
    o.num_heads_q = o.num_heads_kv = (int) H;
    o.seq_len_qo = (int) Sq;
    o.seq_len_kv = (int) Skv;
    o.head_size_qk = o.head_size_vo = 128;
    o.softmax_scale = 1.0f / std::sqrt(128.0f);
    o.is_causal = false;
    o.use_tensor_strides = true;
    o.q_stride_s = (int) q_rs; o.q_stride_h = 128; o.q_stride_b = (int) (Sq * q_rs);
    o.k_stride_s = (int) kv_rs; o.k_stride_h = 128; o.k_stride_b = (int) (Skv * kv_rs);
    o.v_stride_s = (int) kv_rs; o.v_stride_h = 128; o.v_stride_b = (int) (Skv * kv_rs);
    o.o_stride_s = (int) o_rs; o.o_stride_h = 128; o.o_stride_b = (int) (Sq * o_rs);
    using half = cute::half_t;
    if (ark::detail::launch_prefill_kernel_128<false, false, false, half, half, half>(o) != 0) {
        ns_flashvsr_tla_error("silo attention: the kernel refused the problem");
        return -1;
    }
    return 0;
} catch (const std::exception& e) {
    ns_flashvsr_tla_error(std::string("silo attention: ") + e.what());
    return -1;
}
