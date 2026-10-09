// libnextsycl-flash.so - flash attention for the diffusion kernels (nsd.cpp): the half-precision prefill kernel of
// Intel's ARK (intel/auto-round, auto_round_extension/ark, Apache 2.0) on sycl-tla (CUTLASS for SYCL, the matrix
// engine through 2-D block loads). Its own library, as H3's libh3sage.so is: sycl-tla wants its own compiler flags
// (C++17, the SPIR-V extensions for 2-D block I/O and sub-group matrix multiply, the device named at link time) and
// a long compile. nsd.cpp loads it beside itself on first use; without it attention stays on oneDNN.
//
// What it computes, per head h: out[:, h] = softmax(q[:, h] k[:, h]^T * scale) v[:, h], no mask. q [Sq rows], k and v
// [Skv rows], out [Sq rows], each row holding H heads of D halves; rows `*_rs` elements apart (q, k and v may sit
// inside one qkv buffer). Queued on the caller's queue; no wait.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>

#include <sycl/sycl.hpp>
#include "sycl_tla_sdpa.hpp"

namespace {
const sycl::queue* g_default = nullptr;    // the queue last made sycl-tla's default (setting it waits on the old one)
std::string g_err;
}

extern "C" {

const char* nsflash_error() { return g_err.c_str(); }

// 0 on success, else -1 with the reason in nsflash_error(). D: 64 or 128. (ARK's persistent schedule is not used:
// sycl-tla rejects these shapes with it, and its check ends the process rather than returning.)
int nsflash_attention(void* queue, const void* q, const void* k, const void* v, void* out, int64_t Sq, int64_t Skv,
                      int64_t H, int64_t D, int64_t q_rs, int64_t kv_rs, int64_t o_rs, float scale) try {
    if (D != 64 && D != 128) { g_err = "nsflash: head size " + std::to_string(D) + " (64 or 128 only)"; return -1; }
    if (Sq <= 0 || Skv <= 0 || H <= 0 || Sq * q_rs > INT32_MAX || Skv * kv_rs > INT32_MAX || Sq * o_rs > INT32_MAX) {
        g_err = "nsflash: sizes out of range"; return -1;
    }
    auto* qu = static_cast<sycl::queue*>(queue);
    if (g_default != qu) {
        compat::set_default_queue(*qu);
        g_default = qu;
    }
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
    using half = cute::half_t;
    const int rc = D == 128 ? ark::detail::launch_prefill_kernel_128<false, false, false, half, half, half>(o)
                            : ark::detail::launch_prefill_kernel_64<false, false, false, half, half, half>(o);
    if (rc != 0) { g_err = "nsflash: the kernel refused the problem"; return -1; }
    return 0;
} catch (const std::exception& e) {
    g_err = std::string("nsflash: ") + e.what();
    return -1;
}

// H3's SageAttention v1 entry (kernels/sage.cpp of the H3 studio, as it was: libh3sage.so's h3sage_attention): q and
// k int8 [H, S, D] packed (one scale per head per `block` rows, k less its mean), v and out half [H, S, D]. The
// diffusion kernels' default attention for long sequences (nsd.cpp, attention_sage).
int nsflash_sage_square(void* queue, const int8_t* q, const int8_t* k, const void* v, void* out, const float* qscale,
                        const float* kscale, int block, int64_t S, int64_t H, int64_t D, float scale) try {
    if (D != 64 && D != 128) { g_err = "nsflash: head size " + std::to_string(D) + " (64 or 128 only)"; return -1; }
    if (block <= 0 || block % 64 != 0) { g_err = "nsflash: the scale block must be a multiple of 64"; return -1; }
    if (S <= 0 || H <= 0 || S > INT32_MAX / D / H) { g_err = "nsflash: sequence out of range"; return -1; }
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
    const int rc = D == 128 ? ark::detail::launch_sage_prefill_kernel_128<cute::int8_t, cute::int8_t, cute::half_t>(o)
                            : ark::detail::launch_sage_prefill_kernel_64<cute::int8_t, cute::int8_t, cute::half_t>(o);
    if (rc != 0) { g_err = "nsflash: the sage kernel refused the problem"; return -1; }
    return 0;
} catch (const std::exception& e) {
    g_err = std::string("nsflash: ") + e.what();
    return -1;
}

}  // extern "C"
