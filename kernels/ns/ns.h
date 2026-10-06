/* ns.h: libnextsycl's C ABI - what the Rust runtime (crates/ns-sys) calls.
 *
 * Every call returns 0 on success, or -1 with the reason in ns_last_error() (per thread). Device memory is USM
 * device memory of one GPU's own context: each GPU is opened with a context of its own, never the platform's
 * default context, which spans every GPU and makes xe mirror each card's allocations into host RAM (measured
 * 2026-10-05: ~45 GB of host RAM for 40 GB of model on two cards). Copies between GPUs therefore go through
 * host memory (ns_copy_peer).
 */
#ifndef NS_H
#define NS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

const char* ns_last_error(void);
const char* ns_version(void);

/* the Level Zero GPUs, in the order ns_gpu_open takes; name of index i into buf */
int ns_gpu_count(void);
int ns_gpu_name(int index, char* buf, size_t len);
/* opens GPU `index` with its own context and one in-order queue */
int ns_gpu_open(int index, ns_gpu** out);
void ns_gpu_close(ns_gpu* g);
/* the queue, as the imported kernels take it (`void* stream` = sycl::queue*) */
void* ns_gpu_queue(ns_gpu* g);
/* bytes: total device memory, and free (-1 when the driver does not say) */
int ns_gpu_memory(ns_gpu* g, uint64_t* total, int64_t* free_bytes);

int ns_alloc(ns_gpu* g, size_t bytes, void** out);
int ns_free(ns_gpu* g, void* p);
/* pinned host memory of this GPU's context: the fast source and target of its copies */
int ns_alloc_host(ns_gpu* g, size_t bytes, void** out);
int ns_free_host(ns_gpu* g, void* p);
/* queued on the GPU's queue (in order); ns_sync waits for everything queued */
int ns_copy_to(ns_gpu* g, void* dst, const void* src_host, size_t bytes);
int ns_copy_from(ns_gpu* g, void* dst_host, const void* src, size_t bytes);
int ns_copy_dev(ns_gpu* g, void* dst, const void* src, size_t bytes);
int ns_fill(ns_gpu* g, void* dst, uint8_t value, size_t bytes);
int ns_sync(ns_gpu* g);
/* device memory of one GPU to another's, through `staging`: host memory of `bytes` (pinned memory belongs to one
 * context, so it is pinned for at most one of the two copies; plain memory works for both, slower); waits */
int ns_copy_peer(ns_gpu* to, void* dst, ns_gpu* from, const void* src, size_t bytes, void* staging);

/* ---- the bring-up kernels (glm.cpp): float32 activations, row-major, all queued on the GPU's queue ---- */
/* n values of a ggml-typed tensor (0 F32, 1 F16, 30 BF16, 8 Q8_0, 10-14 Q2_K..Q6_K, 16 IQ2_XXS) into float32 */
int ns_dequant(ns_gpu* g, int type, const void* src, size_t n, float* dst);
/* y [T, N] (+)= x [T, K] . w [N, K]^T; rows of x ldx floats apart, rows of y ldy apart */
int ns_gemm(ns_gpu* g, int64_t T, int64_t N, int64_t K, const float* x, int64_t ldx, const float* w, float* y, int64_t ldy,
            int accumulate);
/* batch products: y + b*sy [T, N] (+)= (x + b*sx) [T, K] . (w + b*sw) [N, K]^T */
int ns_gemm_batch(ns_gpu* g, int64_t batch, int64_t T, int64_t N, int64_t K, const float* x, int64_t ldx, int64_t sx, const float* w, int64_t sw,
                  float* y, int64_t ldy, int64_t sy, int accumulate);
/* mHC mixes: m [T, 24] = fn [24, n] . rms(x [T, n]); part: scratch of T * 32 * 25 floats */
int ns_hc_mix(ns_gpu* g, const float* x, const float* fn, float* m, float* part, int64_t T, int64_t n, float eps);
int ns_rms_norm(ns_gpu* g, const float* x, const float* w, float* y, int64_t rows, int64_t C, float eps);
int ns_layer_norm(ns_gpu* g, const float* x, const float* w, const float* b, float* y, int64_t rows, int64_t C, float eps);
/* mHC (docs/glm5next.md): m [T, 24] mixes; X [T, 4, C]; h [T, C]; post [T, 4]; comb [T, 16] */
int ns_hc_pre(ns_gpu* g, const float* m, const float* scale, const float* base, const float* X, float* h, float* post, float* comb,
              float* pre, int64_t T, int64_t C, float eps, int iters);   /* pre [T, 4]: scratch */
int ns_hc_post(ns_gpu* g, const float* y, const float* X, const float* post, const float* comb, float* Xo, int64_t T, int64_t C);
int ns_hc_mean(ns_gpu* g, const float* X, float* y, int64_t T, int64_t C);
/* KDA */
int ns_conv_silu(ns_gpu* g, const float* x, float* state, const float* w, float* out, int64_t T, int64_t D, int k);
int ns_l2_norm(ns_gpu* g, float* x, int64_t rows, int64_t n, float eps);
int ns_kda_gate(ns_gpu* g, float* gate, const float* dt_bias, const float* A, int64_t T, int64_t H, int64_t dh, float low);
int ns_sigmoid(ns_gpu* g, float* x, int64_t n);
int ns_kda_scan(ns_gpu* g, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* S, float* o,
                int64_t T, int64_t H, int64_t d);
int ns_kda_out(ns_gpu* g, const float* o, const float* gate, const float* w, float* y, int64_t T, int64_t H, int64_t d, float eps);
/* FFN, MLA, experts */
int ns_swiglu_clamp(ns_gpu* g, const float* gate, const float* up, float* out, int64_t n, float limit);
int ns_mla_attend(ns_gpu* g, const float* qa, const float* c, float* u, int64_t T, int64_t H, int64_t L, int64_t pos0, float scale);
int ns_scatter_add(ns_gpu* g, float* y, const float* src, const int32_t* idx, const float* w, int64_t n, int64_t C);
int ns_gather(ns_gpu* g, const float* src, const int32_t* idx, float* out, int64_t n, int64_t C);
int ns_add(ns_gpu* g, float* y, const float* x, int64_t n);
/* MoE combine per token: y[t] += sum_{j in [t_ptr[t], t_ptr[t+1])} w[j] * rows[ent[j]], rows [entries, C] */
int ns_moe_combine(ns_gpu* g, float* y, const float* rows, const int32_t* t_ptr, const int32_t* ent, const float* w, int64_t T, int64_t C);

/* ---- decode-width products from the stored blocks (mmvq.cpp) ---- */
size_t ns_q8_1_bytes(int64_t n_in, int64_t ncols);
int ns_mmvq_supported(int type);
/* x [ncols, n_in] float32 -> Q8_1 (36 bytes per 32 values, column j at j * n_in / 32 blocks) */
int ns_quantize_q8_1(ns_gpu* g, const float* x, void* q8_1, int64_t n_in, int64_t ncols);
/* y [ncols, n_out] = W [n_out, n_in] . x for 1..8 columns; W in its ggml type */
int ns_mmvq(ns_gpu* g, int type, const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t n_out, int64_t ncols);
/* a layer's routed experts in two launches (mmvq.cpp): see there */
int ns_moe_grouped_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
size_t ns_moe_scratch_bytes(int64_t entries, int64_t n_ff);
int ns_moe_grouped(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t groups, int64_t entries, const void* x_q8_1,
                   void* scratch, float* out, float limit);

#ifdef __cplusplus
}
#endif
#endif
