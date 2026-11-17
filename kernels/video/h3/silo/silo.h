/* silo.h: the H3 video engine's silo (dist/silo/libnextsycl-h3.so): the kernels of Open-H3-Regenerate-2K's fast video
 * decoder (crates/h3/src/fastvae.rs) that the shared diffusion kernels do not have - the shifted-window attention as
 * variable-length sequences (Intel's ARK prefill kernel on sycl-tla, bfloat16, heads of 64), the per-block global
 * tokens' attention over a whole block (a split-key pass with running softmax state), and an indexed row copy (the
 * window order in and out). Built apart from the kind's library with sycl-tla's flags (kernels/build.sh: the silo's tla/
 * sources), so it does not use the shared part's ns_gpu: calls take the GPU's SYCL queue (nextsycl-core's Gpu::queue,
 * the queue the decoder's diffusion context is created on) and say why they failed in ns_video_h3_silo_error(). Every
 * call is queued on that queue and does not wait. The engine (video/h3/src/silo.rs) binds these when the library is
 * there; without it the regen2k job decodes with the full H3 decoder. */
#ifndef NS_VIDEO_H3_SILO_H
#define NS_VIDEO_H3_SILO_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* ns_video_h3_silo_error(void);

/* Whether the silo's kernels run on this queue's GPU (1: compiled ahead for Xe2, the B70 / B65's matrix engines and 2-D
 * block loads) or the engine must decode with the full H3 decoder (0) */
int ns_video_h3_silo_supported(void* queue);

/* Indexed row copy of 16-bit elements: dst row (didx ? didx[i] : i) = src row (sidx ? sidx[i] : i), i < n, `width`
 * elements a row, rows `sstride` / `dstride` elements apart. width a multiple of 8 and both strides too (16-byte moves). */
int ns_video_h3_silo_rows(void* queue, const void* src, int64_t sstride, const int32_t* sidx, void* dst, int64_t dstride,
                          const int32_t* didx, int64_t n, int64_t width);

/* Attention over `seqs` independent sequences of different lengths, bfloat16, heads of 64, no mask:
 * sequence b's queries are rows [cu_q[b], cu_q[b + 1]) of q and out, its keys and values rows [cu_k[b], cu_k[b + 1])
 * of k and v; every row H * 64 values. cu_q, cu_k, cu_zero: device int32 [seqs + 1] (cu_zero all zeros); max_q, max_k:
 * the longest sequence's; total_q, total_k: cu_q[seqs], cu_k[seqs]. softmax(q k^T / 8) v. */
int ns_video_h3_silo_varlen(void* queue, const void* q, const void* k, const void* v, void* out, int64_t seqs, const int32_t* cu_q,
                            const int32_t* cu_k, const int32_t* cu_zero, int64_t max_q, int64_t max_k, int64_t total_q, int64_t total_k,
                            int64_t H);

/* The running state of S query rows' attention over keys given a part at a time: float [H, S, 66] - the scores' max, the
 * sum of their exponentials, the unnormalized output (64). init: max -inf, the rest 0 (count = H * S). */
int ns_video_h3_silo_glob_init(void* queue, float* state, int64_t count);

/* S <= 8 query rows qg (bfloat16, rows `qs` elements apart, H heads of 64 each) over n more keys and values k, v (rows
 * `kvs` elements apart): the state updated as if those keys had been part of one softmax with the ones before.
 * scratch: float, at least H * S * ceil(n / 256) * 66. */
int ns_video_h3_silo_glob(void* queue, const void* qg, int64_t qs, const void* k, const void* v, int64_t kvs, int64_t n, int64_t H,
                          int64_t S, float* state, float* scratch, int64_t scratch_floats);

/* The state's result: out row s (bfloat16, rows `os` elements apart) = acc / sum, H heads of 64 */
int ns_video_h3_silo_glob_out(void* queue, const float* state, int64_t H, int64_t S, void* out, int64_t os);

#ifdef __cplusplus
}
#endif
#endif
