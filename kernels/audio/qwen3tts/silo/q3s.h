/* q3s.h: Qwen3-TTS's silo (dist/silo/libnextsycl-qwen3tts.so): its frame on the GPU end to end - the talker's and the
 * code predictor's decode in fused kernels (an RMS norm read into the product before it, SwiGLU taken out of the
 * gate / up product, q / k norm + rotary positions + the cache's store in one kernel, the predictor's attention whole
 * in one), the draws on the device (repetition penalty, suppressed ids, temperature, top-k, the seeded draw), and a
 * frame recorded once as a SYCL command graph and replayed. Everything that changes from frame to frame (the talker's
 * position, the frame's index, the codes drawn) is read from device memory, so one recording serves every frame of a
 * request. Float32 activations, half or int8 (a scale a row) weights, a half cache. 0 or -1 (ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when this silo's kernels take these shapes: heads of 128, `group` query heads a key head (1, 2, 4 or 8), products
 * whose K is a multiple of 256 */
int ns_q3s_supported(int64_t head, int64_t group, int64_t k);

/* gemv flags */
#define NS_Q3S_ACC 1    /* out += x . W^T (a residual) instead of out = */
#define NS_Q3S_NORM 2   /* each x row RMS-normed with norm_w (eps) first */
#define NS_Q3S_SWIGLU 4 /* W is [gate; up] (N rows: N/2 each): out[m, n] = silu(gate_n . x) * (up_n . x), N/2 wide */

/* out[m] = x[m] . W^T (* scale + bias), m < M <= 8; W [N, K] half (wt 0) or int8 (wt 1, scale a row); out rows ldo apart */
int ns_q3s_gemv(ns_gpu* g, const float* x, int64_t M, int64_t K, const void* w, int wt, const float* scale, int64_t N, const float* bias,
                float* out, int64_t ldo, int flags, const float* norm_w, float eps);

/* Floats of scratch ns_q3s_attn needs for `splits` */
int64_t ns_q3s_attn_scratch(int64_t Hq, int64_t splits);
/* The talker's attention for its one row, whole: q / k of the qkv row RMS-normed per head and rotated (rotate-half,
 * 128 wide) at the position *pos (device), k and v stored (half) at that position of the cache [Hkv, T, 128], then
 * the H query heads over the cache's keys 0..*pos in `splits` parts (each ceil((pos+1)/splits) keys), merged by the
 * last part to finish (counters: Hkv zeroed ints, left zeroed); out [H x 128] */
int ns_q3s_attn(ns_gpu* g, float* qkv, int64_t H, int64_t Hkv, const float* q_norm, const float* k_norm, float eps, const float* inv_freq,
                const int32_t* pos, void* kc, void* vc, int64_t T, int64_t splits, float* out, float* part, int32_t* counters);

/* The code predictor's attention, whole: S <= 2 new rows of qkv (ws floats apart) at positions p0.., q / k RMS-normed
 * and rotated, k / v stored into the cache [Hkv, T, 128], each row's Hq heads attending causally; out [S, Hq x 128] */
int ns_q3s_attn_small(ns_gpu* g, float* qkv, int64_t ws, int64_t S, int64_t p0, int64_t H, int64_t Hkv, const float* q_norm, const float* k_norm,
                      float eps, const float* inv_freq, void* kc, void* vc, int64_t T, float* out);

/* A request's draws, in device memory (written before its frames, read by the graph) */
typedef struct ns_q3s_draws {
    int32_t sample[2];       /* talker, predictor: 0 the best code, 1 a draw */
    float temperature[2];
    int32_t top_k[2];
    float penalty;           /* the talker's repetition penalty over its codes so far */
    int32_t suppress_from;   /* the talker's ids from here are not drawn (but eos) */
    int32_t eos;
    int32_t min_frames;      /* eos held back for these first frames */
    int32_t forced;          /* 1: each code taken from force[frame][16] (a check), the misses counted */
    int32_t n_text;          /* text rows fed a frame each; then the pad row */
} ns_q3s_draws;

/* A request's frame state, in device memory */
typedef struct ns_q3s_state {
    int32_t frame;           /* the frame being made */
    int32_t pos;             /* the talker's position of the frame's input */
    uint32_t rng[2];         /* splitmix64's state */
    int32_t codes[16];       /* the frame's codes */
    int32_t misses[2];       /* forced: first codes, other codes the greedy draw would not have taken */
} ns_q3s_state;

/* Codebook g's code drawn from logits [n]: for g 0 (the talker's) the repetition penalty over seen[], the suppressed
 * ids, eos held back; then the best or a draw (temperature, top-k, splitmix64). Writes st->codes[g]; g 0 marks seen;
 * `row` (optional) gets the projected table's row of the code, ptab[base + code] [width] (the predictor's next input) */
int ns_q3s_draw(ns_gpu* g, const float* logits, int64_t n, int gi, const ns_q3s_draws* d, ns_q3s_state* st, uint8_t* seen,
                const int32_t* force, const float* ptab, int64_t base, int64_t width, float* row);

/* The talker's next input: text row min(frame, n_text) of feed [n_text + 1, D] plus the frame's sixteen codes'
 * embeddings (table half [rows, D]: codebook 0 at 0, codebook g at v0 + (g - 1) * vg) */
int ns_q3s_frame_in(ns_gpu* g, const void* table, int64_t D, int64_t v0, int64_t vg, const ns_q3s_state* st, const ns_q3s_draws* d,
                    const float* feed, float* out);

/* After a frame: frame + 1, pos + 1 */
int ns_q3s_advance(ns_gpu* g, ns_q3s_state* st);

/* Graphs: the work queued between begin and end on the GPU's queue recorded (not run) into a graph, replayed by launch */
int ns_q3s_graph_begin(ns_gpu* g, void** graph);
int ns_q3s_graph_end(ns_gpu* g, void* graph, void** exec);
int ns_q3s_graph_launch(ns_gpu* g, void* exec);
void ns_q3s_graph_free(void* exec);

/* bytes from device memory to host memory, queued (ordered after the queue's work; ns_mark / ns_await to wait) */
int ns_q3s_read_async(ns_gpu* g, void* host, const void* src, size_t bytes);

#ifdef __cplusplus
}
#endif
