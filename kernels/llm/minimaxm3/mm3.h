/* mm3.h: the minimaxm3 engine's kernels (MiniMax-M3.1-Flash) - its own C ABI, bound by its crate by name
 * (llm/minimaxm3/src/ffi.rs). The bring-up kernels: exact and plain, float32 activations, checked against llama.cpp's
 * dumps (docs/ports/minimax-m31.md); the generic steps (norms, GEMMs, dequantization, gather / scatter) are the
 * glm5next engine's (glm.h). */
#ifndef NS_LLM_MINIMAXM3_H
#define NS_LLM_MINIMAXM3_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* NEOX RoPE (ggml's rope_type NEOX: value i pairs with i + n_rot / 2) on the first n_rot values of each head, in place:
 * x[t * row + h * head + i], t < T at position pos0 + t, h < H; theta_i = pos * base^(-2 i / n_rot) as ggml builds it
 * (the running product) */
int ns_llm_minimaxm3_rope(ns_gpu* g, float* x, int64_t T, int64_t H, int64_t row, int64_t head, int64_t n_rot, int64_t pos0, float base);

/* swiglu-oai (gpt-oss's, llama.cpp's LLM_FFN_SWIGLU_OAI_MOE): out = min(gate, limit) * sigmoid(alpha min(gate, limit))
 * * (clamp(up, -limit, limit) + 1), n values */
int ns_llm_minimaxm3_swiglu_oai(ns_gpu* g, const float* gate, const float* up, float* out, int64_t n, float alpha, float limit);

/* The indexer's block scores: bs[(t * G + g) * nblk + b] = the best q . k of query row (t, g) (iq [T][G][128] float32)
 * over the keys of block b (ik fp16 [pos][128], 128 keys a block) it sees (positions <= pos0 + t); -inf for a block it
 * sees none of, and + 1e30 for the `local` newest blocks (they always rank first) */
int ns_llm_minimaxm3_idx_scores(ns_gpu* g, const float* iq, const uint16_t* ik, float* bs, int64_t T, int64_t G, int64_t nblk, int64_t pos0, int64_t local);

/* The K best blocks of each (t, g) row of bs (scratch: it is overwritten) -> sel[(t * G + g) * K + k], -1 where fewer than
 * K blocks are visible; the lowest block wins a tie */
int ns_llm_minimaxm3_idx_topk(ns_gpu* g, float* bs, int32_t* sel, int64_t T, int64_t G, int64_t nblk, int64_t K);

/* GQA attention, heads of 128: query row (t, hq) (q [T][H][128]) reads the keys of its group's (hq / (H / G)) selected
 * blocks (sel [T][G][K] ids of 128 keys, -1 none; null: every position up to its own) that are at positions <=
 * pos0 + t; kc, vc fp16 [pos][G][128]; softmax of scale x q.k; out [T][H][128] */
int ns_llm_minimaxm3_attend(ns_gpu* g, const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* sel, int64_t K, float* out, int64_t T,
                            int64_t H, int64_t G, int64_t pos0, float scale);

#ifdef __cplusplus
}
#endif
#endif
