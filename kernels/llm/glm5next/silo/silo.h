/* silo.h: the glm5next engine's silo - kernels for the files the shared ones (glm.h) do not cover, in their own
 * library (dist/silo/libnextsycl-glm5next.so, opened by the engine at load; without it the engine runs on the shared
 * kernels alone, which read the IQ2 files). The REAP-50 GGUFs (IQ3_M, Q3_K_M, IQ4_XS, Q4_K_S, Q4_K_M) store their
 * matrices in IQ3_S / IQ4_XS / IQ4_NL / Q4_0 / Q5_0 and their experts' down in IQ3_S or Q6_K. Each op says what it
 * covers (`_supported`); the engine keeps the shared kernel for everything else. */
#ifndef NS_LLM_GLM5NEXT_SILO_H
#define NS_LLM_GLM5NEXT_SILO_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* n values of a ggml-typed tensor into float32 (dequant.cpp; the shared one is ns_dequant): ggml's dequantize_row_*
 * arithmetic, a work-item a block. Covers Q4_0 (2), Q5_0 (6), IQ4_NL (20), IQ3_S (21), IQ4_XS (23); n a multiple of
 * the type's block (32, or 256 for IQ3_S / IQ4_XS) */
int ns_llm_glm5next_dequant_supported(int type);
int ns_llm_glm5next_dequant(ns_gpu* g, int type, const void* src, size_t n, float* dst);

/* n values (a multiple of 256) of a Q6_K matrix into half, for a prompt chunk's half GEMMs (the shared
 * ns_dequant_f16 takes Q6_K only in a build with STRATA_Q6K_EXPERTS): each value ggml's, rounded to half */
int ns_llm_glm5next_dequant_f16_supported(int type);
int ns_llm_glm5next_dequant_f16(ns_gpu* g, int type, const void* src, int64_t n, uint16_t* dst);

/* a layer's routed experts at decode widths (moe.cpp) for the type pairs ns_moe_grouped does not take: gate | up
 * IQ3_S or Q4_K with down IQ3_S or Q6_K (the IQ3_M and Q4_K_M files). ns_moe_grouped's contract: the expert blobs
 * [gate | up | down] at grp_ptr, grp_start [groups + 1], n_groups (one int), ent_dst / ent_tok [entries], x_q8_1 the
 * tokens' q8_1 rows, out [., n_embd] unweighted rows; limit > 0 clamps the SwiGLU as GLM's; phase 0 both halves, 1
 * gate | up and the activation (into the scratch), 2 down. A row's sums are in an order fixed by the kernel alone (16
 * lanes), so rows do not depend on how many entries a launch has */
int ns_llm_glm5next_moe_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
size_t ns_llm_glm5next_moe_scratch_bytes(int64_t entries, int64_t n_ff);
int ns_llm_glm5next_moe(ns_gpu* g, int gu_type, int d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start,
                        const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t groups, int64_t entries, const void* x_q8_1,
                        void* scratch, float* out, float limit, int phase);

#ifdef __cplusplus
}
#endif
#endif
