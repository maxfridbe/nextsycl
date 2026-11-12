# GLM-5.3-Flash REAP-50 (llm/glm5next)

patrickbdevaney's REAP-pruned GGUFs of zai-org/GLM-5.3-Flash (`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF` @ cc7c2b5e,
MIT): the same architecture as the uncensored IQ2 file the box already serves (`glm5-next`: 45 layers, 34 KDA and 11 MLA
with the DSA indexer, mHC, 3 dense layers, one MTP block), with half of each MoE layer's routed experts removed by REAP
(144 of 288 kept; 8 active, 1 shared, expert FFN 2,048). The files come from the publisher's llama.cpp fork (branch
`glm5_next`; its patch and bundle sit beside the GGUFs - read, not run); stock llama.cpp does not load them. The engine
is the existing one (`llm/glm5next`) with a small delta, and the engine's first silo. Cards:
`model-cards/llm/glm-5.3-flash-reap50-*.yml`.

## What differs from the files the engine read before

From the headers (read with `python3 -I`, the first 40 MB of each file) and the fork's patch:

| | llama.cpp (PR 27773) / ds4 files | the REAP-50 files |
|---|---|---|
| routed experts | 288 | 144 (`expert_count`; the router, its bias and the expert tensors are 144 wide) |
| hyper-connection tensors | `hc_attn_fn.weight` ... | `hc_attn_fn` ... (no `.weight`) |
| indexer pool | `indexer_compressor_{ape,gate}.weight` / `indexer.pool_*` | `indexer.kpool_{ape,gate}`, no pool-size key (4: the `ape`'s rows) |
| hc metadata | `hyper_connection.{count,sinkhorn_iterations,epsilon}` | `attention.hc.{mult,sinkhorn_iters,eps}` |
| KDA gate floor | `linear_attention.gate_lower_bound` / `kda.gate_lower_bound` | `ssm.gate_lower_bound` |
| the MTP block | ds4: `trunk_block_count`; llama.cpp: none | counted in `block_count` (46 = 45 + `nextn_predict_layers` 1) |
| KDA decay `ssm_a` | -exp(A_log) (llama.cpp) / A_log (ds4) | exp(A_log) (the fork's graph: `bound * sigmoid(exp(A_log) (w + dt_bias))`; the file's values are 1.4-8.2, all positive) |
| stop tokens | | the header lists one EOS (154820, `<|endoftext|>`); `<|user|>` and `<|observation|>` stop too - nextsycl_tok already adds them for GLM |

`Scheme::Fork` (llm/glm5next/src/model.rs) maps the roles onto these names; the decay's form is `Scheme::a_form`, applied
once at load. `nextsycl llm info` on the IQ3_M file: every tensor present with the shape the runtime needs, none left
over.

Tensor types (bytes of tensors):

| file | size | routed experts (gate / up / down) | attention, shared, dense | head | embedding |
|---|---:|---|---|---|---|
| IQ3_M | 67.17 GiB | IQ3_S / IQ3_S / IQ3_S (2 layers Q4_K) | IQ3_S, Q4_K, IQ4_NL (KDA f_b / g_b) | Q6_K | IQ3_S |
| Q3_K_M | 73.36 GiB | Q3_K / Q3_K / Q4_K | Q3_K, Q4_K, Q5_K, Q4_0 | Q6_K | Q3_K |
| IQ4_XS | 81.99 GiB | IQ4_XS | IQ4_XS, IQ4_NL | Q6_K | IQ4_XS |
| Q4_K_S | 87.09 GiB | Q4_K / Q4_K / Q4_K (2 layers Q5_K) | Q4_K, Q5_K, Q5_0 | Q6_K | Q4_K |
| Q4_K_M | 92.50 GiB | Q4_K / Q4_K / Q4_K or Q6_K (19 layers) | Q4_K, Q6_K, Q5_0 | Q6_K | Q4_K |

## The silo (kernels/llm/glm5next/silo, dist/silo/libnextsycl-glm5next.so)

The engine's shared kernels read what the IQ2 file stores; the silo adds what these files need and nothing else
(`silo.h`; each op says what it covers, the engine keeps the shared kernel for the rest):

- `dequant.cpp`: Q4_0, Q5_0, IQ4_NL, IQ3_S, IQ4_XS into float32 (the load's MLA k_b / v_b and MTP eh_proj, a token's
  embedding row, small prompt products) and Q6_K into half (prompt chunks' half GEMMs) - ggml's `dequantize_row_*`
  arithmetic, a work-item a block.
- `moe.cpp`: a layer's routed experts at decode widths for the pairs the shared grouped kernels (Strata's
  `native_expert_grouped`) do not take - IQ3_S or Q4_K gate | up with IQ3_S or Q6_K down (the IQ3_M and Q4_K_M files).
  `ns_moe_grouped`'s contract (blob, table, phases), ggml's `vec_dot_*_q8_1`, a 16-lane sub-group a row: a row's sums
  in an order fixed by the kernel alone, so verify rows equal one-token rows. `NS_SILO_MOE=0` turns it off (each
  expert's products one at a time through the shared mmvq: the check below).

The Q3_K_M, IQ4_XS and Q4_K_S files run on the shared kernels for everything but their small dequant types.
