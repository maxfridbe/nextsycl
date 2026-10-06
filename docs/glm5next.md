# GLM5-Next: the math

What nextsycl computes for GLM-5.3-Flash, written down once. Read from llama.cpp de25343 (PR 27773:
`src/models/glm5-next.cpp`, the mHC helpers of `src/models/deepseek4.cpp`, `build_moe_ffn` / `build_ffn` in
`src/llama-graph.cpp`, the CPU ops `gated_delta_net`, `swiglu_clamp`, `dsv4_hc_comb` in `ggml-cpu/ops.cpp`);
llama.cpp is the reference the parity checks run against, not a dependency.

Notation: `x` a token's hidden state [4096]; `rms(x) = x / sqrt(mean(x^2) + 1e-5)` (`rms_eps` from the file);
`W x` a matrix-vector product with the stored [out, in] matrix. Indices are per token; batches are rows.

## Residual streams (mHC)

The stream is 4 copies `X [4, 4096]`, all equal to the token embedding at the start (`hc_init`). Around each half
of a block (attention, then feed-forward), with that half's `fn [24, 16384]`, `scale [3]`, `base [24]`:

```text
m     = fn . rms(flatten(X))                          [24]   (rms over all 16384, eps rms_eps)
pre   = sigmoid(m[0:4]  * scale[0] + base[0:4]) + hc_eps  [4]
post  = sigmoid(m[4:8]  * scale[1] + base[4:8]) * 2       [4]
C     = (m[8:24] * scale[2] + base[8:24]) as [dst 4, src 4], element C[dst][src] = c[dst + 4 src]
C     = softmax over dst (for each src), + hc_eps
        then normalize each dst-row over src (sum + eps), and repeat
        (rows over dst, then columns) until sinkhorn_iterations (20) normalizations of each kind are done:
          norm_cols(); for i in 1..20 { norm_rows(); norm_cols() }
        where norm_cols: for each dst, divide C[dst][*] by (eps + sum over src)
              norm_rows: for each src, divide C[*][src] by (eps + sum over dst)
h     = sum_s pre[s] X[s]                              [4096]  the half's input
y     = half(rms_norm(h) * norm_weight)                [4096]  (attn_norm / ffn_norm)
X'[d] = post[d] * y + sum_s C[d][s] X[s]               new streams
```

After the last block: `out = output_norm * rms(mean_s X[s])`, `logits = output . out`.

## KDA (layers with `layer_types` 0 / `head_count_kv` 0)

64 heads of 128 (`d = 8192`), state `S [64, 128 key, 128 value]` per sequence, plus the last 3 inputs of each
short convolution.

```text
q,k,v = silu(causal_conv4(W_{q,k,v} x))                 conv weights [8192, 1, 4] per channel, over this and
                                                        the previous 3 tokens' projections
q,k   = per head: x / sqrt(sum x^2 + 1e-6)              L2 norm (build_gdn_l2_norm: rms_norm with eps 1e-6/128,
                                                        divided by sqrt(128) - the same thing)
g     = f_b . (f_a . x) + dt_bias                       [8192]
g     = lower_bound * sigmoid(-(g * A[head]))           lower_bound = -5 (gate_lower_bound); A = -exp(A_log)
                                                        (llama.cpp's `ssm_a` already holds -exp(A_log); ds4's
                                                        `kda_a_log` is A_log)
beta  = sigmoid(W_beta x)                               [64]
per head, per token (the fused CPU op; state as S[key i][value j]):
  S[i][:] *= exp(g[i])                                  decay per KEY channel
  delta[j] = beta * (v[j] - sum_i S[i][j] k[i])
  S[i][j] += k[i] delta[j]
  o[j]     = (sum_i S[i][j] q[i]) / sqrt(128)
gate  = g_b . (g_a . x)                                 [8192]
out   = W_o ( per head: rms_norm(o) * o_norm * sigmoid(gate) )
```

## MLA + indexer (the other layers: 3, 7, ..., 43; and the MTP block)

No rotary embedding (`rope_dimension_count` 0). The cache holds per token only `c [512]` (the compressed latent).

```text
qr   = rms_norm(W_qa x) * q_a_norm                      [1536]
q    = W_qb qr                                          [64 heads, 256]
c    = rms_norm(W_kva x) * kv_a_norm                    [512]   cached
q~_h[r] = sum_e k_b[h][r][e] q_h[e]                     k_b [64, 512, 256]: the query absorbed into latent space [512]
s_h(t') = q~_h . c(t') / sqrt(256)                      over the selected cached tokens t'
p_h  = softmax(s_h)
u_h  = sum_t' p_h(t') c(t')                             [512]
o_h  = v_b[h] . u_h                                     (v_b [64, 256, 512])  [256]
out  = W_o concat_h(o_h)
```

Selection (the DSA lightning indexer, `kpool` 4, `top_k` 2048): keys are grouped in pools of 4 consecutive
tokens. Per token: `ik = layer_norm(W_ik x) * k_norm + k_norm_b` [128], `ig = W_ig x` [128]; a completed pool's
key is `sum_m softmax_m(ig_m + ape_m) * ik_m` per channel (softmax over the 4 members). Per query:
`iq = W_iqb qr` [32 heads, 128], `w = W_proj x / sqrt(128 * 32)` [32], pool score `sum_h relu(iq_h . pool) * w_h`;
the top `2048 / 4 = 512` completed pools by score, plus the incomplete tail pool's tokens
(`kpool_select_tail`), are what attention reads. **With fewer than 512 completed pools every token is selected:
up to ~2,048 tokens of context the layer is plain causal attention**, which is how bring-up starts. The GCSA file
marks every MLA layer as running its own indexer (`indexer.types` all 1); a layer marked 0 reuses the previous
one's selection.

## Feed-forward

Dense (layers 0-2): `W_down swiglu10(W_gate x, W_up x)` - clamped like the experts (`build_ffn` takes the limit
from `swiglu_clamp_shexp`, which for this model is the expert limit, 10, on every layer).

MoE (layers 3-44):

```text
p      = sigmoid(W_router x)                            [288]
choice = top-8 of (p + exp_probs_b)                     the bias picks, it does not weigh
w      = p[choice] / max(sum p[choice], 6.1e-5) * 2.5   normalized (expert_weights_norm), scaled
y      = sum_e w_e * W_down_e swiglu10(W_gate_e x, W_up_e x)
       + W_down_sh swiglu10(W_gate_sh x, W_up_sh x)     the shared expert, weight 1
swiglu10(g, u) = silu(min(g, 10)) * clamp(u, -10, 10)
```

## MTP (the ds4 file's block 45)

Read from antirez's ds4 (`glm_graph_mtp_step`, `ds4_session_glm_spec_cycle_impl` in `ds4.c`), the reference for the
ds4 file. A plain pre-norm block - no hyper-connections - fed with position p's final hidden state and the token at
p + 1, predicting the token at p + 2:

```text
h     = mean_s X[s]                                     the trunk's last streams (weights 1/4), before output_norm
cur   = eh_proj . concat(enorm * rms(embed(token[p+1])), hnorm * rms(h))      eh_proj [4096, 8192]
cur  += MLA(rms(cur) * attn_norm)                       its own latent cache (slot p), no rotary embedding
cur  += MoE(rms(cur) * ffn_norm)                        router + 288 experts + shared expert, as the trunk's
draft = argmax(output . (rms(cur) * shared_head_norm))  the shared output head
```

ds4 attends only over the slots written since decoding began; nextsycl also runs the block over the prompt (batched,
one chunk behind), so its cache covers the conversation. Indexer selection for the block's attention past ~2,048
tokens comes with the trunk's.

Verification: `[token, draft]` in one 2-row pass; the token drawn for row 0 is compared with the draft (accepted only
when equal - exact sampling for a point-mass proposal); on a reject the KDA states go back to their snapshot after
row 0. Decode-width products run row by row (oneMKL's GEMM picks its kernel by row count), so a verify pass's rows
are bit-identical to one-token passes: greedy output with MTP equals greedy output without (`nextsycl spec-check`).
