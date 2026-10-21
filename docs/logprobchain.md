# LogProbChain

[← nextsycl](../README.md)

Thinking skews an answer's logprobs: the answer copies what the thinking wrote, so it looks certain. With
`"logprob_chain": true` every answer token's entry also carries `chain`: for it and each alternative, the logprob plus,
over this turn's earlier tokens equal to it, the attention its position gives each times that token's own logprob
(p' = p x prod p_j^a_j). The attention is the mean over MLA's 64 heads and 11 layers of the softmax weights of the pass
that produced the token. An example (`reference/logprob-chain-show.py` prints a response this way):

```
question: Which is larger, 9.11 or 9.9? Reply with just the number.   (reasoning_effort high, greedy)
thinking: '9.11 vs 9.9: 9.9 = 9.90 > 9.11. Answer: 9.9'
answer:   '9.9'

'9'        logprob  -0.0056   chained  -0.0078   attention to this turn 0.257
     echoes turn token #0 '9': attention 0.0121 x its logprob -0.1324
     echoes turn token #12 '9': attention 0.0026 x its logprob -0.2098
     echoes turn token #10 '9': attention 0.0031 x its logprob -0.0035
'.'        logprob  -0.0001   chained  -0.0005   attention to this turn 0.316
'9'        logprob  -0.0000   chained  -0.0023   attention to this turn 0.272
     echoes turn token #0 '9': attention 0.0086 x its logprob -0.1324
     echoes turn token #12 '9': attention 0.0047 x its logprob -0.2098
     echoes turn token #31 '9': attention 0.0226 x its logprob -0.0056
```

The last `9` reads as certain (-0.0000) but chains to -0.0023: most of it from the thinking's two hesitant `9`s.
Requests with it decode one token a pass (no draft block), a few small reads a layer more.

Every entry also carries `chained_logprob` beside its `logprob`. GLM-5.3 and Qwen3.8-27B capture the attention (the
27B: its 16 full-attention layers; `NS_Q35_CHAIN_HEADS=max` takes each position's most attentive head instead of the
heads' mean). A first validation on the 27B (Q4, greedy, thinking; `chainval.py` in the session scratchpad, 58
questions with known answers - facts, arithmetic, counting, primes, powers): 57 right. The one wrong answer (Tallinn
first recorded 1219, for 1154) was raw p 1.0000, chained 0.9997 - the chain moved it, but the right answers moved as
much (mean chained p 0.9993 with the heads' mean, 0.982 with their max on the harder 22). Too few wrong answers to say
the chain tracks correctness better; a set the model gets wrong more often is the next step.
