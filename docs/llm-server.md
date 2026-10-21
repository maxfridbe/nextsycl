# The language-model server

[← nextsycl](../README.md)

- **Two cards:** the layers split over the GPUs (each with a SYCL context of its own), the experts in an exclusive
  VRAM / pinned-RAM store.
- **MTP speculative decoding:** the model's own draft block drafts a token, a pass verifies it with the next.
  Verify passes are bit-identical to one-token decode (2 to 5 rows, short and long context: `nextsycl spec-check
  --rows`), so greedy output with MTP equals greedy output without it. At a temperature the draft is sampled from
  the draft block's own distribution and kept with min(1, p/q) (speculative sampling): what is committed is
  distributed exactly as plain sampling, and more drafts are accepted when the model is unsure (61.6% -> 65.0% over
  three prompts at 0.7 / 1.0). Qwen3.8-Flash-Next draws a sampled request's tokens on the GPU (Philox(seed,
  position), the top 64 tokens before top-p) and draws each draft from the draft layer with the uniform of the row
  that will verify it (Strata's coupled draft sampling): 69-70% of drafts accepted at 0.7 / 1.0 instead of 60-64%,
  the text exactly the draws, and the same `seed` the same text. Prompt-lookup drafts (`NS_NGRAM`) are there too, exact, off: on this MoE a longer
  verify pass costs nearly its rows (each brings its own experts).
- **MTP on the dense Qwen3.8-27B too:** its MTP file's draft block proposes 3 tokens a step (chained on its own
  output), one pass verifies them, the DeltaNet states roll back from the pass's snapshots - decode about doubles
  (26 -> 43-49 tok/s on the 4-bit file, 18.5 -> 42 on Q8_0), the same tokens as without (`spec-check`: 0 difference).
- **Kernels tuned per model, in silos:** the shared kernels (`kernels/ns`, Strata's, the diffusion ones) stay as the
  engines on them were tuned; a model's faster versions live in its silo (`kernels/<kind>/<arch>/silo/`, its own
  library `dist/silo/libnextsycl-<arch>.so`), which its engine opens at load and uses for what they cover - the
  shared kernels for the rest, or for everything with `NS_SILO=0` (CONTRIBUTING.md rule 6).
- **Long context, to 256K:** the DSA indexer past 2,048 tokens (each row attends to its top 512 pools of 4 tokens and
  its own unfinished pool), checked against llama.cpp; a context per session (`--ctx 262144,32768` / `NS_CTX`: one
  long session and a short one; default two of 64K). A 253K-token prompt reads in 249 s (~1,020 tok/s) and decodes at
  ~15 tok/s at that depth; a follow-up on it restores its checkpoint in seconds. The model is trained to 1M tokens
  (no RoPE in its MLA); a passphrase is found at 10 / 50 / 90% of 32K-249K documents (`bench --needle`, 9 of 9). The
  latent cache in q8 (544 bytes a token and layer; `NS_KV=f16` for fp16's 1,024 - the same needle result, benchy
  the same or a little faster with q8: `docs/benchy/v1-2026-10-07-q8.md`). Plan and measurements: `docs/256k-context.md`.
- **A prompt path on the XMX units:** prompts are read in chunks of 6,144 tokens, with the dense matrices and MLA's
  attention in fp16 through oneMKL's half GEMMs, and the experts that live in host memory copied on a second queue
  while the previous ones compute. The experts with few tokens skip the fp16 copy: their 2-bit weights are decoded
  straight into the matrix units - gate | up in ESIMD (`xmx::dpas`, a work-group sharing each block's decode through
  local memory; up to 256 tokens on a B70, 128 on a B65), down in `joint_matrix` (up to 128); larger experts are
  expanded to fp16 and multiplied by oneMKL. A prompt over a chunk runs the two GPUs as a pipeline.
- **A prompt cache:** checkpoints of the whole conversation state in host memory, at the end of a prompt's first
  turn, at the start of its last user turn, and at its end; those pushed out of memory go to disk
  (`NS_CACHE_DIR`, default `~/.cache/nextsycl/prompts`, `NS_CACHE_DISK_GIB` 32), and a stop writes the rest there -
  a 256K prompt's checkpoint is ~3.5 GiB, mounted again in seconds instead of re-reading the prompt for minutes.
  The files outlive the server (another context size of the same model and cache form takes them); one unused for
  `NS_CACHE_TTL_HOURS` (24) is removed, and past the size budget the least recently used go first.
- **An OpenAI-compatible server** with streaming and the thinking split out, run as a service: `nextsycl llm start`,
  `stop`, `status`, `ps`, `cache`, `chat`, `logs`, over a control socket.
- **Tool calls (Qwen3.8-Flash-Next):** OpenAI's `tools` go into the model's own template (its `<tools>` list and
  `<tool_call><function=..><parameter=..>` format); the answer's calls come back as `tool_calls`, each argument typed
  by the tool's schema, with `finish_reason: "tool_calls"`, streamed or not; `tool` messages and earlier calls are
  written back in the template's form. The text before a call streams as it comes. GLM-5.3 takes no tools yet.
- **Energy and logprobs in every answer:** `usage.energy_wh` (and `energy_wh` on `/api/chat`'s last line) - the watt-hours
  both cards drew for the request; `logprobs: true` (+ `top_logprobs`, up to 20) returns each answer token's
  log-probability and the likeliest alternatives, as OpenAI's `choices[0].logprobs.content`, streamed or not.
- **Several requests at once:** up to `NS_PARALLEL` (2) conversations decode together - one pass carries a token of
  each, the weights read once for all of them, each row exactly as its own pass would be (`nextsycl batch-check`);
  a single request decodes alone, with the draft block. More wait in order.
  A long prompt is read in one pass that stops at a chunk's end when another request arrives; then in groups of
  chunks with the others' decode steps between them (a chat sent during a 253K prompt's read is answered in ~10 s,
  at ~18 tok/s).
- **LogProbChain** (`logprob_chain: true`, experimental): each answer token's logprob also chained through the attention
  to the turn's own earlier tokens - an answer that only repeats its thinking counts only as sure as the thinking was
  (below).
- **A record of each request:** `nextsycl llm inspect <id>` prints one as JSON (settings, timings, previews of the prompt
  and the answer); the server keeps the last `NS_KEEP_REQUESTS` (100).
- **`POST /api/chat` for web pages:** the same chat as JSON lines (`{"thinking": ...}`, `{"content": ...}`, then a
  `{"done": true, ...}` line with the timings), with CORS for loopback pages and the origins in `NS_CORS`.


What the commands show during and after a chat (GLM-5.3-Flash IQ2 on an Arc Pro B65 and an Arc Pro B70, the B70
last):

```
$ nextsycl llm status
glm-5.3-flash-uncensored - up 1m44s, context 65536, MTP on, 2 request(s) served

GPU  CARD                      VRAM USED      FREE   LAYERS EXPERTS VRAM/HOST   TEMP   VRAM   POWER
1    Arc Pro B65        30.1GiB / 31.9GiB    1.8GiB     0-21      3784 / 2080    36C    38C     77W
0    Arc Pro B70        30.1GiB / 31.9GiB    1.8GiB    22-44      3633 / 3496    42C    40C    105W
                                                                       the GPUs draw 183 W

request #2 (socket): generating, prompt 23 tokens, 0 reused (none), 166 / 400 generated at 20.5 tok/s, 9s, 1565 J so far
prompt cache: (busy)

$ nextsycl llm ps
ID     VIA     STATE        PROMPT           REUSED    READ  GENERATED   TOK/S    ENERGY  AVG W FINISH       AGO
#2     socket  done             23           0 none    0.8s        169    20.2    1613 J    175 stop         34s
#1     socket  done             20           0 none    1.1s        195    19.5    1903 J    171 stop       2m08s

$ nextsycl llm inspect 2
{
  "answer": {
    "chars": 873,
    "preview": "The sky appears blue because of a phenomenon called Rayleigh scattering. Sunlight is made up of all the colors of the visible spectrum, and as it passes through Earth's atmosphere, it collides with molecules of nitrogen and oxygen—particles much smaller than the wavelengths of visible light. Shorter..."
  },
  "api": "/v1/chat/completions",
  "avg_watts": 174.8,
  "checkpoints_saved": 0,
  "drafts": [75, 94],
  "ended": 1791318178,
  "energy_j": 1613.1,
  "finish": "stop",
  "generate_seconds": 8.385545438,
  "generated": 169,
  "id": 2,
  "last_user": "Explain in a paragraph why the sky is blue.",
  "max_tokens": 400,
  "messages": 1,
  "model": "glm-5.3-flash-uncensored",
  "prompt_chars": 43,
  "prompt_tokens": 23,
  "read_seconds": 0.843844816,
  "reused": 0,
  "settings": {"effort": "low", "max_tokens": 400, "stream": true, "temperature": 1.0, "top_p": 0.95},
  "source": "none",
  "started": 1791318169,
  "state": "done",
  "thinking": {"chars": 0, "preview": ""},
  "tok_s": 20.153727774720338,
  "via": "socket"
}
```

TEMP and VRAM are the cards' package and memory temperatures, POWER each card's draw over the last second, and
ENERGY what both cards drew while the request ran (idle power included) - from the xe driver's sensors. `drafts` is
MTP's accepted / proposed drafts; `reused` the prompt tokens the prompt cache or the live session already held.
Every API answer carries the same energy as `usage.energy_wh` (here 0.448 Wh).
