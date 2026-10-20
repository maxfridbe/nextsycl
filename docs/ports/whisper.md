# Whisper large-v3-turbo (speech to text)

`audio/whisper`, from nextcuda (its 2026-10-19 port): openai-whisper's `transcribe` with `word_timestamps`, greedy with
its fallback - the log-mel, the 32-layer encoder and 4-layer decoder in half on the shared diffusion kernels (no
kernel of its own: nsd's products, norms, GELU, attention, the 1-D convolution, and MiniMax Music's transpose), the
timestamp rules and 30 s long-form loop, words timed by the six alignment heads (DTW), the language detected.

Use: `audio.transcribe` at the front door (a job, as `audio.speak`), `/v1/audio/transcriptions` (JSON: `audio` base64
or a data URL; OpenAI's `whisper-1` name and `timestamp_granularities` taken), `nextsycl audio transcribe FILE [--json]`.
Settings `NS_WH_CONDITION`, `NS_WH_NO_SPEECH`, `NS_WH_FALLBACK` (also per-request options). Card:
`model-cards/audio/whisper-large-v3-turbo.yml` (1.6 GB).

## Checked (2026-10-19, B70)

Against openai-whisper itself on the same weights, on the CPU in float32 (`reference/whisper/ref.py`;
`nextsycl audio check <dump> --dir <checkpoint>`), four recordings: JFK's inaugural line (11 s, whisper's own test
file), a 27.7 s voice sample, the same voice and two others back to back (81 s, three windows), a 60 s song.

| recording | encoder rel | logits rel | greedy tokens | word error | segments | words in step | word times off (max) | took |
|-----------|-------------|------------|---------------|-----------|----------|---------------|----------------------|------|
| jfk 11 s | 1.1e-2 | 1.2e-3 | 31/31 | 0.0 % | 1 = 1 | 22/22 | 0.000 s | 0.20 s |
| voice 27.7 s | 1.8e-2 | 2.1e-3 | 126/126 | 0.0 % | 6 = 6 | 95/95 | 0.000 s | 0.60 s |
| three voices 81 s | 8.5e-3 | 2.6e-3 | 87/87 | 0.0 % | 15 = 15 | 222/222 | 0.020 s | 1.61 s |
| song 60 s | 9.6e-3 | 1.1e-3 | 40/40 | 0.0 % | 14 = 14 | 75/75 | 0.000 s | 0.73 s |

(The mel within 8e-7; the encoder's error is half precision's.) Load 0.3 s from the page cache; about 2.0 GiB on the card
with the server loaded.
