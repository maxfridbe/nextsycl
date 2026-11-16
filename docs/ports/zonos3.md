# ZONOS3-Mini: port brief

Zyphra's ZONOS3-Mini (weights Apache-2.0, `Zyphra/ZONOS3-Mini` on Hugging Face, `gguf/` at revision d2d142a8; the
reference runtime github.com/Zyphra/ZONOS3-Mini is MIT) as the audio engine `zonos3` (`audio/zonos3`,
`kernels/audio/zonos3/z3.cpp`), card `zonos3-mini`. Speech in 43 languages at 44.1 kHz: 27 preset voices, a recording
cloned (by its speaker embedding or by its own codes), steering, Zyphra's markup, LACI, short texts, streaming, and
speech editing (`audio.edit`).

## Sources and how it was ported

- **Weights**: Zyphra's own GGUFs, 106 files pinned with their SHA-256s in `model-cards/audio/zonos3-mini.yml`
  (10.5 GB with the full decoder and the encoders): `backbone-ptq-edit.gguf` (the model: Q4_0 experts, Q8_0 mixers,
  the tokenizer and condition tables inside; on plain text the release LM bit for bit), `zac-keep2.gguf` (decoder),
  `zac.gguf` (decoder-full, loaded on first use), `spk-encoder.gguf`, `steer.gguf`, `zac-encoder.gguf` +
  `ssl-teacher.gguf` (raw-reference cloning, loaded for an encode and freed), `edit/accents.gguf` (the speech-edit
  LoRA), FuseAlign (`Zyphra/fusealign` a4d58de4, Apache-2.0), 43 LACI head files and the 27 voices' `.codes.npy` /
  `.embedding.npy`. Read by our own Rust GGUF reader (`crates/gguf`); no ggml or Zyphra C++ in the program.
- **Design and host code**: our sister project nextcuda's port (maxfridbe/nextcuda at faa02ae, `audio/zonos3`,
  `docs/ports/zonos3.md` there), itself written from Zyphra's three runtimes (PyTorch reference, the ggml server, the
  MLX server). The Rust host code is that crate on nextsycl's foundation (the same contract shapes; streaming through
  `generate_streamed`); the GPU code is new: every one of nextcuda's 53 CUDA kernels rewritten in SYCL
  (`kernels/audio/zonos3/z3.cpp`: sub-groups of 32 where the CUDA code reduced over a warp, work-group reductions for
  its block sums, deterministic group sums in LACI's windowed attention). The float32 products are the shared
  diffusion library's (oneMKL), the decoder's layer scales Qwen3-TTS's scaled add. Text normalization is
  `zyphra-textnorm` (crates.io, pure Rust); the SuperBPE tokenizer is `nextsycl-tok` with a new constructor that keeps
  the text between the Split pattern's matches (`from_hf_json_text_keep_gaps`).
- **Outside reference for parity**: Zyphra's ggml runtime built for the CPU in a throwaway checkout
  (`servers/ggml`: zonos3-cli, zac-cli, spk-encoder-cli, zac-encoder-cli) and FuseAlign's `fusealign-cli`;
  `reference/zonos3/ref.sh` makes the dumps, `reference/zonos3/compare.py` compares;
  `nextsycl audio check DIR --model zonos3-mini --stages prompt,lm,forced,decoder,speaker,encoder,stream,align,edit`.

## Parity (B70, against Zyphra's ggml runtime on the CPU, the same GGUFs)

nextcuda's numbers (RTX 5070, the same comparisons) in brackets.

| stage | result |
|---|---|
| prompt ids (Savannah's embedding: speaker row, conditions, SuperBPE text, silence) | identical |
| markup and normalized text, English (`[start]`, `[ipa təmɑːtoʊ]`, `[pause 2]`, `[pause]`, money, a date, units; `[steer]` tags removed on Zyphra's side: steering is not in the ids) and German (a date, money, `[pause 0.5]`) | identical (65 and 54 rows) to `zonos3-cli --request` given the model's own markers (`reference/zonos3` markers.py) |
| backbone logits, the 27-row embedding prompt (`lm`) | cosine 0.999904 mean, 0.99978 worst row, rel 1.2e-2, argmax 98.1 % [0.99969, 0.9959, 2.1e-2, 96.9 %] |
| teacher-forced, a 70-step take of the reference's (`forced`) | cosine 0.999888 mean, 0.99932 worst, rel 1.5e-2, argmax 92.4 %; the take's drawn codes in our top-80 100 % [0.99984, 0.9984, 1.8e-2, 92.9 %, 99.8 %] |
| decoder PCM, 62 frames (`decoder`, keep2) | cosine 1.000000, rel 5.1e-6, max 3.1e-5 [rel 3.3e-5] |
| speaker embedding, Rupert's 13 s (`speaker`) | cosine 1.000000, rel 1.2e-6 [1.2e-6] |
| reference encoder, Rupert's 13 s, 198 frames (`encoder` vs `zac-encoder-cli`) | codes the same in 97.0 / 95.5 / 87.9 / 87.4 / 82.3 / 76.3 / 72.7 / 72.2 / 74.2 / 71.2 % by codebook [97.5 / 95.5 / 89 / 85 / 86 / 80 / 79 / 78 / 76 / 77]; against Zyphra's shipped codes ours 96.0 / 91.9 / 87.4 / 83.3 / 77.3 / 75.3 / 67.7 / 66.7 / 64.6 / 68.2, the reference's own 98.0 / 95.5 / 88.9 / 87.9 / 81.3 / 81.8 / 78.3 / 75.3 / 75.8 / 76.8 |
| FuseAlign, Zyphra's Cillian recording (9.6 s, 22 words; `align` vs `fusealign-cli` on the CPU) | every start and end within 1 ms, presence and placement equal [the same] |
| streaming, the same take whole and streamed (`stream`, option level off) | rel 1.9e-6, max 5.5e-6 [rel 1.2e-6] |
| the edit LM (the adapter on) | not compared numerically here (nextcuda patched zonos3-cli to take the adapter; its LoRA kernels are ported one to one and the base LM above is checked); the edit itself heard word for word, below |

The first codebooks (the semantic one, 97 %) match as closely as nextcuda's; the later residual codebooks drift a few
points more. The encoder runs the reference's BF16 recipe, where a tiny float difference flips a rounding and then a
residual code; the shared diffusion library's convolutions, GELU and layer norm (built with icpx's default floating
point model, not this engine's) are the likely source. A clone from the codes is heard word for word.

## Heard (Whisper large-v3-turbo on the box, through `nextsycl audio serve`)

Text: "The quick brown fox jumps over the lazy dog. Speech synthesis on an Intel Arc card is working well today." unless
named; seed 7; the level of every take measured over its active 50 ms frames (target -19 dBFS).

| take | heard | level |
|---|---|---|
| presets savannah, rupert (their codes), wyatt (speaker-mode embedding) | word for word, all three | -19.2 / -18.9 / -19.0 dBFS |
| clone of Rupert's recording by its embedding | word for word | -19.0 |
| raw-reference clone of Cillian's recording (speaker-mode reference), then a second text from the cached codes | word for word, both | -19.0 / -18.8 |
| a saved voice (`voice: cp-imogen`, the library's recording) | word for word but "Intel R-card" | -19.1 |
| `One.[pause 2] Two.[pause 2] Three, and four.` (8 seeds) | "One, two, three and four."; the two longest pauses 0.55 s on average, 0.30 s without the markers (nextcuda 0.53 / 0.30; Zyphra's runtime 0.52 / 0.28) | -19.0 |
| `I would like a [ipa təmɑːtoʊ], please.` (seeds 1-4) | "tomato" once; "liken Matto", "a no, toe", "to motto" (the ids are Zyphra's exactly) | - |
| `[steer delivery.whisper=4]This part is whispered,[/steer] and this part is not.` | the words alone, no tag read | -18.9 |
| short texts "Yes" (seeds 1-3) and "Hello" | short-text auto: "Yes" 3 of 3, "Hello"; off: "Yes" 2 of 3 ("Yes, and four." once) | -18.5 .. -19.0 |
| controls `delivery.tempo=2,emotions.happy=4` | word for word, 5.5 s against 7.0 s unsteered | -19.0 |
| controls `identity.gender=3` (embedding mode), conditions speaking-rate 18 / loudness -20 / fmax 16000 | word for word ("ArcCard") / word for word, 5.5 s | -18.9 / -19.1 |
| decoder full | word for word | -19.1 |
| cfg 1.5, laci on | word for word | -19.0 / -19.2 |
| level off | word for word | -30.1 (the decoder's own level, 3 dB down) |
| normalize on / off: "On 12/03/2025 it cost $45.50 at 3:30 pm, about 7 km away." | Whisper writes numbers as digits either way; off read "$45.50%" | -19.0 |
| German, liesel | word for word | -19.0 |
| a 1,013-character story, wyatt (four stretches 0.25 s apart) | 62.7 s of sound | -19.2 |
| streamed against whole (rupert, seed 11) | the same words; same samples with level off (above); with level on the causal leveler | see below |
| speech edit: Cillian's 9.7 s, Whisper's transcript, "on weaker hardware." -> "on weaker laptops." | the target word for word; one region 8.93-9.60 s -> 8.93-9.47 s | -21.0 |

Markup through the server: `(laugh)` and `[laugh]` are read ("laugh") and `<laugh/>` is said as "less than laugh slash
greater than" - the server's speech text pass makes brackets into pauses and symbols into words for every speech model;
only the card's annotations' spans (pause, start, ipa, steer) pass as they are.

The streamed take's leveling is causal (a gain from the last 3 s). nextcuda's leveler set its first gain from the first
active frame and then moved at 0.5 s: a quiet onset left the next second up to 17 dB too loud (clipped by the limiter).
Here a lower gain is taken at once and only a higher one is smoothed (`stream.rs`, test
`a_quiet_onset_does_not_make_the_speech_after_it_loud`): the streamed take is now within +0.5 .. -4 dB of the whole
take's level, second by second.

## Speed and VRAM

Through `nextsycl audio serve` (a request's wall time at the client; real-time factor = time / seconds of sound),
2026-11-16, the build copy, one card each:

| request | B70 (gpu 0) | B65 (gpu 1) |
|---|---|---|
| a 7 s sentence, preset | 0.5-0.8 s (RTF 0.07-0.11) | 0.8-0.9 s (RTF 0.09-0.13) |
| a 54.1 s take | 4.63 s (RTF 0.086, 11.7x real time) | 5.26 s (RTF 0.097, 10.3x) |
| a 62.7 s story in four stretches | 4.43 s (RTF 0.071, 14x) | 5.63 s (RTF 0.090, 11x) |
| streamed 7.7 s | first sound after 0.14-0.24 s, all by 0.66-0.83 s | first after 0.17-0.20 s, all by 0.92-1.0 s |
| raw-reference clone, a new recording | 7.2-8.7 s (the encode 4.6-4.8 s + loading the encoders 1.1-4.2 s); the next request from the cached codes 0.32 s | 8.6-10.1 s |
| speech edit, a 9.7 s recording, one region | 7.7 s (the encoders, FuseAlign and the edit LM loaded for it) | 9.3 s |
| FuseAlign alone (`align`, 22 words) | 0.74 s (+1.1 s loading) | - |
| decoder full, its first request | 2.8 s (loading 4 GiB) | - |

nextcuda on an RTX 5070: ~240 rows a second, 53 s of speech in 3.2 s (RTF 0.06). FuseAlign's full attention first ran
on the port of nextcuda's group-a-query kernel (atomics into local memory) and took 5.2 s; on the tiled online-softmax
kernel the backbone's attention uses (heads of 32 / 64 / 128) 0.74 s, its word times unchanged (within 1 ms of
fusealign-cli).

VRAM (the card's used memory polled every 0.1 s, the zonos3 server's own 1,853 MiB once loaded): 1.81 GiB loaded
(backbone 808 MiB, keep2 decoder 710, edit adapter 245, the speaker encoder), 2.2 GiB at the peak of a 54 s take, 8.1 GiB
at the peak of a raw-reference encode or an edit (the reference encoder and SSL teacher, +6.3 GiB, freed after), +4.0 GiB
with the full decoder (the audio server then ends after the answer so a fresh one serves the next request - its rule for a
grown card).

## API (the same shapes as nextcuda's: the platform routes any "zonos3" id to a box listing it)

- `audio.speak` / `/v1/audio/speech`: `input`, `voice` (a preset: savannah - the default -, wyatt, rupert, imogen,
  cillian, jett, morgana, sakura, thaddeus, willow, gaspard, liesel, wolfgang, ananya, arjun, ginevra, lorenzo, minjun,
  seoyeon, haoran, meilin, leonor, thiago, rodrigo, ximena, chaitanya, sravya, none; or a saved voice), `ref_audio`
  (a recording, its first 60 s, no transcript), `cfg` (speaker guidance, 1 = off), `seconds`, `seed`, `stream: true`
  (sound as it is made), `options`: `speaker-mode` reference|embedding, `controls` "ID=DOSE,...", `speaking-rate`,
  `loudness`, `fmax`, `decoder` keep2|full, `normalize` on|off, `level` on|off, `laci` on|off, `short-text`
  auto|off, `temperature` 0.75, `top-k` 80, `top-p` 0.69, `repetition-penalty` 2.4, `silence` 3.
- `audio.edit` (RPC, a job like audio.speak; the audio server's `POST /v1/audio/edit`): `{audio (base64, data: URL or
  upload:<name>; <= 30 s), original_text, target_text, model?, seed?, language?, options?, response_format (url |
  wav)}` -> `{created, data: [{url, ...}], regions: [{original: [s, e], edited: [s, e], old_words}], nextsycl: {took,
  seconds, seed}}`. Capability `speech_edit` (with method `audio.edit`) for a card whose `speech` lists `edit`.
- Markup in the text (kept as written through the server's speech text pass: the engine's `AudioEngine::markup`, the same patterns as the card's `annotations[].keep` - test `the_kept_spans_are_the_cards`):
  `[pause]`, `[pause 0.25|0.5|1|2]`, `[start]`, `[ipa ...]`, `[steer ID=DOSE,...]words[/steer]`.

## Open

- The edit LM with its adapter is not compared with Zyphra's runtime numerically (nextcuda's check needed a patched
  zonos3-cli); `nextsycl audio check DIR --stages forced` with a file `edit` in DIR runs ours.
- The reference encoder's later codebooks sit 3-6 points further from the reference than nextcuda's (above); its
  convolutions run on the shared diffusion library's plain-loop `nsd_conv1d` (the encode is ~4.7 s of a 13 s
  recording's 7-9 s). A GEMM-based convolution in a silo (`kernels/audio/zonos3/silo`) would speed both; not started.
- No silo yet: the engine runs on its own kernels and the shared ones; tuning (the Q4_0 expert products, a graph of the
  decode step) is open.
- Not ported (as on nextcuda): CJK word segmentation for edits of unspaced languages, caller word timings, edit regions'
  own controls, clips over 30 s; accent adapters (none ship).
