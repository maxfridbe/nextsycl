# Goal: voice cloning for Voxtral TTS without its encoder, on the GPU

Status: **stage A done, shipped as experimental** (2026-10-17): `nextsycl audio voice enroll`, the gate's table and the
prior's sweep in `docs/ports/voxtraltts.md`. Stage B not needed (A takes 1-2 minutes a voice on a B65).
Source of the idea: <https://bbioon.com/blog/fixing-voxtral-voice-cloning-without-an-encoder>.

## The problem

Voxtral TTS is prompted with a voice as frames of codec codes: per 80 ms frame, 1 semantic VQ index (codebook of
8,192 x 256) and 36 acoustic FSQ codes (21 levels), each frame's code embeddings summed into one row of 3,072. The 20
preset voices ship as those rows (`voice_embedding/<name>.pt`, e.g. [218, 3072]). A new voice needs the codes of its
recording, made by the codec's encoder - and Mistral's open checkpoint (`mistralai/Voxtral-4B-TTS-2603`, checked
2026-10-17: 386 tensors, `audio_tokenizer.*` = decoder_blocks 112, output_proj 2, quantizer 2) has no encoder.
vLLM-Omni refuses `ref_audio` for it; cloning runs only in Mistral's hosted studio.

## The idea (from the post)

Treat the missing encoder as an optimization problem: fit the codes so that the released decoder reproduces the
reference clip. Continuous parameters per frame, quantized in the forward pass and passed straight through in the
backward pass (straight-through estimator), loss L1 on the waveform plus a multi-resolution STFT loss:

```python
normalized = torch.tanh(self.values)
scaled = ((normalized + 1) / 2) * (self.levels - 1)
quantized = scaled.round()
return scaled + (quantized - scaled).detach()
```

The post overfits one 8 s clip in about an hour on an M-series Mac, reports "indistinguishable" by ear, and gives
no hyperparameters and no measurements.

## The goal

A saved voice from the voice library can be **enrolled** for Voxtral once, **on the GPU, in minutes rather than an
hour**, and from then on Voxtral speaks in it as it does a preset. Shipped only if it measurably sounds like the
speaker (the gate below).

## GPU plan

### Stage A: per-voice inversion on the card (first)

- **Where it runs**: PyTorch with Intel's XPU backend (`torch.xpu`, upstream since 2.5) on a B70/B65, in a venv
  inside the oneAPI container image, with the decoder built from `consolidated.safetensors` as the reference code
  does. The whole optimization stays on the card: decoder ~300 M parameters, a 10-20 s clip is 125-250 frames, and
  forward + backward fit in a few GiB. The nextsycl engines are inference-only (no backward kernels), so the fit
  is a tool beside them, not engine code; its result is the only thing the engine reads.
- **Parameters**: per frame a continuous 256-d semantic vector (snapped to the nearest codeword through a
  straight-through estimator, or a softmax over the 8,192 codewords annealed to one-hot) and 36 continuous acoustic
  values (FSQ straight-through, as above).
- **Loss**: L1 waveform + multi-resolution STFT (+ mel L1), the post's recipe. Then the in-distribution term below.
- **Speed tricks**: bf16 autocast, the decoder's chunks in one batch, Adam with a short warm-up, early stop when the
  STFT loss plateaus. Target: under 5 minutes a voice on a B70.

### The risk, and the term that answers it

Many code sequences decode to the same audio, and the backbone was trained on the encoder's codes, the semantic
code above all. Codes that reproduce the clip may still be unlike anything the backbone saw, and give a weak or
garbled voice. So add a prior from the model itself: transcribe the clip (Whisper), feed [the transcript, the
fitted frames] through the backbone teacher-forced, and add the negative log-likelihood of the semantic codes (and
the flow head's fit of the acoustic ones) to the loss, with a weight swept against the gate's numbers. The codes
are then ones that reproduce the clip **and** that the model finds likely for that text. The 3B backbone's
backward pass fits on a 32 GB card at these lengths in bf16 (weights frozen; gradients only reach the codes).

### Stage B: a trained encoder (if stage A works but is too slow or uneven)

Amortize the inversion: train a small encoder once on the GPU, then enrollment is one forward pass (under a
second). Training pairs come from the model itself: the backbone with the 20 presets and many texts generates
codes in its own distribution, the decoder turns them into audio, and the encoder learns audio -> codes (semantic
cross-entropy, FSQ regression), fine-tuned on stage A's fits of real voices. Hours of GPU once; nothing per voice.

## How it plugs in

- **Storage**: the fitted frames (codes, and the summed [N, 3072] rows) are kept with the saved voice as a
  per-model file; a voice without it is not offered to Voxtral.
- **Model card**: a capability such as `enrolled`, so the front door's voice routing (`speech_model`,
  `audio.voices.list`) offers enrolled voices to Voxtral and the others to the models that clone directly.
- **Front door**: "Enroll for Voxtral" on a saved voice, a background job that needs a card like any other (it
  asks for its memory through the same placement as models, keeps 1.5 GiB free, and never runs beside a load on
  that card). CLI: `nextsycl audio enroll <voice> --model voxtral-4b-tts`.

## Gate: what decides shipping

For at least 3 reference speakers:

| measure | how |
|---|---|
| likeness | cosine of speaker embeddings (the x-vector encoder in nextsycl's Qwen3-TTS base engine, or ECAPA) between the reference and generated speech |
| words | Whisper WER on the generated text |
| fit | the decoder's reconstruction of the clip from the fitted codes (STFT / mel distance) |

Compared against: the nearest of the 20 presets (best likeness), Breeze TTS 2's clone and VoxCPM2's clone of the
same clip. Shipped (marked experimental) only if the enrolled voice clearly beats the nearest preset on likeness
with a WER near the presets'. Otherwise the tool and the table stay in `docs/ports/voxtraltts.md` and the served
model stays presets-only.

## Not in scope

Changing Voxtral's weights, or anything that would make it a different model; cloning a voice without the
speaker's consent (the voice library's existing rules apply).
