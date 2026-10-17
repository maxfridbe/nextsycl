"""VibeVoice-1.5B in its transformers implementation (transformers' models/vibevoice, the Microsoft team's port of
github.com/vibevoice-community/VibeVoice), on the CPU in float32 - the reference for nextsycl's audio engine
(audio/vibevoice): a script's prompt, the stages' tensors and a run's frames, dumped as .npy for `nextsycl audio check`.

    python ref.py HF_CHECKPOINT_DIR --script "Speaker 1: ...\nSpeaker 2: ..." [--voice a.wav --voice b.wav]
                  [--cfg 1.3] [--steps 10] [--frames 60] [--seed 42] [--generate] [--whisper] --out DIR

HF_CHECKPOINT_DIR is vibevoice/VibeVoice-1.5B-hf (the same weights as microsoft/VibeVoice-1.5B, renamed). The prompt
is the processor's own chat template (one voice a speaker, in the order the speakers first speak; the text "Speaker N:"
numbers made 0-based as upstream's processor does). The frame loop is upstream's
generate (vibevoice-community/VibeVoice 07cb79f, modular/modeling_vibevoice_inference.py; transformers'
VibeVoiceGenerationMixin._sample is the same loop) written out: the language model's next token by argmax over the
four tokens it may say (audio start / end, audio, end of text) on its raw logits; for an audio token the negative
branch (audio start, then the same inputs as the positive one) is stepped, the diffusion head denoises a latent from
noise (DPM-Solver++ multistep, v-prediction, cosine betas; cfg between the two branches' last hidden states), the
acoustic decoder turns it into 3,200 samples (streaming: its convolutions' padding cache), the semantic encoder reads
them back (streaming too) and the next input is the acoustic connector's latent plus the semantic connector's
features. As upstream: an audio end clears both streaming caches, an audio start restarts the negative branch (its
audio start at position 0, the next input at 1), and a run takes at most twice the prompt's length in steps.
Torch's generator is seeded and drawn in the order generate() draws it (the voices' VAE noise, then a [2, 64] noise a
frame), so --generate checks the loop against transformers' model.generate - which also guides the token choice
itself (the generation config's guidance_scale adds an UnbatchedClassifierFreeGuidanceLogitsProcessor; upstream does
not), so it may end a frame or two apart while the frames they share are identical.

Dumps (DIR): request.json; ids.npy (the prompt's token ids, the audio token where a voice frame goes);
voice{i}.wav (the recording as read, mono 24 kHz) and voice{i}_mean.npy [F, 64] (the acoustic encoder's latents of
the normalized, padded recording); voice_feats.npy [F, 64] (the latents as the prompt uses them: the VAE noise added,
then the bias and scale) and voice_embeds.npy [F, 1536]; prefill.npy [L, 1536] (the language model's input rows);
sched.npy [steps, 2] (timesteps, sigmas); tokens.npy [S] (each step's token); logits.npy [S, 4] (the four tokens'
logits: audio start, audio end, audio, end of text); cond.npy [T, 2, 1536] (each audio frame's positive and negative
last hidden state); noise.npy [T, 64]; eps0.npy [T, 2, 64] (the head's first-step output on both branches);
latents.npy [T, 64]; semantic.npy [T, 128]; embeds.npy [T, 1536] (the next input rows); speech.npy / speech.wav
(24 kHz); with --whisper whisper.txt (openai/whisper-small's transcript of speech.wav).
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from scipy.io import wavfile


def read_wav(path):
    """a recording's samples (float32, channels mixed) and rate"""
    sr, w = wavfile.read(path)
    if w.dtype.kind == "i":
        w = w.astype(np.float32) / float(np.iinfo(w.dtype).max + 1)
    elif w.dtype.kind == "u":
        w = (w.astype(np.float32) - 128.0) / 128.0
    w = w.astype(np.float32)
    return (w.mean(1) if w.ndim > 1 else w), sr


def parse_script(script):
    """upstream's processor (_parse_script): "Speaker N: text" lines, N made 0-based when none is 0; a line without
    a speaker continues the one before; no speaker at all: the whole text is speaker 0's"""
    lines = []
    for raw in script.replace("\\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            lines.append([int(m.group(1)), m.group(2).strip()])
        elif lines:
            lines[-1][1] = (lines[-1][1] + " " + line).strip()
        else:
            lines.append([0, line])
    low = min(s for s, _ in lines)
    if low > 0:
        lines = [[s - 1, t] for s, t in lines]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--script", required=True)
    ap.add_argument("--voice", action="append", default=[], help="a recording a speaker, speaker 0 first")
    ap.add_argument("--cfg", type=float, default=1.3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--frames", type=int, default=60, help="at most this many language model steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generate", action="store_true", help="also run model.generate with the same seed and compare")
    ap.add_argument("--whisper", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoProcessor, VibeVoiceForConditionalGeneration
    from transformers.cache_utils import DynamicCache

    proc = AutoProcessor.from_pretrained(a.model)
    model = VibeVoiceForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32)
    model.float().eval()
    cfg = model.config
    m = model.model

    def save(n, v):
        np.save(os.path.join(a.out, n), np.ascontiguousarray(v))

    lines = parse_script(a.script)
    voices = []
    for i, p in enumerate(a.voice):
        wav, sr = read_wav(p)
        if sr != 24000:
            import torchaudio.functional as AF
            wav = AF.resample(torch.from_numpy(wav), sr, 24000).numpy()
        wavfile.write(os.path.join(a.out, f"voice{i}.wav"), 24000, wav.astype(np.float32))
        voices.append(wav)
    # the conversation: a message a line, the speaker's recording on its first line
    conv, given = [], set()
    for s, t in lines:
        content = [{"type": "text", "text": t}]
        if s < len(voices) and s not in given:
            content.append({"type": "audio", "audio": voices[s]})
            given.add(s)
        conv.append({"role": str(s), "content": content})
    with open(os.path.join(a.out, "request.json"), "w") as f:
        json.dump({"script": a.script.replace("\\n", "\n"), "lines": lines, "voices": len(voices), "cfg": a.cfg, "steps": a.steps,
                   "frames": a.frames, "seed": a.seed}, f, ensure_ascii=False)
    inputs = proc.apply_chat_template(conv, return_dict=True, tokenize=True, add_generation_prompt=True, sampling_rate=24000)
    ids = inputs["input_ids"]
    save("ids.npy", ids[0].long().numpy())
    print("prompt:", repr(proc.tokenizer.decode(ids[0])))

    torch.manual_seed(a.seed)
    embed = model.get_input_embeddings()
    valid = [cfg.audio_bos_token_id, cfg.audio_eos_token_id, cfg.audio_token_id, cfg.eos_token_id]
    sched = model._build_default_noise_scheduler(model.generation_config)
    sched.set_timesteps(num_inference_steps=a.steps)
    save("sched.npy", np.stack([sched.timesteps.float().numpy(), np.asarray(sched.sigmas[:-1], dtype=np.float32)], 1))

    with torch.inference_mode():
        emb = embed(ids)
        if voices:
            iv, pm = inputs["input_values"], inputs["padding_mask"]
            # each recording's encoder latents alone (no noise): the check's first comparison
            for i in range(len(voices)):
                n = int(pm[i].sum())
                frames = -(-n // 3200)
                lat = m.audio_tower.encoder(iv[i:i + 1, :, :frames * 3200]).latents[0]
                save(f"voice{i}_mean.npy", lat.numpy())
            feats = m.get_audio_features(iv, pm)
            save("voice_feats.npy", feats.last_hidden_state.numpy())
            save("voice_embeds.npy", feats.pooler_output.numpy())
            mask = (ids == cfg.audio_token_id)[..., None].expand_as(emb)
            emb = emb.masked_scatter(mask, feats.pooler_output)
        save("prefill.npy", emb[0].numpy())

        lm = m.language_model
        cache = DynamicCache(config=cfg.text_config)
        h = lm(inputs_embeds=emb, past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
        neg_cache = DynamicCache(config=cfg.text_config)
        inputs_embeds = None
        acoustic_cache = semantic_cache = None
        tokens, logits4, conds, noises, eps0s, latents, sems, embeds, chunks = [], [], [], [], [], [], [], [], []
        bos = embed(torch.tensor([[cfg.audio_bos_token_id]]))
        for step in range(min(a.frames, 2 * ids.shape[1])):
            logits = model.lm_head(h)[0]
            l4 = logits[valid]
            tok = valid[int(torch.argmax(l4))]
            tokens.append(tok)
            logits4.append(l4.numpy())
            nxt = embed(torch.tensor([[tok]]))
            if tok == cfg.audio_eos_token_id:
                acoustic_cache = semantic_cache = None
            if tok == cfg.audio_bos_token_id:
                # upstream keeps only the first position's keys (its audio start) and goes on from position 1
                neg_cache = DynamicCache(config=cfg.text_config)
                lm(inputs_embeds=bos, past_key_values=neg_cache, use_cache=True)
            if tok == cfg.audio_token_id:
                neg_in = inputs_embeds if inputs_embeds is not None else bos
                hn = lm(inputs_embeds=neg_in, past_key_values=neg_cache, use_cache=True).last_hidden_state[:, -1]
                cond = torch.cat([h, hn], 0)
                conds.append(cond.numpy())
                x = torch.randn(2, cfg.audio_config.hidden_size)
                noises.append(x[0].numpy().copy())
                sched.set_timesteps(num_inference_steps=a.steps)
                for k, t in enumerate(sched.timesteps):
                    comb = torch.cat([x[:1], x[:1]], 0)
                    eps = m.diffusion_head(comb, t.repeat(2).to(comb), condition=cond)
                    if k == 0:
                        eps0s.append(eps.numpy())
                    g = eps[1:] + a.cfg * (eps[:1] - eps[1:])
                    x = sched.step(torch.cat([g, g], 0), t, x).prev_sample
                lat = x[:1].unsqueeze(1)
                latents.append(lat[0, 0].numpy())
                out = model._decode_audio_latent(lat, torch.tensor([True]), 1, acoustic_cache)
                acoustic_cache = out.padding_cache
                chunks.append(out.audio[0, 0].numpy())
                so = m.semantic_tokenizer_encoder(out.audio, padding_cache=semantic_cache, use_cache=True)
                semantic_cache = so.padding_cache
                sems.append(so.latents[0, 0].numpy())
                nxt = m.multi_modal_projector(lat) + m.semantic_connector(so.latents)
                embeds.append(nxt[0, 0].numpy())
            inputs_embeds = nxt
            if tok == cfg.eos_token_id:
                break
            h = lm(inputs_embeds=nxt, past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
            print(f"step {step}: token {tok}{'' if tok != cfg.audio_token_id else f', frame {len(latents)}'}", flush=True)
    save("tokens.npy", np.array(tokens, dtype=np.int64))
    save("logits.npy", np.stack(logits4))
    if latents:
        save("cond.npy", np.stack(conds))
        save("noise.npy", np.stack(noises))
        save("eps0.npy", np.stack(eps0s))
        save("latents.npy", np.stack(latents))
        save("semantic.npy", np.stack(sems))
        save("embeds.npy", np.stack(embeds))
        speech = np.concatenate(chunks).astype(np.float32)
        save("speech.npy", speech)
        wavfile.write(os.path.join(a.out, "speech.wav"), 24000, speech)
        print(f"{len(latents)} frames, {len(speech) / 24000:.2f} s; prefill {emb.shape[1]} rows")

    if a.generate:
        torch.manual_seed(a.seed)
        model.generation_config.max_new_tokens = a.frames
        audio = model.generate(**inputs, guidance_scale=a.cfg, num_diffusion_steps=a.steps)[0]
        g = audio.reshape(-1).float().numpy()
        n = min(len(g), len(speech))
        print(f"generate(): {len(g)} samples, the loop here {len(speech)}; max |difference| over {n}: {np.abs(g[:n] - speech[:n]).max():.3e}")

    if a.whisper and latents:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
        text = asr({"raw": speech, "sampling_rate": 24000})["text"]
        with open(os.path.join(a.out, "whisper.txt"), "w") as f:
            f.write(text.strip() + "\n")
        print("whisper:", text)


if __name__ == "__main__":
    main()
