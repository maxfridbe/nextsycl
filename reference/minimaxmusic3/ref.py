"""reference/minimaxmusic3/ref.py: MiniMax Music 3 on the CPU (diffusers' own MiniMaxMusic3 classes and pipeline
helpers, float32 weights and activations), dumping every stage the engine is checked against.

    python ref.py ar     --model <MiniMax-Music3 dir> --out DIR [--frames 24] [--seed 7] [--prompt TEXT --lyrics TEXT]
    python ref.py dit    --model DIR --out DIR [--steps 30] [--seed 7]     (after ar: its frame_hiddens.npy)
    python ref.py voc    --model DIR --out DIR                             (after dit: its latents.npy)
    python ref.py chunks --model DIR --out DIR [--frames 330] [--steps 3]  (the windows' overlap and the stitching)

The stages run one at a time (each frees its weights) and talk through DIR:
  ar:  tokens.json (the conditional prompt's ids), prefill.npy [2, 4096] (the prompt's last hidden state, conditional
       and unconditional), c0_logits.npy [16385] (the first frame's guided logits over the semantic codes and the end
       token, before top-k), codes.npy [F+1, 8] (every frame's sampled codes - frame 0 only advances past
       <|audio_start|>), lm_hidden.npy [F+1, 2, 4096] (the language model's hidden state each frame starts from),
       depth.npy [F+1, 7, 4096] (the depth decoder's conditional hidden state for c1..c7), frame_hiddens.npy
       [F, 8 * 4096] (what conditions the flow-matching stage)
  dit: condition.npy [L, 2048], noise.npy [128, L], sigmas.npy, v0_cond.npy / v0_uncond.npy (step 0's velocities),
       latents_<i>.npy [128, L] after each step, latents.npy (the last)
  voc: audio.npy [2, L * 512] and audio.wav (44.1 kHz)
  chunks: frame_hiddens.npy (seeded, synthetic), chunk_starts.json, noise_<k>.npy, latents_<k>.npy (each window,
       uncropped, after --steps), audio.npy (stitched)

The engine runs the same frames teacher-forced (the codes from codes.npy, not its own draws: the reference's sampler
is torch's), so a difference is the engine's arithmetic, not the dice.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)

CAPTION = ("Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
           "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
           "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus.")
LYRICS = "[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe"


def save(out, name, a):
    np.save(os.path.join(out, name), np.ascontiguousarray(a.detach().float().cpu().numpy() if torch.is_tensor(a) else a, dtype=np.float32))


def sub(a, name):
    return os.path.join(a.model, name)


# ------------------------------------------------------------------------------------------------ autoregressive
def run_ar(a):
    from diffusers import MiniMaxMusic3RVQDepthDecoder
    from diffusers.modular_pipelines.minimax_music3 import encoders as E
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(sub(a, "tokenizer"))
    text = (f"{E._IM_START}{E._CAPTION_START}{E._clean_caption(a.prompt)}{E._CAPTION_END}"
            f"{E._LYRICS_START}{E._normalize_lyrics(a.lyrics)}{E._LYRICS_END}{E._IM_END}{E._AUDIO_START}")
    ids = tok(text, return_tensors="pt")["input_ids"]
    json.dump({"text": text, "ids": ids[0].tolist()}, open(os.path.join(a.out, "tokens.json"), "w"))
    unc = ids.clone()
    unc[:, 1:-2] = E._AUDIO_CFG_TOKEN_ID
    text_ids = torch.cat((ids, unc), 0)
    print(f"prompt: {ids.shape[1]} tokens", file=sys.stderr)

    lm = Qwen3ForCausalLM.from_pretrained(sub(a, "language_model"), torch_dtype=torch.float32).eval()
    dd = MiniMaxMusic3RVQDepthDecoder.from_pretrained(sub(a, "rvq_depth_decoder"), torch_dtype=torch.float32).eval()
    gen = torch.Generator("cpu").manual_seed(a.seed)

    out = lm.model(inputs_embeds=lm.model.embed_tokens(text_ids), use_cache=True)
    past = out.past_key_values
    last = out.last_hidden_state[:, -1]
    save(a.out, "prefill.npy", last)
    mask = torch.ones(lm.config.vocab_size, dtype=torch.bool)
    mask[E._AUDIO_CODE_OFFSET:E._AUDIO_CODE_OFFSET + E._SEMANTIC_VOCAB_SIZE] = False
    mask[E._AUDIO_END_TOKEN_ID] = False

    codes, lm_h, depth, frames = [], [], [], []
    for f in range(a.frames + 1):
        logits = lm.lm_head(last).float().masked_fill(mask, -float("inf"))
        c, u = logits[0:1], logits[1:2]
        guided = u + (c - u) * E._AR_CFG_SCALE
        if f == 0:
            # the end token, then the semantic codes: the engine's logit order
            sel = torch.cat((guided[0, E._AUDIO_END_TOKEN_ID:E._AUDIO_END_TOKEN_ID + 1],
                             guided[0, E._AUDIO_CODE_OFFSET:E._AUDIO_CODE_OFFSET + E._SEMANTIC_VOCAB_SIZE]))
            save(a.out, "c0_logits.npy", sel)
        thr = torch.topk(c, E._AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(c < thr, -float("inf")).masked_fill(mask.unsqueeze(0), -float("inf"))
        s = E._sample_top_k(guided, gen)
        if int(s.item()) == E._AUDIO_END_TOKEN_ID:
            print(f"the end token at frame {f}", file=sys.stderr)
            break
        sem = s - E._AUDIO_CODE_OFFSET
        fc, dh = E._generate_depth_codes(lm, dd, last, sem.repeat(2), gen)
        codes.append(fc[0])
        lm_h.append(last.clone())
        depth.append(dh[0].reshape(7, -1))
        if f > 0:
            frames.append(torch.cat((last[:1], dh), -1)[0])
        fb = E._embed_audio_frame(lm, dd, fc)
        out = lm.model(inputs_embeds=fb, past_key_values=past, use_cache=True)
        past = out.past_key_values
        last = out.last_hidden_state[:, -1]
        print(f"frame {f}: {fc[0].tolist()}", file=sys.stderr)
    np.save(os.path.join(a.out, "codes.npy"), torch.stack(codes).numpy().astype(np.int32))
    save(a.out, "lm_hidden.npy", torch.stack(lm_h))
    save(a.out, "depth.npy", torch.stack(depth))
    save(a.out, "frame_hiddens.npy", torch.stack(frames))


# ------------------------------------------------------------------------------------------------ flow matching
def load_fm(a):
    from diffusers import FlowMatchEulerDiscreteScheduler, MiniMaxMusic3ConditionEncoder, MiniMaxMusic3Transformer1DModel
    ce = MiniMaxMusic3ConditionEncoder.from_pretrained(sub(a, "condition_encoder"), torch_dtype=torch.float32).eval()
    tr = MiniMaxMusic3Transformer1DModel.from_pretrained(sub(a, "transformer"), torch_dtype=torch.float32).eval()
    sc = FlowMatchEulerDiscreteScheduler.from_pretrained(sub(a, "scheduler"))
    return ce, tr, sc


def denoise(tr, sc, cond, lat, steps, cfg, prev=None, noise_prompt=None, overlap=0, each=None):
    sc.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps))
    for i, t in enumerate(sc.timesteps):
        if overlap > 0:
            lat[..., :overlap] = (1.0 - (1.0 - 1e-6) * t) * noise_prompt + t * prev[..., :overlap]
        ts = t.expand(1)
        vc = tr(hidden_states=lat, timestep=ts, encoder_hidden_states=cond, return_dict=False)[0]
        vu = tr(hidden_states=lat, timestep=ts, encoder_hidden_states=torch.zeros_like(cond), return_dict=False)[0]
        v = vu + cfg * (vc - vu)
        if each:
            each(i, vc, vu)
        lat = sc.step(v, t, lat, return_dict=False)[0]
        if each:
            each(i, None, lat)
    return lat


def run_dit(a):
    ce, tr, sc = load_fm(a)
    fh = torch.from_numpy(np.load(os.path.join(a.out, "frame_hiddens.npy")))[None]
    cond = ce(fh)
    save(a.out, "condition.npy", cond[0])
    gen = torch.Generator("cpu").manual_seed(a.seed)
    lat = torch.randn((1, 128, cond.shape[1]), generator=gen)
    save(a.out, "noise.npy", lat[0])
    sc.set_timesteps(sigmas=np.linspace(1.0, 1.0 / a.steps, a.steps))
    save(a.out, "sigmas.npy", sc.sigmas)

    def each(i, x, y):
        if x is not None:
            if i == 0:
                save(a.out, "v0_cond.npy", x[0])
                save(a.out, "v0_uncond.npy", y[0])
        else:
            save(a.out, f"latents_{i}.npy", y[0])
            print(f"step {i}", file=sys.stderr)
    lat = denoise(tr, sc, cond, lat, a.steps, 1.7, each=each)
    save(a.out, "latents.npy", lat[0])


def run_voc(a):
    from diffusers import MiniMaxMusic3Vocoder
    voc = MiniMaxMusic3Vocoder.from_pretrained(sub(a, "vocoder"), torch_dtype=torch.float32).eval()
    lat = torch.from_numpy(np.load(os.path.join(a.out, "latents.npy")))[None]
    wav = voc(lat)[0].clamp(-1, 1)
    save(a.out, "audio.npy", wav)
    write_wav(os.path.join(a.out, "audio.wav"), wav.numpy(), 44100)


def write_wav(path, wav, rate):
    import wave
    pcm = (np.clip(wav.T, -1, 1) * 32767).round().astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(wav.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def run_chunks(a):
    from diffusers import MiniMaxMusic3Vocoder
    from diffusers.modular_pipelines.minimax_music3 import before_denoise as B, decoders as D, denoise as N
    ce, tr, sc = load_fm(a)
    gen = torch.Generator("cpu").manual_seed(a.seed)
    fh = torch.randn((1, a.frames, 8 * 4096), generator=gen) * 2.0
    save(a.out, "frame_hiddens.npy", fh[0])
    nf = fh.shape[1]
    starts = [0] if nf <= B._CHUNK_FRAMES else list(range(0, nf - B._CHUNK_HOP, B._CHUNK_HOP))
    json.dump(starts, open(os.path.join(a.out, "chunk_starts.json"), "w"))
    prev_lat, prev_cond, chunks = None, None, []
    for k, s in enumerate(starts):
        cond = ce(fh[:, s:min(s + B._CHUNK_FRAMES, nf)])
        overlap = 0
        if prev_lat is not None:
            overlap = min(prev_lat.shape[-1], cond.shape[1])
            cond[:, :overlap] = prev_cond[:, :overlap]
        lat = torch.randn((1, 128, cond.shape[1]), generator=gen)
        save(a.out, f"noise_{k}.npy", lat[0])
        np_ = lat[..., :overlap].clone() if overlap > 0 else None
        lat = denoise(tr, sc, cond, lat, a.steps, 1.7, prev_lat, np_, overlap)
        if overlap > 0:
            lat[..., :overlap] = prev_lat[..., :overlap]
        o0 = max(0, lat.shape[-1] - 2 * N._OVERLAP_LATENT_LENGTH)
        o1 = max(o0, lat.shape[-1] - N._OVERLAP_LATENT_LENGTH)
        prev_lat, prev_cond = lat[..., o0:o1], cond[:, o0:o1]
        save(a.out, f"latents_{k}.npy", lat[0])
        chunks.append(lat)
        print(f"window {k} at frame {s}: {cond.shape[1]} latents, overlap {overlap}", file=sys.stderr)
    del tr
    voc = MiniMaxMusic3Vocoder.from_pretrained(sub(a, "vocoder"), torch_dtype=torch.float32).eval()
    parts = []
    for k, lat in enumerate(chunks):
        w = voc(lat)
        left = 0 if k == 0 else D._CROP_LEFT_LATENT * 512
        right = 0 if k == len(chunks) - 1 else D._CROP_RIGHT_LATENT * 512
        parts.append(w[..., left:w.shape[-1] - right])
    save(a.out, "audio.npy", torch.cat(parts, -1)[0].clamp(-1, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["ar", "dit", "voc", "chunks"])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prompt", default=CAPTION)
    p.add_argument("--lyrics", default=LYRICS)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    {"ar": run_ar, "dit": run_dit, "voc": run_voc, "chunks": run_chunks}[a.stage](a)


if __name__ == "__main__":
    main()
