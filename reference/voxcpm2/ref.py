"""VoxCPM2 in its own package (voxcpm), on the CPU in float32 - the reference for nextsycl's audio engine
(audio/voxcpm2): a mode's speech and the stages' tensors, dumped as .npy for `nextsycl audio check`.

    python ref.py MODEL_DIR tts      --text "..." --out DIR             (zero-shot; "(a description)text" designs a voice)
    python ref.py MODEL_DIR clone    --text "..." --ref ref.wav --out DIR            (the reference's timbre, isolated)
    python ref.py MODEL_DIR continue --text "..." --prompt p.wav --prompt-text "..." --out DIR   (continuation)
    python ref.py MODEL_DIR tokens   --text "..."                        (the token ids only)

Dumps (DIR): request.json; tokens.npy, text_mask.npy, audio_mask.npy, audio_feat.npy (the prompt as the model takes
it: [L], [L], [L], [L, 4, 64]); lm0.npy / res0.npy (the base and residual language models' last hidden state after the
prefill: the audio-start token's); noise.npy [N, 64, 4] (every draw of the flow sampler, in
order); feats.npy [T, 4, 64] (the patches made); speech.npy / speech.wav (48 kHz); for clone / continue the 16 kHz
audio the VAE encoded (ref16k.npy / prompt16k.npy).
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("mode", choices=["tts", "clone", "continue", "tokens"])
    ap.add_argument("--text", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--prompt-text", default="")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=2000)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    from voxcpm.modules.locdit import unified_cfm

    m = VoxCPM2Model.from_local(a.model, optimize=False, device="cpu")
    m = m.to(torch.float32)
    m.config.dtype = "float32"
    # the caches were made in the checkpoint's dtype: again in float32
    for lm in (m.base_lm, m.residual_lm):
        lm.setup_cache(1, m.config.max_length, "cpu", torch.float32)
    if a.mode == "tokens":
        print(json.dumps(m.text_tokenizer(a.text)))
        return
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "request.json"), "w") as f:
        json.dump({"mode": a.mode, "text": a.text, "prompt_text": a.prompt_text, "steps": a.steps, "cfg": a.cfg, "seed": a.seed}, f, ensure_ascii=False)
    seen = {"noise": []}

    # the flow sampler's draws
    real_randn = torch.randn

    def randn(*args, **kw):
        z = real_randn(*args, **kw)
        if len(z.shape) == 3 and z.shape[1] == m.feat_dim:
            seen["noise"].append(z[0].float().numpy().copy())
        return z
    unified_cfm.torch.randn = randn

    # the prompt and the first hidden states, caught on the way
    real_inf = m._inference

    def inference(text, text_mask, feat, feat_mask, **kw):
        seen["tokens"] = text[0].numpy().astype(np.int32)
        seen["text_mask"] = text_mask[0].numpy().astype(np.int32)
        seen["audio_mask"] = feat_mask[0].numpy().astype(np.int32)
        seen["audio_feat"] = feat[0].float().numpy()
        gen = real_inf(text, text_mask, feat, feat_mask, **kw)
        return gen
    m._inference = inference

    real_base = m.base_lm.forward
    real_res = m.residual_lm.forward

    def base(inputs_embeds, is_causal=True):
        out, kv = real_base(inputs_embeds=inputs_embeds, is_causal=is_causal)
        if "lm0" not in seen:
            seen["lm0"] = out[0, -1].float().numpy()
        return out, kv

    def res(inputs_embeds, is_causal=True):
        out, kv = real_res(inputs_embeds=inputs_embeds, is_causal=is_causal)
        if "res0" not in seen:
            seen["res0"] = out[0, -1].float().numpy()
        return out, kv
    m.base_lm.forward = base
    m.residual_lm.forward = res

    real_enc = m._encode_wav

    def encode(path, **kw):
        import librosa
        audio, _ = librosa.load(path, sr=m._encode_sample_rate, mono=True)
        seen["ref16k" if kw.get("padding_mode") == "right" else "prompt16k"] = audio.astype(np.float32)
        return real_enc(path, **kw)
    m._encode_wav = encode

    real_decode = m.audio_vae.decode

    def decode(z, *x, **k):
        seen["latents"] = z[0].float().numpy()
        return real_decode(z, *x, **k)
    m.audio_vae.decode = decode

    kw = dict(target_text=a.text, inference_timesteps=a.steps, cfg_value=a.cfg, max_len=a.max_len)
    if a.mode == "clone":
        kw["reference_wav_path"] = a.ref
    elif a.mode == "continue":
        kw["prompt_wav_path"] = a.prompt
        kw["prompt_text"] = a.prompt_text
    wav = m.generate(**kw)
    wav = wav.squeeze().float().numpy()
    noise = np.stack(seen.pop("noise"))
    np.save(os.path.join(a.out, "noise.npy"), noise)
    lat = seen.pop("latents")  # [64, T*4]
    feats = lat.reshape(64, -1, 4).transpose(1, 2, 0)
    np.save(os.path.join(a.out, "feats.npy"), np.ascontiguousarray(feats))
    for k, v in seen.items():
        np.save(os.path.join(a.out, f"{k}.npy"), np.ascontiguousarray(v))
    np.save(os.path.join(a.out, "speech.npy"), wav.astype(np.float32))
    sf.write(os.path.join(a.out, "speech.wav"), wav, m.sample_rate)
    print(f"{len(wav) / m.sample_rate:.2f} s of speech, {feats.shape[0]} patches, {noise.shape[0]} draws; prompt {seen['tokens'].shape}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
