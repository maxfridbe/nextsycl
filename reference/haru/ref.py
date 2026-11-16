"""Haru-4B's own PyTorch code (github.com/higurashi-ai/Haru, haru/) in float32 on the CPU, dumped stage by stage for
`nextsycl audio check DUMP --stages codec,prompt,frames,head` (audio/haru/src/check.rs).

    python3 -I ref.py --src HARU_REPO/haru --model MODEL_DIR --voice voice.wav --user user.wav --out DUMP [--frames 12] [--speak-first]

What it writes (float32 / int32 .npy):
    enc_in.npy [T]            the user clip at 24 kHz, whole frames (zeros after it)
    enc_mean.npy [F, 64]      the codec encoder's latent means of it (streaming = whole clip: the codec is causal)
    dec_in.npy [F, 64]        latents for the decoder (the voice's means)
    dec_out.npy [F * 3200]    the codec decoder's audio of them
    card.txt, card_ids.npy    the character card and its ChatML ids
    p_tokens / p_model / p_user / p_mmask / p_umask.npy   the prompt's three streams (latents raw: the codec's scale)
                              and masks; the voice's latents are its means (no noise: deterministic)
    p_emb.npy [n, 2560]       the summed input of each prompt position
    p_h{l}.npy [2560]         the last position's residual after layer l (0..31; 32: after the final norm)
    p_logits.npy [vocab]      its text logits (float32)
    f_tok / f_model / f_user.npy [K] / [K, 64] / [K, 64]  the frames teacher-forced: the text token, the model's
                              latent (raw) and the user's (raw: the codec means of user.wav) of position prompt + k
    f_hidden.npy [K + 1, 2560], f_logits.npy [K + 1, vocab]  the hidden state that predicts frame k (row 0: the prompt's)
    f_noise.npy, f_latent.npy [K + 1, 64]  the head's noise and its latent (raw) from that hidden state (guidance 1.5)
The frames' tokens are the logits' argmax and the model's latents the head's own: a greedy run of the session.
"""
import argparse
import json
import math
import os
import sys
import wave

import numpy as np


def read_wav(path):
    with wave.open(path) as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise SystemExit(f"{path}: {width * 8}-bit samples")
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != 24000:
        from scipy.signal import resample_poly
        g = math.gcd(rate, 24000)
        x = resample_poly(x, 24000 // g, rate // g).astype(np.float32)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the Haru repository's haru/ (the package's parent)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--speak-first", action="store_true", help="the prompt's opening grid (PAD, EOP over room tone and silence); frame 0 a word")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import torch
    torch.set_num_threads(a.threads)
    from haru.models.loaders import load_model
    from haru.prompt import build_prompt, card_templates, render_card, system_prompt_ids
    from haru.audio import prepare_reference

    os.makedirs(a.out, exist_ok=True)
    save = lambda name, x: np.save(os.path.join(a.out, name), np.ascontiguousarray(x))
    model = load_model(a.model, device="cpu", dtype=torch.float32)
    codec = model.codec
    # deterministic: the codec's latents are its means (no noise drawn)
    codec.sample = lambda mean, generator=None: mean

    # ---- codec
    user = read_wav(a.user)
    f = math.ceil(len(user) / 3200)
    enc_in = np.zeros(f * 3200, np.float32)
    enc_in[: len(user)] = user
    save("enc_in.npy", enc_in)
    with torch.no_grad():
        mean = codec.encode(torch.from_numpy(enc_in))[0].float().numpy()
    save("enc_mean.npy", mean)
    voice = prepare_reference(read_wav(a.voice))
    with torch.no_grad():
        vmean = codec.encode(torch.from_numpy(voice))[0].float()
        save("dec_in.npy", vmean.numpy())
        save("dec_out.npy", codec.decode(vmean[None])[0].float().numpy())
    print(f"codec: {f} user frames, {vmean.shape[0]} voice frames", flush=True)

    # ---- the prompt
    t = card_templates()["templates"]["en"]
    cards = json.load(open(os.path.join(a.src, "haru", "assets", "cards", "en.json")))
    card = render_card(cards["chat"], t)
    open(os.path.join(a.out, "card.txt"), "w").write(card)
    save("card_ids.npy", np.array(system_prompt_ids(model.tokenizer, card), np.int32))
    p = build_prompt(model.config, codec, model.tokenizer, card, torch.from_numpy(voice), a.speak_first, None)
    save("p_tokens.npy", p.tokens.numpy().astype(np.int32))
    save("p_model.npy", p.model_latents.numpy())
    save("p_user.npy", p.user_latents.numpy())
    save("p_mmask.npy", p.model_mask.numpy().astype(np.int32))
    save("p_umask.npy", p.user_mask.numpy().astype(np.int32))
    hs = {}
    hooks = [layer.register_forward_hook(lambda m, i, o, l=l: hs.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, -1].float().numpy().copy()))
             for l, layer in enumerate(model.lm.layers)]
    cache = model.new_cache()
    with torch.no_grad():
        emb = model.embed(p.tokens[None], p.model_latents[None], p.user_latents[None], p.model_mask[None], p.user_mask[None])
        save("p_emb.npy", emb[0].float().numpy())
        hidden = model(emb, cache)[:, -1]
    for h in hooks:
        h.remove()
    for l, v in hs.items():
        save(f"p_h{l}.npy", v)
    save(f"p_h{len(hs)}.npy", hidden[0].float().numpy())
    logits = model.logits(hidden)[0]
    save("p_logits.npy", logits.numpy())
    print(f"prompt: {p.length} positions ({len(system_prompt_ids(model.tokenizer, card))} card ids), argmax {int(logits.argmax())}", flush=True)

    # ---- frames, teacher-forced (a greedy session: argmax token, the head's latent from a fixed noise)
    gen = torch.Generator().manual_seed(1234)
    K = min(a.frames, mean.shape[0])
    toks, mlat, ulat, hid, lgs, noises, lats = [], [], [], [], [], [], []
    with torch.no_grad():
        for k in range(K + 1):
            lg = model.logits(hidden)[0]
            noise = torch.randn(1, model.config.latent_dim, generator=gen)
            lat = model.unscale_latents(model.head.sample(hidden, noise, 1.5))[0]
            hid.append(hidden[0].float().numpy())
            lgs.append(lg.numpy())
            noises.append(noise[0].numpy())
            lats.append(lat.numpy())
            if k == K:
                break
            if k == 0 and a.speak_first:
                # the session's force_text: frame 0 is a word (PAD, EOP, EOS banned)
                ban = lg.clone()
                ban[[model.config.pad_token_id, model.config.eop_token_id, model.config.eos_token_id]] = float("-inf")
                tok = int(ban.argmax())
            else:
                tok = int(lg.argmax())
            u = torch.from_numpy(mean[k])[None]
            e = model.embed(torch.tensor([[tok]]), lat[None, None], u[None])
            hidden = model(e, cache)[:, -1]
            toks.append(tok)
            mlat.append(lat.numpy())
            ulat.append(mean[k])
    save("f_tok.npy", np.array(toks, np.int32))
    save("f_model.npy", np.stack(mlat))
    save("f_user.npy", np.stack(ulat))
    save("f_hidden.npy", np.stack(hid))
    save("f_logits.npy", np.stack(lgs))
    save("f_noise.npy", np.stack(noises))
    save("f_latent.npy", np.stack(lats))
    print(f"frames: {K}, tokens {toks}", flush=True)


if __name__ == "__main__":
    main()
