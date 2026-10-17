"""Encoderless enrolment for Voxtral 4B TTS (experimental): a voice's codes fitted to its recording through the codec's
released decoder, on the GPU (PyTorch XPU) - docs/proposals/voxtral-encoderless-clone.md, stage A.

The open checkpoint has the codec's decoder but not its encoder, so a recording cannot be turned into the frames of
codes the backbone is prompted with. Here the codes are searched for instead: per 80 ms frame (1,920 samples) a continuous 256-d
semantic vector (the decoder sees its nearest codeword of the semantic VQ codebook, straight through) and 36
continuous acoustic values (tanh, scaled to the 21 FSQ levels and rounded, straight through), fitted so that the
decoder (upstream's own VoxtralTTSAudioTokenizer, from the safetensors) reproduces the clip: L1 on the waveform plus
a multi-resolution STFT loss (spectral convergence and log magnitude) plus log-mel L1. A first share of the steps
runs relaxed (the decoder sees the continuous vectors and unrounded levels), then quantized; the semantic vectors are
pulled toward their codewords (a commitment term). The result is the frames' codes as the model emits them (the
specials' 2 on each, [F, 37] int64); the engine appends the end-of-audio frame and embeds them as upstream's
encode_tokens does (the presets' rows are made the same way).

    python enroll.py CHECKPOINT_DIR --code VLLM_OMNI_DIR --audio sample.wav --out codes.npy [--device xpu] [--steps 800]
                     [--relax 0.3] [--lr 0.05] [--seconds 15] [--report report.json]

Also written: <out>.wav (the decoder's speech from the fitted codes) and the report (losses, the fit's log-mel
distance, the time taken).

The in-distribution prior (--text, the clip's transcript; --prior W): the backbone (Ministral 3B, bf16 on the card,
frozen) reads [a preset voice] [the transcript] and then the fitted frames teacher-forced, and the negative
log-likelihood of each frame's semantic code under its semantic head is added to the loss with weight W (every
--prior-every steps of the quantized phase; the code's choice passes its gradient through a softmax over the
codewords' distances). With --text and W 0 the NLL is only measured: how likely the backbone finds the fitted codes
for that text (the presets' own codes are its training distribution).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref import audio_args, make_codec, modules  # noqa: E402

RATE = 24000
PATCH = 240
# a frame (12.5 a second): the patch times the decoder's three 2x upsamplings
FRAME = 1920
LEVELS = 21
SPECIAL = 2


def load_audio(path, seconds):
    import soundfile as sf

    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(w.mean(1))
    if sr != RATE:
        import torchaudio

        w = torchaudio.functional.resample(w, sr, RATE)
    w = w[: int(seconds * RATE)]
    # zeros to a whole frame (upstream's encoder pads to a patch, its strided convolutions to a frame)
    pad = (-len(w)) % FRAME
    return F.pad(w, (0, pad)), pad


def stft_mag(x, n_fft, hop):
    """|STFT|, floored: a clip's digital silence (gated recordings: exact zeros) would otherwise weigh most in the log terms"""
    win = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, window=win, return_complex=True).abs().clamp_min(1e-4)


def mel_basis(n_fft, n_mels, device):
    """A Slaney-style mel filterbank [n_mels, n_fft / 2 + 1] (no torchaudio needed)"""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    freqs = np.linspace(0, RATE / 2, n_fft // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(0.0), hz2mel(RATE / 2), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return torch.from_numpy(fb).to(device)


class Loss:
    """L1 waveform + multi-resolution STFT (spectral convergence + log magnitude) + log-mel L1 against a fixed clip"""

    RES = [(512, 128), (1024, 256), (2048, 512)]

    def __init__(self, ref, wave_w=1.0, stft_w=1.0, mel_w=1.0):
        self.ref = ref
        self.w = (wave_w, stft_w, mel_w)
        self.mags = [stft_mag(ref, n, h) for n, h in self.RES]
        self.mel = mel_basis(1024, 80, ref.device)
        self.ref_mel = torch.log(self.mel @ self.mags[1] + 1e-4)

    def __call__(self, y):
        wave = (y - self.ref).abs().mean()
        st = 0.0
        for (n, h), m in zip(self.RES, self.mags):
            g = stft_mag(y, n, h)
            st = st + torch.linalg.norm(m - g) / torch.linalg.norm(m) + (torch.log(g) - torch.log(m)).abs().mean()
        st = st / len(self.RES)
        mel = (torch.log(self.mel @ stft_mag(y, 1024, 256) + 1e-4) - self.ref_mel).abs().mean()
        return self.w[0] * wave + self.w[1] * st + self.w[2] * mel, {"wave": float(wave), "stft": float(st), "mel": float(mel)}


class Prior:
    """The backbone's NLL of a run of fitted frames' semantic codes after a prompt (vLLM's MistralForCausalLM written out,
    bf16, frozen)"""

    def __init__(self, sf, params, dev, prompt_rows):
        g = lambda n: sf.get_tensor(n).to(dev, torch.bfloat16)
        self.p = params
        self.layers = [{k: g(f"layers.{i}.{k}.weight") for k in ["attention_norm", "attention.wq", "attention.wk", "attention.wv", "attention.wo",
                                                                  "ffn_norm", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]}
                       for i in range(params["n_layers"])]
        self.norm = g("norm.weight")
        self.emb = g("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight")
        self.head = g("acoustic_transformer.semantic_codebook_output.weight")
        self.prompt = prompt_rows.to(dev, torch.bfloat16)
        hd = params["head_dim"]
        inv = 1.0 / (params["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        n = len(prompt_rows) + 2048
        ang = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
        self.cos, self.sin = torch.cos(ang).to(dev), torch.sin(ang).to(dev)
        for l in self.layers:
            for w in l.values():
                w.requires_grad_(False)

    def rope(self, x):
        L = x.shape[0]
        c, s = self.cos[:L, None, :].to(x.dtype), self.sin[:L, None, :].to(x.dtype)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)

    def forward(self, x):
        p = self.p
        L, hd, nh, nkv = x.shape[0], p["head_dim"], p["n_heads"], p["n_kv_heads"]
        eps = p["norm_eps"]
        rms = lambda v, w: (v.float() * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True) + eps)).to(v.dtype) * w
        for l in self.layers:
            h = rms(x, l["attention_norm"])
            q = self.rope((h @ l["attention.wq"].T).view(L, nh, hd)).transpose(0, 1)
            k = self.rope((h @ l["attention.wk"].T).view(L, nkv, hd)).repeat_interleave(nh // nkv, 1).transpose(0, 1)
            v = (h @ l["attention.wv"].T).view(L, nkv, hd).repeat_interleave(nh // nkv, 1).transpose(0, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(0, 1).reshape(L, nh * hd)
            x = x + a @ l["attention.wo"].T
            h = rms(x, l["ffn_norm"])
            x = x + (F.silu(h @ l["feed_forward.w1"].T) * (h @ l["feed_forward.w3"].T)) @ l["feed_forward.w2"].T
        return rms(x, self.norm)

    def nll(self, sem_p, levels):
        """sem_p [F, 8192]: each frame's semantic code as a distribution (one-hot forward, straight through);
        levels [F, 36] long (0..20): the frames' summed embeddings teacher-forced after the prompt; the mean NLL"""
        codes = SPECIAL + 8192
        sem_rows = sem_p.to(self.emb.dtype) @ self.emb[SPECIAL:codes]
        idx = codes + torch.arange(36, device=levels.device) * (LEVELS + SPECIAL) + levels + SPECIAL
        rows = sem_rows + self.emb[idx].sum(1)
        x = torch.cat([self.prompt, rows[:-1]], 0)
        h = self.forward(x)[len(self.prompt) - 1:]
        # upstream masks [EMPTY_AUDIO] and the padding: [END_AUDIO] and the codes compete
        logp = torch.log_softmax((h @ self.head[1:codes].T).float(), -1)[:, 1:]
        return -(sem_p.float() * logp).sum(-1).mean()


def prompt_rows(sf, model_dir, text, voice):
    """mistral-common's speech prompt for `text` with a preset voice, its rows (the voice's at its [AUDIO] positions)"""
    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_file(os.path.join(model_dir, "tekken.json"))
    ids = tok.instruct_tokenizer.encode_speech_request(SpeechRequest(input=text, voice=voice)).tokens
    v = torch.load(os.path.join(model_dir, "voice_embedding", f"{voice}.pt"), map_location="cpu").float()
    emb = sf.get_tensor("mm_audio_embeddings.tok_embeddings.weight")
    rows = emb[torch.tensor(ids)].float()
    rows[torch.tensor([i for i, t in enumerate(ids) if t == 24])] = v
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--code", required=True, help="a checkout of github.com/vllm-project/vllm-omni")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--relax", type=float, default=0.5, help="the share of the steps run without quantization")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--commit", type=float, default=0.25, help="the semantic vectors' pull toward their codewords")
    ap.add_argument("--seconds", type=float, default=15.0, help="the most of the clip used")
    ap.add_argument("--bf16", action="store_true", help="the decoder under bf16 autocast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None)
    ap.add_argument("--text", default=None, help="the clip's transcript: the prior's text (measured; added with --prior)")
    ap.add_argument("--prior", type=float, default=0.0, help="the weight of the backbone's NLL of the semantic codes")
    ap.add_argument("--prior-every", type=int, default=4, help="the prior on every Nth step of the quantized phase")
    ap.add_argument("--prior-voice", default="neutral_female", help="the preset voice ahead of the transcript in the prior's prompt")
    ap.add_argument("--measure", default=None, help="only the prior's NLL of these codes [F, 37] (with --text): a check of it on the model's own")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    params = json.load(open(os.path.join(a.model, "params.json")))
    from safetensors import safe_open

    sf = safe_open(os.path.join(a.model, "consolidated.safetensors"), framework="pt")
    codec = make_codec(modules(a.code, flash=dev.type != "cpu"), sf, params, audio_args(params)).to(dev)
    for p in codec.parameters():
        p.requires_grad_(False)
    cb = codec.quantizer.semantic_codebook.embedding.to(dev)  # [8192, 256]
    if a.measure:
        c = torch.from_numpy(np.load(a.measure)).long().to(dev)
        c = c[c[:, 0] != 1]
        pr = Prior(sf, params, dev, prompt_rows(sf, a.model, a.text, a.prior_voice))
        with torch.no_grad():
            nll = pr.nll(F.one_hot(c[:, 0] - SPECIAL, 8192).float(), (c[:, 1:] - SPECIAL).clamp(0, LEVELS - 1))
        print(json.dumps({"measure": a.measure, "frames": len(c), "nll": float(nll)}))
        return
    ref, pad = load_audio(a.audio, a.seconds)
    ref = ref.to(dev)
    frames = len(ref) // FRAME
    loss_fn = Loss(ref)
    prior = Prior(sf, params, dev, prompt_rows(sf, a.model, a.text, a.prior_voice)) if a.text else None
    t_load = time.time() - t0

    # the parameters: semantic vectors from random codewords, acoustic values at the middle level
    z = cb[torch.randint(0, cb.shape[0], (frames,), device=dev)].clone() + 0.01 * torch.randn(frames, cb.shape[1], device=dev)
    z.requires_grad_(True)
    ac = torch.zeros(frames, 36, device=dev, requires_grad=True)
    opt = torch.optim.Adam([z, ac], lr=a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)) * 0.9 + 0.1))
    relax_until = int(a.relax * a.steps)

    def quantized(z, ac, relaxed):
        v = torch.tanh(ac)
        scaled = (v + 1) / 2 * (LEVELS - 1)
        if relaxed:
            sem, lev = z, scaled
        else:
            idx = torch.cdist(z.detach(), cb).argmin(-1)
            q = cb[idx]
            sem = z + (q - z).detach()
            lev = scaled + (scaled.round() - scaled).detach()
        acoustic = lev * 2 / (LEVELS - 1) - 1
        return torch.cat([sem, acoustic], -1)

    def semantic_choice(z, ac):
        """each frame's semantic code one-hot (forward) with a softmax over the codewords' distances behind it (backward),
        and the frames' levels"""
        d2 = torch.cdist(z, cb).pow(2)
        temp = d2.min(-1).values.median().detach().clamp_min(1e-6)
        soft = torch.softmax(-d2 / temp, -1)
        hard = F.one_hot(d2.argmin(-1), cb.shape[0]).to(soft.dtype)
        lev = ((torch.tanh(ac) + 1) / 2 * (LEVELS - 1)).round().long().clamp(0, LEVELS - 1).detach()
        return hard + soft - soft.detach(), lev

    def decode(emb):
        x = emb.T.unsqueeze(0)  # [1, 292, F]
        if a.bf16:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                y = codec._forward_decoder(x)
            return y.float()[0, 0]
        return codec._forward_decoder(x)[0, 0]

    hist = []
    t1 = time.time()
    for step in range(a.steps):
        relaxed = step < relax_until
        y = decode(quantized(z, ac, relaxed))
        loss, parts = loss_fn(y)
        if not relaxed:
            idx = torch.cdist(z.detach(), cb).argmin(-1)
            loss = loss + a.commit * F.mse_loss(z, cb[idx])
            if prior is not None and a.prior > 0 and step % a.prior_every == 0:
                nll = prior.nll(*semantic_choice(z, ac))
                loss = loss + a.prior * nll
                parts["nll"] = float(nll)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == a.steps - 1:
            hist.append({"step": step, "loss": float(loss), **parts, "relaxed": relaxed, "s": round(time.time() - t1, 1)})
            print(json.dumps(hist[-1]), flush=True)
    t_fit = time.time() - t1

    with torch.no_grad():
        idx = torch.cdist(z, cb).argmin(-1)
        lev = ((torch.tanh(ac) + 1) / 2 * (LEVELS - 1)).round().long().clamp(0, LEVELS - 1)
        codes = torch.cat([idx[:, None] + SPECIAL, lev + SPECIAL], 1).cpu().numpy().astype(np.int64)
        # the fit: the decoder's speech from the codes as the engine will read them (upstream's decode path)
        y = codec.decode(torch.from_numpy(codes - SPECIAL).to(dev).T.unsqueeze(0), dtype=torch.float32)[0, 0]
        _, final = loss_fn(y)
        if prior is not None:
            final["nll"] = float(prior.nll(*semantic_choice(z, ac)))
    np.save(a.out, codes)
    try:
        import soundfile

        soundfile.write(os.path.splitext(a.out)[0] + ".wav", y.cpu().numpy(), RATE)
    except ImportError:
        pass
    rep = {"frames": frames, "seconds": round(frames * FRAME / RATE, 2), "device": str(dev), "steps": a.steps, "relax": a.relax, "lr": a.lr,
           "commit": a.commit, "bf16": a.bf16, "prior": a.prior, "prior_every": a.prior_every, "load_s": round(t_load, 1), "fit_s": round(t_fit, 1), "final": final,
           "semantic_codes_used": int(len(set(idx.tolist()))), "history": hist}
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "history"}))


if __name__ == "__main__":
    main()
