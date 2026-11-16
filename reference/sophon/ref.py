"""Sophon-Realtime's thinker (DuplexJev: Qwen3-ASR audio encoder -> Ultravox projector -> Qwen3-32B) on the CPU in
float32 - the reference for nextsycl's audio engine (audio/sophon): each stage's tensors dumped as .npy for
`nextsycl audio check`.

The audio side is the release's own code (sophon-realtime/finetune/sophon_swift: modeling_qwen3_asr_encoder.py,
modeling_duplexjev.py's projector, processing.py's features), loaded from CODE (the cloned repository). The language
model is too large for this box's memory in float32 (64 layers, 31 B weights): it is run a layer at a time - each
layer's FP8 matrices (per-output-channel scales) made float32, applied to every position, dropped - the maths of
transformers' Qwen3 (RMS norm, q / k norms, rotate-half RoPE at theta 5e6, grouped attention, SwiGLU). The served model
quantizes activations to FP8 per token as well (vLLM w8a8); this reference keeps them float32 (what nextsycl does).

    python -I ref.py CODE THINKER reply  --wav q16k.wav --out DIR [--lang en] [--follow ids.npy]
    python -I ref.py CODE THINKER check  --wav q16k.wav --out DIR [--lang en] [--said "..."]

reply: system prompt + the user's speech (duplex.py's reply request); check: the turn question (duplex.py's
turn_question) over the speech. --follow: token ids written after the prompt (nextsycl's greedy answer): they are
appended and the reference's argmax at every position dumped (a teacher-forced comparison of the writing).

Dumps (DIR): request.json, audio16k.npy (the clip), mel.npy [128, L], tower.npy [n, 896], audio_emb.npy [ntok, 5120], ids.npy (the prompt),
embeds.npy [S, 5120] (the LM's input rows), h{l}.npy (the residual stream after layer l, for l in LAYERS),
norm.npy (after the final norm, last row), logits.npy (the last row), argmax.npy (every row's argmax),
and for check: letters.json (each option letter's logit and P(finished), P(backchannel)).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import types
import wave

import numpy as np
import torch
import torch.nn.functional as F

LAYERS = [0, 1, 15, 31, 47, 62, 63]
SR = 16000
SYS_EN = ("You are Sophon, a model of the Sophon series trained by Adventists AI (AI降临派); you can hear, see and speak. You hear the user's speech directly, "
          "and your reply will be spoken aloud. Respond the way a person would in conversation, in one or two natural spoken sentences, paying attention to "
          "the user's emotion. Write only the words to be spoken: no parenthetical notes, emoji or formatting; write numbers and symbols the way they are "
          "read aloud. When it fits, you may use sound tags for laughter, coughs and emphasis.")
LET = "ABCDEFGHIJ"
TURN_Q = ("Has the user finished speaking?", ["finished, the assistant can reply", "not finished, still talking",
                                               "just a backchannel, not taking the turn", "hesitating or asking to wait"])
TPL = ("Question: {q}", "Options:", "Answer with only the letter of the correct option.", "The user said: <|audio|>", 'The assistant had just said: "{c}"')


def perm_of(qid, opts, seed=0):
    key = "\x1f".join([qid, "\x1e".join(opts), str(seed)]).encode()
    rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))
    p = list(range(len(opts)))
    rng.shuffle(p)
    return p


def turn_question(said):
    q, opts = TURN_Q
    tq, to, ti, head, sd = TPL
    p = perm_of("turn", opts)
    s = " ".join((said or "").split())[-300:]
    text = (sd.format(c=s) + "\n\n" if s else "") + head + "\n\n" + "\n".join([tq.format(q=q), "", to] + [f"{LET[i]}. {opts[j]}" for i, j in enumerate(p)] + ["", ti])
    return text, p


def load_module(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    if package:
        m.__package__ = package
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def code_modules(code):
    """The release's audio modules as a package of their own (its __init__ is not run: it imports ms-swift's register)"""
    d = os.path.join(code, "finetune", "sophon_swift")
    pkg = types.ModuleType("sophon_swift")
    pkg.__path__ = [d]
    sys.modules["sophon_swift"] = pkg
    load_module("sophon_swift.configuration_qwen3_asr_encoder", os.path.join(d, "configuration_qwen3_asr_encoder.py"), "sophon_swift")
    enc = load_module("sophon_swift.modeling_qwen3_asr_encoder", os.path.join(d, "modeling_qwen3_asr_encoder.py"), "sophon_swift")
    dj = load_module("sophon_swift.modeling_duplexjev", os.path.join(d, "modeling_duplexjev.py"), "sophon_swift")
    proc = load_module("sophon_swift.processing", os.path.join(d, "processing.py"), "sophon_swift")
    return enc, dj, proc


def read_wav16k(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR, f"{path}: 16 kHz expected (resample it first)"
        assert w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
    return x


class Shards:
    """The thinker's safetensors shards, a tensor at a time"""

    def __init__(self, d):
        from safetensors import safe_open
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        # the vision tower's file may be left out of the download (the audio path does not read it)
        idx = {k: f for k, f in idx.items() if os.path.exists(os.path.join(d, f))}
        self.where = idx
        self.files = {f: safe_open(os.path.join(d, f), "pt") for f in sorted(set(idx.values()))}

    def get(self, name):
        return self.files[self.where[name]].get_tensor(name)

    def has(self, name):
        return name in self.where

    def linear(self, name):
        """A matrix as float32: FP8 times its per-output-channel scale, or bf16 as is"""
        w = self.get(name + ".weight")
        if self.has(name + ".weight_scale"):
            s = self.get(name + ".weight_scale").float().reshape(-1, 1)
            return w.float() * s
        return w.float()


def rms(x, w, eps=1e-6):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w.float()


def rope(x, pos, theta=5e6):
    """rotate-half RoPE: x [S, H, 128]"""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = (pos.double()[:, None] * inv[None, :])
    cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def lm_forward(st, x, out, dump_layers=LAYERS):
    """x [S, 5120] float32 -> the final norm's rows; dumps the residual after the LAYERS"""
    S = x.shape[0]
    pos = torch.arange(S)
    P = "language_model.model."
    n_layer = 64
    for l in range(n_layer):
        t0 = time.time()
        p = f"{P}layers.{l}."
        h = rms(x, st.get(p + "input_layernorm.weight"))
        q = h @ st.linear(p + "self_attn.q_proj").T
        k = h @ st.linear(p + "self_attn.k_proj").T
        v = h @ st.linear(p + "self_attn.v_proj").T
        q = rms(q.view(S, 64, 128), st.get(p + "self_attn.q_norm.weight"))
        k = rms(k.view(S, 8, 128), st.get(p + "self_attn.k_norm.weight"))
        q, k = rope(q, pos), rope(k, pos)
        v = v.view(S, 8, 128)
        k = k.repeat_interleave(8, dim=1)
        v = v.repeat_interleave(8, dim=1)
        a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True)[0]
        a = a.transpose(0, 1).reshape(S, 64 * 128)
        x = x + a @ st.linear(p + "self_attn.o_proj").T
        h = rms(x, st.get(p + "post_attention_layernorm.weight"))
        g = h @ st.linear(p + "mlp.gate_proj").T
        u = h @ st.linear(p + "mlp.up_proj").T
        x = x + (F.silu(g) * u) @ st.linear(p + "mlp.down_proj").T
        if l in dump_layers:
            np.save(os.path.join(out, f"h{l}.npy"), x.numpy())
        print(f"layer {l}: {time.time() - t0:.1f} s", flush=True)
    return rms(x, st.get(P + "norm.weight"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("thinker")
    ap.add_argument("mode", choices=["reply", "check"])
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--said", default="")
    ap.add_argument("--follow", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    enc_mod, dj, proc_mod = code_modules(a.code)
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    tok = AutoTokenizer.from_pretrained(a.thinker)
    fe = WhisperFeatureExtractor.from_pretrained(os.path.join(a.thinker, "audio"))
    cfg = dj.DuplexJevConfig(**json.load(open(os.path.join(a.thinker, "audio", "config.json"))))
    proc = proc_mod.SophonRealtimeProcessor(tok, fe, stack_factor=cfg.stack_factor, audio_token_index=cfg.audio_token_index)

    # the features (processing.py) and the audio tower + projector (modeling_duplexjev.py), float32
    wav = read_wav16k(a.wav)
    np.save(os.path.join(a.out, "audio16k.npy"), wav)
    feats = proc.audio_features([wav])
    mel = feats["audio_values"][0]
    np.save(os.path.join(a.out, "mel.npy"), mel.numpy())
    st = Shards(a.thinker)
    tower = enc_mod.Qwen3ASREncoder(cfg.audio_config).float()
    sd = {k[len("audio_tower."):]: st.get(k).float() for k in st.where if k.startswith("audio_tower.")}
    missing = tower.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing
    projector = dj.DuplexJevProjector(cfg).float()
    projector.load_state_dict({k[len("multi_modal_projector."):]: st.get(k).float() for k in st.where if k.startswith("multi_modal_projector.")})
    x = feats["audio_values"].float()
    h = tower(x, audio_len=feats["audio_lens"]).last_hidden_state
    n = int(enc_mod._get_feat_extract_output_lengths(torch.tensor(x.shape[-1]), tower.n_window))
    if h.shape[1] < n:
        h = F.pad(h, (0, 0, 0, n - h.shape[1]))
    np.save(os.path.join(a.out, "tower.npy"), h[0, :n].numpy())
    emb = projector(h[:, :n])[0, : int(feats["audio_token_len"][0])]
    np.save(os.path.join(a.out, "audio_emb.npy"), emb.numpy())
    ntok = emb.shape[0]
    print(f"audio: {len(wav) / SR:.2f} s, {mel.shape[-1]} mel frames, {n} tower rows, {ntok} tokens")

    # the prompt (duplex.py's rendering; vLLM's processor expands each <|audio|> to its tokens)
    if a.mode == "reply":
        msgs = [{"role": "system", "content": SYS_EN}, {"role": "user", "content": "<|audio|>"}]
        perm = None
    else:
        text, perm = turn_question(a.said)
        msgs = [{"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False).input_ids
    at = ids.index(cfg.audio_token_index)
    ids = ids[:at] + [cfg.audio_token_index] * ntok + ids[at + 1:]
    n_prompt = len(ids)
    if a.follow:
        ids = ids + [int(t) for t in np.load(a.follow).reshape(-1)]
    np.save(os.path.join(a.out, "ids.npy"), np.array(ids, np.int32))
    json.dump({"mode": a.mode, "wav": a.wav, "lang": a.lang, "prompt": prompt, "n_prompt": n_prompt, "ntok": ntok, "perm": perm,
               "follow": a.follow}, open(os.path.join(a.out, "request.json"), "w"), ensure_ascii=False, indent=1)
    E = st.get("language_model.model.embed_tokens.weight")
    rows = E[torch.tensor(ids)].float()
    rows[at:at + ntok] = emb
    np.save(os.path.join(a.out, "embeds.npy"), rows.numpy())
    del E
    nrm = lm_forward(st, rows, a.out)
    head = st.get("language_model.lm_head.weight").float()
    logits = nrm @ head.T
    np.save(os.path.join(a.out, "norm.npy"), nrm[n_prompt - 1].numpy())
    np.save(os.path.join(a.out, "logits.npy"), logits[n_prompt - 1].numpy())
    np.save(os.path.join(a.out, "argmax.npy"), logits.argmax(-1).numpy().astype(np.int32))
    top = torch.topk(logits[n_prompt - 1], 8)
    print("top:", [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)])
    if a.mode == "check":
        lt = {}
        for i in range(len(perm)):
            tid = tok(LET[i], add_special_tokens=False).input_ids
            lt[LET[i]] = float(logits[n_prompt - 1, tid[0]])
        z = np.array([lt[LET[i]] for i in range(len(perm))])
        z = np.exp(z - z.max())
        z /= z.sum()
        res = {"letters": lt, "p_finished": float(z[perm.index(0)]), "p_backchannel": float(z[perm.index(2)])}
        json.dump(res, open(os.path.join(a.out, "letters.json"), "w"), indent=1)
        print(res)
    if a.follow:
        am = logits.argmax(-1)[n_prompt - 1:-1].numpy()
        fol = np.load(a.follow).reshape(-1)
        agree = int((am == fol).sum())
        print(f"teacher-forced: {agree}/{len(fol)} of nextsycl's tokens are the reference's argmax")
        print("reference greedy continuation:", repr(tok.decode(am.tolist())))


if __name__ == "__main__":
    main()
