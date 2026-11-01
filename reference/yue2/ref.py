"""YuE2 (M-A-P's YuE2-3B and YuE2-Vae) on the CPU in float32 - the reference for nextsycl's engine (audio/yue2): the
stages' tensors dumped for `nextsycl audio check`. The arithmetic is upstream's own (the yue2_infer 0.1.5 wheel,
installed as it is: protocol.py's prompts, sampling.py's draws, nar.py's CachedNAR and midpoint solver, modeling_vae.py's
tiled decode); the language model's weights are nextsycl's file (the Q8_0 GGUF dequantized, `--weights q8`, the
default) or the bf16 checkpoint (`--weights bf16`), so a check can tell the port's error from the file's.

    python -I ref.py [--device cuda:0 --dtype bf16] --lm LM_DIR --vae VAE_DIR --gguf yue2-lm-q8_0.gguf --out DIR
        [--style TEXT] [--lyrics-file FILE] [--cot off|melody|full] [--abc-file FILE] [--seed N] [--cfg X]
        [--abc-max N] [--sem-max N] [--steps N] [--stages tokens,ar,nar,vae]

LM_DIR: YuE2-3B as downloaded (config.json, model.safetensors, qwen.tiktoken); VAE_DIR: YuE2-Vae's. The song is cut
short for the CPU (on a 12 GB card: `--device cuda:N --dtype bf16`, the language model in bfloat16 as upstream runs it; `--abc-max` score tokens, `--sem-max` semantic tokens, 25 a second; `--steps` ODE steps): the
stages are the same at any length.

Dumps (DIR, .npy float32 unless said): request.json (style, lyrics, cot, seed, cfg as the pipeline takes them);
prefix.npy (int32: the first phase's prompt - token_prefixes with no score yet), text.txt (the prompt's text);
abc.npy (int32: the score's tokens drawn, empty for cot off) and abc.txt; sem_prefix.npy (int32: the semantic phase's
prompt), neg_prefix.npy (int32: its classifier-free twin, when the guidance is not 1); codes.npy (int32: the semantic
codes drawn, 0..32767); ar_rows.npy (int32: positions in the whole sequence prefix + abc + ABC_END + MUSIC_START + codes)
and ar_logits [rows, 184704] (the logits there, one causal pass over the whole sequence: the teacher-forced AR);
neg_logits [184704] (the twin's prefill logits); noise [frames, 64] (the chunk's torch.randn draw); vel0 [frames, 64]
(the first velocity, at t = 1); states [steps + 1, frames, 64] (the ODE's state before each midpoint step and after the
last); mids [steps, frames, 64] (each step's midpoint state); latents [frames, 64]; audio [samples, 2] (the tiled
decode, clamped) and ref.wav.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)


def save(out, name, t):
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    if a.dtype not in (np.int32,):
        a = a.astype(np.float32)
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(a))


def ids(out, name, v):
    np.save(os.path.join(out, name + ".npy"), np.asarray(list(v), dtype=np.int32))


# ---- the GGUF's tensors (Q8_0, F32, BF16) as float32, by their names in the checkpoint

def gguf_tensors(path):
    f = open(path, "rb")

    def r(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    def rs():
        n, = r("Q")
        return f.read(n)

    def skip(t):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t == 8:
            rs()
        elif t == 9:
            et, n = r("IQ")
            for _ in range(n):
                skip(et)
        else:
            f.read(sizes[t])

    magic, ver, nt, nkv = r("4sIQQ")
    assert magic == b"GGUF", path
    align = 32
    for _ in range(nkv):
        k = rs().decode()
        t, = r("I")
        if k == "general.alignment":
            align, = r("I")
        else:
            skip(t)
    infos = []
    for _ in range(nt):
        name = rs().decode()
        nd, = r("I")
        dims = r("Q" * nd)
        ty, off = r("IQ")
        infos.append((name, dims, ty, off))
    base = (f.tell() + align - 1) // align * align
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for name, dims, ty, off in infos:
        n = int(np.prod(dims))
        shape = tuple(reversed(dims))
        o = base + off
        if ty == 0:
            a = np.frombuffer(mm[o:o + 4 * n], dtype=np.float32).copy()
        elif ty == 30:
            a = (np.frombuffer(mm[o:o + 2 * n], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif ty == 8:
            blocks = np.frombuffer(mm[o:o + n // 32 * 34], dtype=np.uint8).reshape(-1, 34)
            d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
            q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
            a = (q * d).reshape(-1)
        else:
            raise SystemExit(f"{name}: GGUF type {ty} is not read here")
        yield name, a.reshape(shape)


def checkpoint_name(g):
    """The checkpoint's name of a GGUF tensor (the inverse of HOT-Step's convert-yue2.py)"""
    fixed = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight",
             "vae2llm.weight": "vae2llm.weight", "vae2llm.bias": "vae2llm.bias", "llm2vae.weight": "llm2vae.weight", "llm2vae.bias": "llm2vae.bias",
             "time_embd.0.weight": "time_embedder.mlp.0.weight", "time_embd.0.bias": "time_embedder.mlp.0.bias",
             "time_embd.1.weight": "time_embedder.mlp.2.weight", "time_embd.1.bias": "time_embedder.mlp.2.bias",
             "latent_pos_embed.weight": "latent_pos_embed.pe"}
    if g in fixed:
        return fixed[g]
    _, i, rest = g.split(".", 2)
    p = f"model.layers.{i}."
    for pre, attn, norm in (("nar_", "nar_self_attn", "nar_input_layernorm"), ("", "self_attn", "input_layernorm")):
        table = {f"{pre}attn_norm.weight": norm + ".weight", f"{pre}attn_q.weight": attn + ".q_proj.weight", f"{pre}attn_k.weight": attn + ".k_proj.weight",
                 f"{pre}attn_v.weight": attn + ".v_proj.weight", f"{pre}attn_output.weight": attn + ".o_proj.weight",
                 f"{pre}attn_q_norm.weight": attn + ".q_norm.weight", f"{pre}attn_k_norm.weight": attn + ".k_norm.weight"}
        if rest in table:
            return p + table[rest]
    mlp = {"ffn_norm.weight": "post_attention_layernorm.weight", "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
           "ffn_down.weight": "mlp.down_proj.weight", "nar_ffn_norm.weight": "nar_pre_mlp_layernorm.weight",
           "nar_ffn_gate.weight": "nar_mlp.gate_proj.weight", "nar_ffn_up.weight": "nar_mlp.up_proj.weight", "nar_ffn_down.weight": "nar_mlp.down_proj.weight"}
    return p + mlp[rest]


def load_lm(a):
    from yue2.modeling_yue2 import YuE2ForCausalLM
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    m = YuE2ForCausalLM.from_pretrained(a.lm, dtype=dt, low_cpu_mem_usage=True).eval()
    if a.weights == "q8":
        sd = m.state_dict()
        seen = 0
        for g, arr in gguf_tensors(a.gguf):
            name = checkpoint_name(g)
            t = sd[name]
            if tuple(t.shape) != arr.shape:
                raise SystemExit(f"{g} -> {name}: {arr.shape} vs {tuple(t.shape)}")
            t.copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(t.dtype))
            seen += 1
        print(f"ref: {seen} tensors from {a.gguf}", file=sys.stderr)
    return m.to(a.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--gguf", default="")
    ap.add_argument("--weights", default="q8", choices=["q8", "bf16"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="Pop, upbeat, female vocal, piano, drums, bright, catchy")
    ap.add_argument("--lyrics-file", default="")
    ap.add_argument("--cot", default="off")
    ap.add_argument("--abc-file", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--abc-max", type=int, default=256)
    ap.add_argument("--sem-max", type=int, default=250)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--stages", default="tokens,ar,nar,vae")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    stages = a.stages.split(",")
    from yue2.protocol import (SongRequest, GenerationConfig, Sampling, token_prefixes, negative_prefix, CODEC_OFFSET, ABC_END, MUSIC_START,
                               MUSIC_END)
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    from dataclasses import replace
    lyrics = open(a.lyrics_file, encoding="utf-8").read() if a.lyrics_file else "[Verse]\nLights on the water, the night is young\nWe sing together, the song is sung\n"
    abc_in = open(a.abc_file, encoding="utf-8").read() if a.abc_file else None
    req = SongRequest(style=a.style, lyrics=lyrics, cot=a.cot, seed=a.seed, abc=abc_in, cfg_scale=a.cfg)
    tok = YuE2TextTokenizer(os.path.join(a.lm, "qwen.tiktoken"))
    gc = GenerationConfig()
    with open(os.path.join(a.out, "request.json"), "w", encoding="utf-8") as fh:
        json.dump({"style": req.style, "lyrics": req.lyrics, "cot": req.cot, "seed": req.seed, "cfg": req.guidance, "abc": abc_in,
                   "abc_max": a.abc_max, "sem_max": a.sem_max, "steps": a.steps, "weights": a.weights}, fh, ensure_ascii=False, indent=1)
    prefix = token_prefixes(req, tok)
    ids(a.out, "prefix", prefix)
    with open(os.path.join(a.out, "text.txt"), "w", encoding="utf-8") as fh:
        fh.write(req.text())
    print(f"ref: prefix {len(prefix)} tokens", file=sys.stderr)
    if stages == ["tokens"]:
        return

    from yue2.sampling import generate_tokens
    m = load_lm(a)
    abc_ids = []
    if "ar" in stages:
        if req.cot != "off" and abc_in is None:
            s = replace(gc.abc, max_tokens=a.abc_max, min_tokens=min(gc.abc.min_tokens, a.abc_max))
            abc_ids, timing, trunc = generate_tokens(m, prefix, s, req.seed, "abc", use_cuda_graph=False)
            print(f"ref: abc {len(abc_ids)} tokens, truncated {trunc}, {timing['seconds']:.1f} s", file=sys.stderr)
        elif abc_in is not None:
            abc_ids = tok.encode(abc_in)
        ids(a.out, "abc", abc_ids)
        with open(os.path.join(a.out, "abc.txt"), "w", encoding="utf-8") as fh:
            fh.write(tok.decode(abc_ids))
        sem_prefix = token_prefixes(req, tok, abc_ids if req.cot != "off" else None)
        ids(a.out, "sem_prefix", sem_prefix)
        neg = negative_prefix(req, tok, abc_ids) if req.guidance != 1 else None
        if neg is not None:
            ids(a.out, "neg_prefix", neg)
        s = replace(gc.semantic, max_tokens=a.sem_max, min_tokens=min(gc.semantic.min_tokens, a.sem_max))
        sem, timing, trunc = generate_tokens(m, sem_prefix, s, req.seed, "semantic", negative=neg, cfg_scale=req.guidance,
                                             legacy_off=req.cot == "off", use_cuda_graph=False)
        codes = [t - CODEC_OFFSET for t in sem]
        ids(a.out, "codes", codes)
        print(f"ref: {len(codes)} codes, truncated {trunc}, {timing['seconds']:.1f} s", file=sys.stderr)
        # the teacher-forced logits: one causal pass over the whole sequence
        seq = sem_prefix + sem
        n_abc0 = len(prefix) - 1
        rows = sorted(set([len(prefix) - 1] + [n_abc0 + i for i in range(0, max(1, len(abc_ids)), max(1, len(abc_ids) // 6))][: 7 if abc_ids else 0]
                          + [len(sem_prefix) - 1 + i for i in range(0, max(1, len(sem)), max(1, len(sem) // 8))][:9]))
        rows = [r for r in rows if r < len(seq)]
        out = m(torch.tensor([seq], device=a.device), use_cache=False, logits_to_keep=0).logits[0]
        ids(a.out, "ar_rows", rows)
        save(a.out, "ar_logits", out[rows])
        if neg is not None:
            save(a.out, "neg_logits", m(torch.tensor([neg], device=a.device), use_cache=False, logits_to_keep=1).logits[0, -1])
    if "nar" in stages:
        from yue2.nar import song_chunks, CachedNAR
        sem_prefix = np.load(os.path.join(a.out, "sem_prefix.npy")).tolist()
        codes = np.load(os.path.join(a.out, "codes.npy")).tolist()
        chunks = song_chunks(sem_prefix, codes, req.seed)
        if len(chunks) != 1:
            raise SystemExit(f"{len(chunks)} chunks: the check takes one")
        c = chunks[0]
        save(a.out, "noise", c.noise)
        eng = CachedNAR(m, c)
        steps = a.steps
        state = c.noise.clone().to(a.device, eng.dtype)
        dt = 1.0 / steps
        states, mids = [state.clone()], []
        for step in range(steps):
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
            v = eng.velocity(state, raw)
            if step == 0:
                save(a.out, "vel0", v)
            mid = state - v * (dt / 2)
            mids.append(mid.clone())
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64)).clamp(-20, 20).item()
            state = state - eng.velocity(mid, raw_mid) * dt
            states.append(state.clone())
            print(f"ref: step {step + 1}/{steps}", file=sys.stderr)
        save(a.out, "states", torch.stack(states))
        save(a.out, "mids", torch.stack(mids))
        save(a.out, "latents", state.float())
    if "vae" in stages:
        from yue2.modeling_vae import YuE2VAE
        v = YuE2VAE.from_pretrained(a.vae, decoder_only=True, device="cpu")
        z = torch.from_numpy(np.load(os.path.join(a.out, "latents.npy"))).float().T.unsqueeze(0)
        audio = v.decode_tiled(z, core_frames=512, halo_frames=16, output_device="cpu")
        wav = audio[0].float().clamp(-1, 1).T.contiguous().numpy()
        save(a.out, "audio", wav)
        import soundfile as sf
        sf.write(os.path.join(a.out, "ref.wav"), wav, 48000, subtype="PCM_16")
        print(f"ref: {wav.shape[0] / 48000:.2f} s of audio", file=sys.stderr)


if __name__ == "__main__":
    main()
