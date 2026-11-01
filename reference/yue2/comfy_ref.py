"""YuE2-3B on the CPU in float32 through ComfyUI's own port (comfy/text_encoders/yue2.py, comfy/ldm/yue2/model.py,
comfy/ldm/audio/autoencoder.py: "adapted from M-A-P YuE2") - a second reference for nextsycl's engine where upstream's
wheel (ref.py) is not on the box. Same dumps as ref.py, read by `nextsycl audio check`. The weights are the engine's own
files: yue2-lm-q8_0.gguf dequantized (the AR path into Comfy's text encoder, the NAR path into its diffusion model) and
yue2-vae-standard-f32.gguf (its weight norm folded).

    PYTHONPATH=COMFYUI python -I comfy_ref.py --gguf yue2-lm-q8_0.gguf --vae yue2-vae-standard-f32.gguf --out DIR \
        [--style TEXT] [--lyrics-file FILE] [--cot off|melody|full] [--seed N] [--cfg X] [--abc-max N] [--sem-max N] [--steps N]

The prompt's ids come from a small BPE here (the GGUF's tiktoken pattern and its token table: rank = id, NFC first), not
from the engine's tokenizer and not from a tokenizer.json. What differs from upstream, and what a comparison must allow:
the draws are Comfy's torch.multinomial (no engine shares them: the codes are only ever teacher-forced), and Comfy's
transposed convolution has output_padding = stride % 2 where upstream's has none, so its audio runs 64 samples longer:
the dump is cut to upstream's length (1920 T - 64).
"""
import argparse
import json
import os
import struct
import sys
import unicodedata

import numpy as np
import regex
import torch

sys.argv, ARGV = [sys.argv[0], "--cpu"], sys.argv[1:]
torch.set_grad_enabled(False)


def save(out, name, t):
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(a.astype(np.float32)))


def ids(out, name, v):
    np.save(os.path.join(out, name + ".npy"), np.asarray(list(v), dtype=np.int32))


# ---- the GGUF: metadata and tensors as float32

def gguf(path, want_meta=()):
    f = open(path, "rb")

    def r(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    def rs():
        n, = r("Q")
        return f.read(n)

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def val(t):
        if t == 8:
            return rs()
        if t == 9:
            et, n = r("IQ")
            return [val(et) for _ in range(n)]
        return f.read(sizes[t])

    magic, ver, nt, nkv = r("4sIQQ")
    assert magic == b"GGUF", path
    align, meta = 32, {}
    for _ in range(nkv):
        k = rs().decode()
        t, = r("I")
        if k in want_meta:
            meta[k] = val(t)
        elif k == "general.alignment":
            align, = struct.unpack("<I", val(t))
        else:
            val(t)
    infos = []
    for _ in range(nt):
        name = rs().decode()
        nd, = r("I")
        dims = r("Q" * nd)
        ty, off = r("IQ")
        infos.append((name, dims, ty, off))
    base = (f.tell() + align - 1) // align * align
    mm = np.memmap(path, dtype=np.uint8, mode="r")

    def tensors():
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
    return meta, tensors


# ---- the tokenizer: tiktoken's rule - merge the adjacent pair whose joined bytes rank lowest (rank = id)

def byte_alphabet():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Bpe:
    def __init__(self, meta):
        self.rank = {t.decode("utf-8", "surrogatepass"): i for i, t in enumerate(meta["tokenizer.ggml.tokens"])}
        self.pat = regex.compile(meta["yue2.tokenizer.pretokenize_regex"].decode())
        self.enc = byte_alphabet()

    def encode(self, text):
        out = []
        for m in self.pat.findall(unicodedata.normalize("NFC", text)):
            parts = [self.enc[b] for b in m.encode("utf-8")]
            while len(parts) > 1:
                best, at = None, None
                for i in range(len(parts) - 1):
                    r = self.rank.get(parts[i] + parts[i + 1])
                    if r is not None and (best is None or r < best):
                        best, at = r, i
                if at is None:
                    break
                parts[at:at + 2] = [parts[at] + parts[at + 1]]
            out.extend(self.rank[p] for p in parts)
        return out


# ---- the weights in Comfy's names: q | k | v and gate | up stacked, as its merged projections

def split_weights(tensors):
    ar, nar = {}, {}
    grouped = {}
    for g, a in tensors():
        fixed = {"token_embd.weight": (ar, "embed_tokens.weight"), "output_norm.weight": (None, "norm.weight"), "output.weight": (ar, "lm_head.weight"),
                 "vae2llm.weight": (nar, "vae2llm.weight"), "vae2llm.bias": (nar, "vae2llm.bias"), "llm2vae.weight": (nar, "llm2vae.weight"),
                 "llm2vae.bias": (nar, "llm2vae.bias"), "time_embd.0.weight": (nar, "time_embedder.mlp.0.weight"), "time_embd.0.bias": (nar, "time_embedder.mlp.0.bias"),
                 "time_embd.1.weight": (nar, "time_embedder.mlp.2.weight"), "time_embd.1.bias": (nar, "time_embedder.mlp.2.bias"),
                 "latent_pos_embed.weight": (nar, "latent_pos_embed.pe")}
        t = torch.from_numpy(np.ascontiguousarray(a))
        if g in fixed:
            d, n = fixed[g]
            if d is None:
                ar[n] = t
                nar["model." + n] = t
            else:
                d[n] = t
            continue
        _, i, rest = g.split(".", 2)
        is_nar = rest.startswith("nar_")
        d = nar if is_nar else ar
        pre = "model.layers." if is_nar else "layers."
        rest = rest[4:] if is_nar else rest
        p = f"{pre}{i}."
        table = {"attn_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight", "attn_output.weight": "self_attn.o_proj.weight",
                 "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight", "ffn_down.weight": "mlp.down_proj.weight"}
        if rest in table:
            d[p + table[rest]] = t
        elif rest in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
            grouped.setdefault((id(d), p + "self_attn.qkv_proj.weight"), (d, {}))[1][rest[5]] = t
        elif rest in ("ffn_gate.weight", "ffn_up.weight"):
            grouped.setdefault((id(d), p + "mlp.gate_up_proj.weight"), (d, {}))[1][rest[4:]] = t
        else:
            raise SystemExit(f"{g}: not placed")
    for (_, name), (d, parts) in grouped.items():
        order = ["q", "k", "v"] if "qkv" in name else ["gate.weight", "up.weight"]
        d[name] = torch.cat([parts[o] for o in order])
    return ar, nar


def load_into(m, sd, what):
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if "encoder." not in k]
    if missing or unexpected:
        raise SystemExit(f"{what}: missing {missing[:8]} unexpected {unexpected[:8]}")


def vae_weights(tensors):
    """The GGUF's dec.* in Comfy's OobleckDecoder (decoder.layers...), the weight norm already folded"""
    sd = {}
    for g, a in tensors():
        if not g.startswith("dec."):
            continue
        t = torch.from_numpy(np.ascontiguousarray(a))
        p = g[4:]
        if p.startswith("conv_in."):
            n = "decoder.layers.0." + p[8:]
        elif p.startswith("snake_out."):
            n = "decoder.layers.7." + p[10:]
        elif p.startswith("conv_out."):
            n = "decoder.layers.8." + p[9:]
        else:
            _, b, q = p.split(".", 2)
            base = f"decoder.layers.{b}.layers."
            if q.startswith("snake_pre."):
                n = base + "0." + q[10:]
            elif q.startswith("upsample."):
                n = base + "1." + q[9:]
            else:
                _, r, rest = q.split(".", 2)
                sub = {"snake1.": "0.", "conv1.": "1.", "snake2.": "2.", "conv2.": "3."}
                k = next(k for k in sub if rest.startswith(k))
                n = f"{base}{2 + int(r)}.layers.{sub[k]}{rest[len(k):]}"
        sd[n] = t
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="Pop, upbeat, female vocal, piano, drums, bright, catchy")
    ap.add_argument("--lyrics-file", default="")
    ap.add_argument("--cot", default="full")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--abc-max", type=int, default=160)
    ap.add_argument("--sem-max", type=int, default=225)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--stages", default="ar,nar,vae")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(ARGV)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    stages = a.stages.split(",")

    import comfy.ops
    from comfy.ldm.yue2.model import YuE2
    from comfy.text_encoders import yue2 as cy
    from comfy.ldm.audio.autoencoder import AudioOobleckVAE

    meta, tensors = gguf(a.gguf, ("tokenizer.ggml.tokens", "yue2.tokenizer.pretokenize_regex"))
    tok = Bpe(meta)
    lyrics = open(a.lyrics_file, encoding="utf-8").read() if a.lyrics_file else "[Verse]\nLights on the water, the night is young\nWe sing together, the song is sung\n"
    cfg = a.cfg if a.cfg is not None else (1.01 if a.cot == "off" else 1.0)
    with open(os.path.join(a.out, "request.json"), "w", encoding="utf-8") as fh:
        json.dump({"style": a.style, "lyrics": lyrics, "cot": a.cot, "seed": a.seed, "cfg": cfg, "abc": None, "abc_max": a.abc_max, "sem_max": a.sem_max, "steps": a.steps,
                   "weights": "q8", "reference": "comfy"}, fh, ensure_ascii=False, indent=1)
    instr = cy.INSTRUCTIONS[a.cot]
    text = f"{instr}\n[Tags]\n{a.style}\n[Lyrics]\n{lyrics}\n"
    with open(os.path.join(a.out, "text.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    first = [cy.EOD] + tok.encode(text) + [cy.ABC_START]
    # the protocol's first phase prompt: cot off closes the (absent) score at once
    prefix = first + [cy.ABC_END, cy.MUSIC_START] if a.cot == "off" else first
    ids(a.out, "prefix", prefix)
    print(f"ref: prefix {len(prefix)} tokens", file=sys.stderr, flush=True)
    if stages == ["tokens"]:
        return

    ar_sd, nar_sd = split_weights(tensors)
    dt = torch.float32
    ops = comfy.ops.manual_cast
    te = cy.YuE2TEModel(device="cpu", dtype=dt, model_options={"custom_operations": ops}, config={"fixed_kv": False})
    load_into(te.model, ar_sd, "the AR path")
    te.model.eval()
    abc_ids = []
    if "ar" in stages:
        if a.cot != "off":
            abc_ids, trunc = te._generate(first, a.seed, a.abc_max, "abc", dt, temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005, penalty_window=100,
                                          min_tokens=min(32, a.abc_max))
            print(f"ref: score {len(abc_ids)} tokens, truncated {trunc}", file=sys.stderr, flush=True)
        ids(a.out, "abc", abc_ids)
        sem_prefix = prefix if a.cot == "off" else first + abc_ids + [cy.ABC_END, cy.MUSIC_START]
        ids(a.out, "sem_prefix", sem_prefix)
        neg_head = [cy.EOD] + tok.encode(instr)
        neg = neg_head + ([cy.MUSIC_START] if a.cot == "off" else [cy.ABC_START] + abc_ids + [cy.ABC_END, cy.MUSIC_START])
        if cfg != 1.0:
            ids(a.out, "neg_prefix", neg)
        sem, trunc = te._generate(sem_prefix, a.seed, a.sem_max, "semantic", dt, negative=neg if cfg != 1.0 else None, cfg_scale=cfg, legacy_off=a.cot == "off", temperature=1.0,
                                  top_p=0.95, top_k=100, repetition_penalty=1.2, penalty_window=50, min_tokens=min(200, a.sem_max))
        codes = [t - cy.CODEC_OFFSET for t in sem]
        ids(a.out, "codes", codes)
        print(f"ref: {len(codes)} codes, truncated {trunc}", file=sys.stderr, flush=True)
        seq = sem_prefix + sem
        n_abc0 = len(first) - 1
        rows = sorted(set([len(first) - 1] + ([n_abc0 + i for i in range(0, max(1, len(abc_ids)), max(1, len(abc_ids) // 6))][:7] if abc_ids else [])
                          + [len(sem_prefix) - 1 + i for i in range(0, max(1, len(sem)), max(1, len(sem) // 8))][:9]))
        rows = [r for r in rows if r < len(seq)]
        x, _ = te.model(torch.tensor([seq]), dtype=dt)
        ids(a.out, "ar_rows", rows)
        save(a.out, "ar_logits", te.model.lm_head(x[0, rows]))
        if cfg != 1.0:
            xn, _ = te.model(torch.tensor([neg]), dtype=dt)
            save(a.out, "neg_logits", te.model.lm_head(xn[0, -1]))
    if "nar" in stages:
        sem_prefix = np.load(os.path.join(a.out, "sem_prefix.npy")).tolist()
        codes = np.load(os.path.join(a.out, "codes.npy")).tolist()
        frames = len(codes)
        cond, chunks = te._acoustic_conditioning(sem_prefix, [c + cy.CODEC_OFFSET for c in codes], dt)
        assert len(chunks) == 1
        m = YuE2(dtype=dt, device="cpu", operations=ops)
        load_into(m, nar_sd, "the NAR path")
        m.eval()
        torch.manual_seed(a.seed)
        noise = torch.randn((frames, 64), dtype=torch.float32)
        save(a.out, "noise", noise)
        def vel(state, raw):
            t = torch.sigmoid(torch.tensor([raw], dtype=torch.float32))
            return m(state.T[None], t, cond, chunks)[0].T
        steps = a.steps
        state = noise.clone()
        dts = 1.0 / steps
        states, mids = [state.clone()], []
        logit = lambda t: torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
        for step in range(steps):
            t = 1.0 - step * dts
            v = vel(state, logit(t))
            if step == 0:
                save(a.out, "vel0", v)
            mid = state - v * (dts / 2)
            mids.append(mid.clone())
            state = state - vel(mid, logit(t - dts / 2)) * dts
            states.append(state.clone())
            print(f"ref: step {step + 1}/{steps}", file=sys.stderr, flush=True)
        save(a.out, "states", torch.stack(states))
        save(a.out, "mids", torch.stack(mids))
        save(a.out, "latents", state)
    if "vae" in stages:
        _, vt = gguf(a.vae)
        v = AudioOobleckVAE(channels=64, c_mults=[1, 2, 4, 8, 16, 32], strides=[2, 2, 4, 4, 5, 6], sample_latent=False)
        for mod in v.decoder.modules():
            if torch.nn.utils.parametrize.is_parametrized(mod, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
        load_into(v, vae_weights(vt), "the VAE")
        v.eval()
        z = torch.from_numpy(np.load(os.path.join(a.out, "latents.npy"))).float().T[None]
        audio = v.decoder(z)[0].clamp(-1, 1).T
        n = 1920 * z.shape[-1] - 64
        print(f"ref: Comfy's decoder {audio.shape[0]} samples, kept {n}", file=sys.stderr, flush=True)
        save(a.out, "audio", audio[:n])


if __name__ == "__main__":
    main()
