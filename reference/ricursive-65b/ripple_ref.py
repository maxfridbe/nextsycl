#!/usr/bin/env python3
"""A hand-written float32 CPU reference of Ricursive Ripple-65B (written from ricursiveAI's modeling_ricursive.py,
configuration_ricursive.py and config.json; none of their code is run, and trust_remote_code is not involved), for
the first N blocks and the head, on one of two sets of weights:

  --st DIR    the BF16 weights fetched by fetch_tensors.py (the model itself: how far a quantized file is from it)
  --gguf FILE the weights of a GGUF file dequantized to float32 (Q4_K / Q5_K / Q6_K / Q8_0 / IQ4_XS / F32 / F16 / BF16,
              ggml's formulas): what an engine running that file must reproduce - the check of its kernels

    python3 -I ripple_ref.py (--st DIR | --gguf FILE) --layers N --ids IDS --out DIR

IDS: token ids, comma / space separated (a file). Writes the float32 tensors the engine dumps with NS_RC_DUMP=DIR
(embd [t, 5120], l_out-L [t, 5120] for every block, result_norm and logits of the last row) and compare.py compares.
With --layers 80 it is the whole model (slow and 130 GB of BF16 or the 40 GB file; only the shards a run needs).

Needs numpy only. The model (config.json): hidden 5120, 80 query / 8 key-value heads of 128, FFN 40960 clamped at 20,
RMS eps 1e-6, rotary theta 1e4 on the sliding-window layers only (window 1024, a query sees the 1024 keys ending at
itself), every fifth layer (4, 9, ...) global with no rotary positions, embeddings scaled by sqrt(5120).
"""
import argparse
import json
import mmap
import os
import struct
import sys

import numpy as np

HID, NH, NKV, HD, FF = 5120, 80, 8, 128, 40960
EPS, THETA, WINDOW, CLAMP = 1e-6, 10000.0, 1024, 20.0
EMBED_SCALE = float(np.float32(5120 ** 0.5))
IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def swa_layer(l):
    return l % 5 != 4


# ---- weights -----------------------------------------------------------------------------------------------------

class SafetensorsWeights:
    """fetch_tensors.py's DIR: BF16 as raw files"""

    def __init__(self, d):
        self.d = d
        self.meta = json.load(open(os.path.join(d, "index.json")))

    def get(self, name, rows=None):
        raw = np.fromfile(os.path.join(self.d, name + ".bf16"), dtype=np.uint16)
        a = (raw.astype(np.uint32) << 16).view(np.float32).reshape(self.meta[name])
        return a if rows is None else a[rows]

    def has(self, name):
        return name in self.meta


GGUF_TO_HF = {
    "attn_norm": "input_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "attn_gate": "self_attn.g_proj", "attn_vres_lambda": "self_attn.value_res_lambda", "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    "post_ffw_norm": "post_feedforward_layernorm",
}
HF_TO_GGUF = {v: k for k, v in GGUF_TO_HF.items()}

GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
        14: ("Q6_K", 256, 210), 23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2)}


def k4_scale_min(sc):
    """get_scale_min_k4 for the 8 sub-blocks of every Q4_K / Q5_K block: sc [n, 12] uint8 -> (scales, mins) [n, 8]"""
    s = np.empty((sc.shape[0], 8), np.float32)
    m = np.empty((sc.shape[0], 8), np.float32)
    s[:, :4] = sc[:, 0:4] & 63
    m[:, :4] = sc[:, 4:8] & 63
    s[:, 4:] = (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4)
    m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return s, m


def f16(b):
    return b.copy().view(np.float16).astype(np.float32)


def dequant(ty, raw, n):
    """n values of a ggml type from its blocks (raw: uint8)"""
    name, qk, bs = GGML[ty]
    if name == "F32":
        return raw.view(np.float32)[:n].copy()
    if name == "F16":
        return raw.view(np.float16)[:n].astype(np.float32)
    if name == "BF16":
        return (raw.view(np.uint16)[:n].astype(np.uint32) << 16).view(np.float32)
    nb = n // qk
    b = raw[: nb * bs].reshape(nb, bs)
    if name == "Q8_0":
        d = f16(np.ascontiguousarray(b[:, 0:2]))[:, 0]
        return (d[:, None] * b[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)
    if name in ("Q4_K", "Q5_K"):
        d = f16(np.ascontiguousarray(b[:, 0:2]))[:, 0]
        dmin = f16(np.ascontiguousarray(b[:, 2:4]))[:, 0]
        s, m = k4_scale_min(b[:, 4:16])
        out = np.empty((nb, 256), np.float32)
        if name == "Q4_K":
            qs = b[:, 16:144].reshape(nb, 4, 32)
            lo, hi = (qs & 0xF).astype(np.float32), (qs >> 4).astype(np.float32)
        else:
            qh = b[:, 16:48]
            qs = b[:, 48:176].reshape(nb, 4, 32)
            lo, hi = (qs & 0xF).astype(np.float32), (qs >> 4).astype(np.float32)
            for j in range(4):
                lo[:, j] += ((qh >> (2 * j)) & 1) * 16.0
                hi[:, j] += ((qh >> (2 * j + 1)) & 1) * 16.0
        for j in range(4):
            out[:, 64 * j:64 * j + 32] = d[:, None] * s[:, 2 * j:2 * j + 1] * lo[:, j] - dmin[:, None] * m[:, 2 * j:2 * j + 1]
            out[:, 64 * j + 32:64 * j + 64] = d[:, None] * s[:, 2 * j + 1:2 * j + 2] * hi[:, j] - dmin[:, None] * m[:, 2 * j + 1:2 * j + 2]
        return out.reshape(-1)
    if name == "Q6_K":
        ql, qh = b[:, 0:128], b[:, 128:192]
        sc = b[:, 192:208].view(np.int8).astype(np.float32)
        d = f16(np.ascontiguousarray(b[:, 208:210]))[:, 0]
        out = np.empty((nb, 256), np.float32)
        for h in range(2):
            l_ = ql[:, 64 * h:64 * h + 64]
            h_ = qh[:, 32 * h:32 * h + 32]
            q1 = ((l_[:, :32] & 0xF) | (((h_ >> 0) & 3) << 4)).astype(np.int32) - 32
            q2 = ((l_[:, 32:] & 0xF) | (((h_ >> 2) & 3) << 4)).astype(np.int32) - 32
            q3 = ((l_[:, :32] >> 4) | (((h_ >> 4) & 3) << 4)).astype(np.int32) - 32
            q4 = ((l_[:, 32:] >> 4) | (((h_ >> 6) & 3) << 4)).astype(np.int32) - 32
            s = sc[:, 8 * h:8 * h + 8]
            isx = np.arange(32) // 16
            for k, q in enumerate((q1, q2, q3, q4)):
                out[:, 128 * h + 32 * k:128 * h + 32 * k + 32] = d[:, None] * s[:, 2 * k + isx] * q
        return out.reshape(-1)
    if name == "IQ4_XS":
        d = f16(np.ascontiguousarray(b[:, 0:2]))[:, 0]
        sh = np.ascontiguousarray(b[:, 2:4]).view(np.uint16)[:, 0].astype(np.int32)
        sl = b[:, 4:8].astype(np.int32)
        qs = b[:, 8:136].reshape(nb, 8, 16)
        out = np.empty((nb, 8, 32), np.float32)
        for ib in range(8):
            ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32).astype(np.float32)
            out[:, ib, :16] = dl[:, None] * IQ4NL[qs[:, ib] & 0xF]
            out[:, ib, 16:] = dl[:, None] * IQ4NL[qs[:, ib] >> 4]
        return out.reshape(-1)
    raise SystemExit("unsupported ggml type %d" % ty)


class GgufWeights:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.t = {}
        self._parse()

    def _parse(self):
        mm = self.mm
        pos = [0]

        def rd(fmt):
            sz = struct.calcsize("<" + fmt)
            v = struct.unpack_from("<" + fmt, mm, pos[0])
            pos[0] += sz
            return v

        def rs():
            n, = rd("Q")
            s = mm[pos[0]:pos[0] + n].decode("utf-8", "replace")
            pos[0] += n
            return s

        def rv(t):
            if t == 8:
                return rs()
            if t == 9:
                et, n = rd("IQ")
                return [rv(et) for _ in range(n)]
            fm = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
            return rd(fm)[0]

        assert mm[:4] == b"GGUF"
        pos[0] = 4
        _, nt, nkv = rd("IQQ")
        self.meta = {}
        for _ in range(nkv):
            k = rs()
            t, = rd("I")
            self.meta[k] = rv(t)
        infos = []
        for _ in range(nt):
            name = rs()
            nd, = rd("I")
            dims = rd("Q" * nd)
            ty, off = rd("IQ")
            infos.append((name, dims, ty, off))
        align = self.meta.get("general.alignment", 32)
        data = (pos[0] + align - 1) // align * align
        for name, dims, ty, off in infos:
            self.t[name] = (dims, ty, data + off)

    def has(self, name):
        return name in self.t

    def get(self, name, rows=None):
        dims, ty, off = self.t[name]
        n = int(np.prod(dims))
        shape = tuple(reversed(dims))
        qk, bs = GGML[ty][1], GGML[ty][2]
        if rows is not None:
            # rows of a 2-D tensor: only their blocks
            per = shape[1] // qk * bs
            out = [dequant(ty, np.frombuffer(self.mm, np.uint8, per, off + int(r) * per), shape[1]) for r in rows]
            return np.stack(out)
        raw = np.frombuffer(self.mm, np.uint8, n // qk * bs, off)
        return dequant(ty, raw, n).reshape(shape)


class Weights:
    """the model's tensors by HF name, from either backend"""

    def __init__(self, st=None, gguf=None):
        self.st, self.gg = st, gguf

    def get(self, hf, rows=None):
        if self.st:
            return self.st.get(hf, rows)
        if hf == "model.embed_tokens.weight":
            g = "token_embd.weight"
        elif hf == "model.norm.weight":
            g = "output_norm.weight"
        elif hf == "lm_head.weight":
            g = "output.weight"
        else:
            p = hf.split(".")
            rest = ".".join(p[3:])
            if rest.endswith(".weight"):
                rest = rest[:-len(".weight")]
            g = "blk.%s.%s.weight" % (p[2], HF_TO_GGUF[rest])
        return self.gg.get(g, rows)


# ---- the model ---------------------------------------------------------------------------------------------------

def rms(x, w):
    return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)).astype(np.float32)) * w


def rope(x, pos):
    """x [t, heads, 128]; rotate_half, theta 1e4 over all 128 features"""
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    ang = pos[:, None].astype(np.float32) * inv[None, :]
    emb = np.concatenate([ang, ang], -1)
    c, s = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * c + np.concatenate([-x2, x1], -1) * s


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def block(w, l, x, x0, pos):
    p = "model.layers.%d." % l
    t = x.shape[0]
    h = rms(x, w.get(p + "input_layernorm.weight"))
    q = (h @ w.get(p + "self_attn.q_proj.weight").T).reshape(t, NH, HD)
    k = (h @ w.get(p + "self_attn.k_proj.weight").T).reshape(t, NKV, HD)
    lam = float(w.get(p + "self_attn.value_res_lambda").reshape(-1)[0])
    blend = np.float32(1.0 / (1.0 + np.exp(-lam)))
    vin = x0 + blend * (h - x0)          # torch.lerp(value_residual, hidden, blend)
    v = (vin @ w.get(p + "self_attn.v_proj.weight").T).reshape(t, NKV, HD)
    q = rms(q, w.get(p + "self_attn.q_norm.weight"))
    k = rms(k, w.get(p + "self_attn.k_norm.weight"))
    swa = swa_layer(l)
    if swa:
        q, k = rope(q, pos), rope(k, pos)
    qh = q.transpose(1, 0, 2)                            # [NH, t, HD]
    kh = np.repeat(k.transpose(1, 0, 2), NH // NKV, 0)
    vh = np.repeat(v.transpose(1, 0, 2), NH // NKV, 0)
    sc = np.einsum("hid,hjd->hij", qh, kh) / np.float32(HD ** 0.5)
    i, j = pos[:, None], pos[None, :]
    ok = j <= i
    if swa:
        ok &= j >= i - (WINDOW - 1)
    sc = np.where(ok[None], sc, -np.inf)
    sc = sc - sc.max(-1, keepdims=True)
    pr = np.exp(sc)
    pr /= pr.sum(-1, keepdims=True)
    att = np.einsum("hij,hjd->hid", pr, vh).transpose(1, 0, 2).reshape(t, NH * HD)
    gate = sigmoid(h @ w.get(p + "self_attn.g_proj.weight").T)
    o = (att * gate) @ w.get(p + "self_attn.o_proj.weight").T
    x = x + rms(o, w.get(p + "post_attention_layernorm.weight"))
    h = rms(x, w.get(p + "pre_feedforward_layernorm.weight"))
    g = h @ w.get(p + "mlp.gate_proj.weight").T
    u = h @ w.get(p + "mlp.up_proj.weight").T
    g = np.minimum(g, CLAMP)
    u = np.clip(u, -CLAMP, CLAMP)
    ff = (silu(g) * u) @ w.get(p + "mlp.down_proj.weight").T
    return x + rms(ff, w.get(p + "post_feedforward_layernorm.weight"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st")
    ap.add_argument("--gguf")
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-head", action="store_true")
    a = ap.parse_args()
    if bool(a.st) == bool(a.gguf):
        raise SystemExit("exactly one of --st DIR and --gguf FILE")
    w = Weights(SafetensorsWeights(a.st) if a.st else None, GgufWeights(a.gguf) if a.gguf else None)
    ids = [int(v) for v in open(a.ids).read().replace(",", " ").split()]
    os.makedirs(a.out, exist_ok=True)
    pos = np.arange(len(ids))
    x = w.get("model.embed_tokens.weight", rows=ids).astype(np.float32) * np.float32(EMBED_SCALE)
    x0 = x.copy()
    x.astype(np.float32).tofile(os.path.join(a.out, "embd.f32"))
    for l in range(a.layers):
        x = block(w, l, x, x0, pos).astype(np.float32)
        x.tofile(os.path.join(a.out, "l_out-%d.f32" % l))
        print("layer", l, "done", flush=True)
    if not a.no_head:
        n = rms(x[-1:], w.get("model.norm.weight"))
        n.astype(np.float32).tofile(os.path.join(a.out, "result_norm.f32"))
        (n @ w.get("lm_head.weight").T).astype(np.float32).tofile(os.path.join(a.out, "logits.f32"))


if __name__ == "__main__":
    main()
