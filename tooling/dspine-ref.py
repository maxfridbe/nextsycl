#!/usr/bin/env python3
"""A float32 numpy reference for the DSpine draft (llm/qwen35/src/dspine.rs), checked against the engine's own run.

The engine, run with NS_Q35_DSPINE_DUMP=DIR, writes the target's aux states of every pass (aux-<p0>-<t>.f32, [5][t][5120]
float32) and, for each draft, draft-<pos>.bin: the block, its embeddings, the final rows, the last layer's raw / rms /
packed, the head's logits, the candidates' head rows (half), the candidates, the refinement's picks and the drafts. This
redoes the draft from the aux states and the checkpoint's weights (CPU, float32, bfloat16 weights widened) and prints how
close the engine's numbers are:

  python3 -I dspine-ref.py CHECKPOINT_DIR DUMP_DIR [draft-<pos>.bin ...]

Untrusted input: the checkpoint is Tencent's, read as data only (no code from it runs). Needs numpy only."""
import glob
import json
import math
import os
import struct
import sys

import numpy as np

EPS = 1e-6


class Ckpt:
    def __init__(self, d):
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        f = open(os.path.join(d, "model.safetensors"), "rb")
        n, = struct.unpack("<Q", f.read(8))
        self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(os.path.join(d, "model.safetensors"), mode="r", dtype=np.uint8)

    def t(self, name):
        i = self.h[name]
        a, b = i["data_offsets"]
        raw = np.asarray(self.mm[self.base + a:self.base + b])
        if i["dtype"] == "BF16":
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(i["shape"])
        if i["dtype"] == "I64":
            return raw.view(np.int64).reshape(i["shape"])
        raise ValueError(i["dtype"])

    def rows(self, name, ids):
        i = self.h[name]
        a, _ = i["data_offsets"]
        cols = i["shape"][1]
        out = np.empty((len(ids), cols), np.float32)
        for k, r in enumerate(ids):
            raw = np.asarray(self.mm[self.base + a + r * cols * 2:self.base + a + (r + 1) * cols * 2])
            out[k] = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return out


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def rope(x, pos, theta):
    d = x.shape[-1]
    inv = theta ** (-2.0 * np.arange(d // 2) / d)
    a = pos[:, None] * inv[None, :]
    c, s = np.cos(a)[:, None, :], np.sin(a)[:, None, :]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ck_dir, dump = sys.argv[1], sys.argv[2]
    ck = Ckpt(ck_dir)
    cfg = ck.cfg
    H, L, HQ, HKV, D = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    theta = float(cfg["rope_parameters"]["rope_theta"])
    window = cfg.get("sliding_window") or 0
    R, C = 512, 1024
    b = "selected_prefix_bridge."
    # the context: every pass's aux rows by position
    passes = []
    for p in glob.glob(os.path.join(dump, "aux-*-*.f32")):
        _, p0, t = os.path.basename(p)[:-4].split("-")
        passes.append((int(p0), int(t), p))
    passes.sort()
    # a verify pass's rows past what was kept are written again by the next pass (from the kept position): the later
    # pass (a greater p0) wins
    feats = np.zeros((max(p0 + t for p0, t, _ in passes), 5 * H), np.float32)
    for p0, t, p in passes:
        feats[p0:p0 + t] = np.fromfile(p, np.float32).reshape(5, t, H).transpose(1, 0, 2).reshape(t, 5 * H)
    print(f"context: {len(feats)} positions from {len(passes)} passes")
    ctx_all = rms(feats @ ck.t("fc.weight").T, ck.t("hidden_norm.weight"))
    files = sys.argv[3:] or sorted(glob.glob(os.path.join(dump, "draft-*.bin")))
    wnorm = ck.t("norm.weight")
    for fn in files:
        raw = open(fn, "rb").read()
        pos, tail, t, K = struct.unpack("<4I", raw[:16])
        o = 16
        tokens = np.frombuffer(raw, np.uint32, t, o)
        o += 4 * t

        def take(dt, n):
            nonlocal o
            a = np.frombuffer(raw, dt, n, o)
            o += a.nbytes
            return a
        n = t - 1
        vocab = cfg["vocab_size"]
        x0 = take(np.float32, t * H).reshape(t, H)
        e_normed = take(np.float32, t * H).reshape(t, H)
        e_raw = take(np.float32, t * H).reshape(t, H)
        e_rms = take(np.float32, t)
        e_packed = take(np.float32, t * 3 * R).reshape(t, 3 * R)
        e_logits = take(np.float32, n * vocab).reshape(n, vocab)
        rows16 = take(np.float16, n * K * H).reshape(n, K, H).astype(np.float32)
        cand = take(np.int32, n * K).reshape(n, K)
        pick = take(np.int32, n * K).reshape(n, K)
        drafts = take(np.uint32, n)
        ctx = ctx_all[:pos]
        assert pos == len(ctx) and (window == 0 or pos + t <= window), "this reference does not do the window"
        cb_anchor = np.sqrt(C) * ck.rows(b + "codebook", [int(tail)])[0]
        x = x0.copy()
        sqrt_c = math.sqrt(C)
        for l in range(L):
            p = f"layers.{l}."
            ln = rms(x, ck.t(p + "input_layernorm.weight"))
            q = rms((ln @ ck.t(p + "self_attn.q_proj.weight").T).reshape(t, HQ, D), ck.t(p + "self_attn.q_norm.weight"))
            kn = ck.t(p + "self_attn.k_norm.weight")
            kp, vp = ck.t(p + "self_attn.k_proj.weight"), ck.t(p + "self_attn.v_proj.weight")
            k = rms((ln @ kp.T).reshape(t, HKV, D), kn)
            v = (ln @ vp.T).reshape(t, HKV, D)
            kc = rms((ctx @ kp.T).reshape(-1, HKV, D), kn)
            vc = (ctx @ vp.T).reshape(-1, HKV, D)
            kc = rope(kc, np.arange(pos, dtype=np.float32), theta)
            q = rope(q, np.arange(pos, pos + t, dtype=np.float32), theta)
            k = rope(k, np.arange(pos, pos + t, dtype=np.float32), theta)
            K_all, V_all = np.concatenate([kc, k]), np.concatenate([vc, v])
            att = np.zeros((t, HQ, D), np.float32)
            mask = np.zeros((t, pos + t), bool)
            for i in range(t):
                mask[i, :pos + i + 1] = True
            for h in range(HQ):
                s = np.einsum("td,sd->ts", q[:, h], K_all[:, h // (HQ // HKV)]) / math.sqrt(D)
                s = np.where(mask, s, -1e30)
                s = np.exp(s - s.max(-1, keepdims=True))
                s /= s.sum(-1, keepdims=True)
                att[:, h] = s @ V_all[:, h // (HQ // HKV)]
            r = x + att.reshape(t, HQ * D) @ ck.t(p + "self_attn.o_proj.weight").T
            hn = rms(r, ck.t(p + "post_attention_layernorm.weight"))
            g = hn @ ck.t(p + "mlp.gate_proj.weight").T
            u = hn @ ck.t(p + "mlp.up_proj.weight").T
            raw_l = r + (g / (1 + np.exp(-g)) * u) @ ck.t(p + "mlp.down_proj.weight").T
            rm = np.sqrt(np.maximum((raw_l * raw_l).mean(-1), 1.1754943508222875e-38))
            nrm = raw_l / (rm[:, None] + EPS)
            rw = ck.t(b + ("read_weight" if l == L - 1 else "early_read_weights"))
            rw = rw if l == L - 1 else rw[l]
            proj = nrm @ rw.T
            codes = sqrt_c * proj / (np.linalg.norm(proj, axis=-1, keepdims=True) + EPS)
            codes[0] = cb_anchor
            pw = np.concatenate([ck.t(b + "gate.weight"), ck.t(b + "predecessor_gate.weight"), ck.t(b + "prev.weight")])
            packed = codes @ pw.T
            qq, ss, vv = packed[:, :R], packed[:, R:2 * R], packed[:, 2 * R:]
            msg = np.zeros((t, R), np.float32)
            msg[1:] = sigmoid(qq[1:] + ss[:-1] + ck.t(b + "gate.bias")) * vv[:-1]
            delta = msg @ ck.t(b + "feature_write_weights")[l].T
            out = raw_l.copy()
            out[1:] += rm[1:, None] * delta[1:]
            x = out
        normed = rms(x, wnorm)
        print(f"== {os.path.basename(fn)}: position {pos}, anchor {int(tail)}, block {t}")
        print(f"  final rows (cosine, rows 1..): {[round(cos(normed[i], e_normed[i]), 5) for i in range(1, t)]}")
        print(f"  raw (cosine): {cos(raw_l, e_raw):.6f}  rms rel. err {np.abs(rm - e_rms).max() / rm.max():.2e}  packed cosine {cos(packed, e_packed):.6f}")
        top = np.argsort(-e_logits, axis=-1)[:, :K]
        same = [set(top[i].tolist()) == set(cand[i].tolist()) for i in range(n)]
        print(f"  candidates are the logits' top {K}: {same}")
        # the refinement, from the engine's candidates and their head rows
        pwr = np.concatenate([ck.t(b + "predecessor_gate.weight"), ck.t(b + "prev.weight")])
        wl = ck.t(b + "feature_write_weights")[L - 1]
        bias = ck.t(b + "gate.bias")
        idx = np.zeros((n, K), np.int32)
        for i in range(n):
            base = (e_raw[i + 1] * wnorm) @ rows16[i].T
            tg = (rows16[i] * wnorm) @ wl
            for kk in range(K):
                sid = int(tail) if i == 0 else int(cand[i - 1, kk])
                tsv = (sqrt_c * ck.rows(b + "codebook", [sid])[0]) @ pwr.T
                m = sigmoid(e_packed[i + 1, :R] + tsv[:R] + bias) * tsv[R:]
                idx[i, kk] = int(np.argmax(base + e_rms[i + 1] * (tg @ m)))
        print(f"  refinement picks the same as the engine's: {int((idx == pick).sum())} of {idx.size}")
        j, chain = 0, []
        for i in range(n):
            p_ = idx[i, 0] if i == 0 else idx[i, j]
            chain.append(int(cand[i, p_]))
            j = p_
        print(f"  drafts: engine {drafts.tolist()}  reference {chain}  {'SAME' if chain == drafts.tolist() else 'DIFFER'}")


if __name__ == "__main__":
    main()
