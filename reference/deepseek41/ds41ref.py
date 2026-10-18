"""DeepSeek-V4.1-Flash's forward on the CPU (PyTorch, float32), from DeepSeek's inference/model.py (1017, rev 1bae0e1)
with this GGUF's weights dequantized (gguf41.py): a whole prompt from position 0, one layer at a time, so the 106 GiB
model never has to fit (each layer's attention weights, then only the experts its tokens pick, are expanded and
dropped). What the model's own code simulates in low precision is kept (the window keys fp8, the compressed keys fp4 with
E4M3 scales, the indexer's queries and keys fp4, the attention output fp8 before wo_a); the activations are float32,
not bf16 - the engine's are too.

  python ds41ref.py MODEL.gguf ENGRAM_DIR OUT_DIR --tokens "1 2 3" [--layers 40] [--teacher DIR] [--mtp]

writes OUT_DIR/tokens.txt, next.txt, index.tsv and NAME.f32 per tap, in the layout `nextsycl llm check` reads (the
engine's tap names: l_in-L, hc_engram-L, attn_norm-L, attn_out-L, ffn_norm-L, ffn_out-L, l_out-L, result_norm,
result_output, ...). --teacher DIR: each layer's input read from the engine's saved taps (NS_CHECK_SAVE=DIR) instead of
the reference's own previous layer - a layer's error alone."""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf41 import Gguf, VqBlob, Engram, engram_hash, bf16_round  # noqa: E402

torch.set_num_threads(int(os.environ.get('NS_REF_THREADS', '24')))
F = torch.float32


# ---- the model's low-precision simulations (kernel.py's act_quant / fp4_act_quant with inplace=True)

def pow2_ceil(x):
    """2^ceil(log2 x) for x > 0 (fast_round_scale)."""
    m, e = torch.frexp(x)
    e = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(x), e)


QSIM = os.environ.get('NS_DS41_QSIM', '1') != '0'


def fp8_sim(x, block=32):
    """E4M3 with a power-of-two scale a block (amax >= 1e-4), quantized and back."""
    if not QSIM:
        return x
    sh = x.shape
    v = x.reshape(-1, block)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax / 448.0)
    q = (v / s).clamp(-448, 448).to(torch.float8_e4m3fn).to(F)
    return (q * s).reshape(sh)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_round(y):
    """|y| <= 6 to the nearest E2M1 value, ties to the even code."""
    a = y.abs()
    d = (a.unsqueeze(-1) - E2M1).abs()
    best = d.min(-1, keepdim=True).values
    cand = (d == best)
    code = torch.where(cand[..., 0::2].any(-1), (cand[..., 0::2].int().argmax(-1)) * 2, cand.int().argmax(-1))
    return torch.sign(y) * E2M1[code]


def fp4_sim(x, block, e4m3_scale):
    """E2M1 a value with a scale a block: a power of two (amax >= 6 x 2^-126, the indexer's), or an E4M3 amax / 6
    (amax >= 6 x 2^-9, the compressed keys')."""
    if not QSIM:
        return x
    sh = x.shape
    v = x.reshape(-1, block)
    amax = v.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s = (amax.clamp_min(6 * 2.0 ** -9) / 6.0).to(torch.float8_e4m3fn).to(F)
    else:
        s = pow2_ceil(amax.clamp_min(6 * 2.0 ** -126) / 6.0)
    return (e2m1_round((v / s).clamp(-6, 6)) * s).reshape(sh)


# ---- pieces

def rms(x, w, eps):
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def freqs(dim, n, orig, base, factor, beta_fast, beta_slow):
    f = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    if orig > 0:
        def cd(r):
            return dim * math.log(orig / (r * 2 * math.pi)) / (2 * math.log(base))
        lo, hi = max(math.floor(cd(beta_fast)), 0), min(math.ceil(cd(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float64) - lo) / max(hi - lo, 1e-3)).clamp(0, 1)
        f = f / factor * ramp + f * (1 - ramp)
    ang = torch.outer(torch.arange(n, dtype=torch.float64), f)
    return torch.cos(ang).to(F), torch.sin(ang).to(F)


def rope(x, cs, pos, inverse=False):
    """x [..., rd] (adjacent pairs), rotated at positions pos (same leading dim as x's first)."""
    c, s = cs[0][pos], cs[1][pos]
    while c.dim() < x.dim():
        c, s = c.unsqueeze(-2), s.unsqueeze(-2)
    if inverse:
        s = -s
    a, b = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def sinkhorn(mixes, scale, base, hc, iters, eps):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).reshape(*mixes.shape[:-1], hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class Ref:
    def __init__(self, path, engram_dir):
        self.g = g = Gguf(path)
        a = g.arch
        self.D, self.L = a('embedding_length'), a('block_count')
        self.H, self.HD, self.RD = a('attention.head_count'), a('attention.key_length'), a('rope.dimension_count')
        self.QL, self.OG, self.OL = a('attention.q_lora_rank'), a('attention.output_group_count'), a('attention.output_lora_rank')
        self.WIN = a('attention.sliding_window')
        self.IH, self.ID, self.ITOP = a('attention.indexer.head_count'), a('attention.indexer.key_length'), a('attention.indexer.top_k')
        self.ratio = a('attention.compress_ratios')
        self.kv_src = a('attention.kv_source_layers')
        self.idx_src = a('attention.index_source_layers')
        self.cand_src, self.cand_top, self.cand_bs = a('attention.candidate.source_layer'), a('attention.candidate.topk_blocks'), a('attention.candidate.block_size')
        self.eps = a('attention.layer_norm_rms_epsilon')
        self.hc_eps, self.hc_iters, self.HC = a('hyper_connection.epsilon'), a('hyper_connection.sinkhorn_iterations'), a('hyper_connection.count')
        self.NE, self.NU, self.scale = a('expert_count'), a('expert_used_count'), a('expert_weights_scale')
        self.lim = 10.0
        self.rope_base, self.comp_base = a('rope.freq_base'), a('attention.compress_rope_freq_base')
        self.yarn = (a('rope.scaling.original_context_length'), a('rope.scaling.factor'), a('rope.scaling.yarn_beta_fast'), a('rope.scaling.yarn_beta_slow'))
        self.engram = Engram(g, engram_dir) if engram_dir else None
        self.eng_layers = a('engram.layer_ids')
        self.mtp_targets = a('mtp.target_layers')

    def name(self, l, r):
        return f'blk.{l}.{r}' if l < self.L else f'mtp.{l - self.L}.{r}'

    def w(self, l, r):
        return torch.from_numpy(np.ascontiguousarray(self.g.get(self.name(l, r)), np.float32))

    def has(self, l, r):
        return self.g.has(self.name(l, r))

    def cs(self, l, n):
        o, fac, bf, bs = self.yarn
        if self.ratio[l]:
            return freqs(self.RD, n, o, self.comp_base, fac, bf, bs)
        return freqs(self.RD, n, 0, self.rope_base, fac, bf, bs)

    def src_of(self, l, srcs):
        s = [x for x in srcs if x <= l]
        return s[-1] if s else None

    # ---- attention (a whole prompt from position 0)
    def attention(self, l, x, sh, tap, qrows=None):
        """x [T, D] normed (every row: the keys); qrows: the query rows computed (all by default); sh: what source
        layers hand down (comp, idx_k, topk, cand)"""
        T = x.shape[0]
        if qrows is None:
            qrows = torch.arange(T)
        R = len(qrows)
        ratio = self.ratio[l]
        cs = self.cs(l, T + 1)
        pos = torch.arange(T)
        qr = rms(x[qrows] @ self.w(l, 'attn_q_a.weight').T, self.w(l, 'attn_q_a_norm.weight'), self.eps)
        q = (qr @ self.w(l, 'attn_q_b.weight').T).reshape(R, self.H, self.HD)
        q[..., -self.RD:] = rope(q[..., -self.RD:], cs, qrows)
        tap(f'q-{l}', q)
        kv = rms(x @ self.w(l, 'attn_kv.weight').T, self.w(l, 'attn_kv_a_norm.weight'), self.eps)
        kv[:, -self.RD:] = rope(kv[:, -self.RD:], cs, pos)
        kv = fp8_sim(kv)
        tap(f'kv-{l}', kv)
        # the keys each row reads: its window, then the compressed ones it is given
        keys = [kv]
        win_idx = qrows[:, None] - torch.arange(self.WIN)[None, :]
        mask = [(win_idx >= 0)]
        sel = [win_idx.clamp_min(0)]
        if ratio:
            latent = None
            if l in self.kv_src:
                if ratio == 1:
                    lat = rms(x @ self.w(l, 'attn_compressor_kv.weight').T, self.w(l, 'attn_compressor_norm.weight'), self.eps)
                else:
                    n = T // ratio
                    kvc = (x[:n * ratio] @ self.w(l, 'attn_compressor_kv.weight').T).reshape(n, ratio, -1)
                    sc = (x[:n * ratio] @ self.w(l, 'attn_compressor_gate.weight').T).reshape(n, ratio, -1)
                    lat = rms((kvc * sc.softmax(1)).sum(1), self.w(l, 'attn_compressor_norm.weight'), self.eps)
                gpos = torch.arange(lat.shape[0]) * ratio
                ik = rms(lat @ self.w(l, 'indexer.wk.weight').T, self.w(l, 'indexer.k_norm.weight'), self.eps)
                ik[:, -self.RD:] = rope(ik[:, -self.RD:], cs, gpos)
                sh['idx_k'] = fp4_sim(ik, 32, False)
                ck = lat.clone()
                ck[:, -self.RD:] = rope(ck[:, -self.RD:], cs, gpos)
                sh['comp'] = fp4_sim(ck, 16, True)
                tap(f'comp-{l}', sh['comp'])
            comp = sh['comp']
            ncomp = comp.shape[0]
            lens = (qrows + 1) // ratio
            if l in self.idx_src and ncomp > 0:
                iq = (qr @ self.w(l, 'indexer.attn_q_b.weight').T).reshape(R, self.IH, self.ID)
                iq[..., -self.RD:] = rope(iq[..., -self.RD:], cs, qrows)
                iq = fp4_sim(iq, 32, False)
                wt = (x[qrows] @ self.w(l, 'indexer.proj.weight').T) * (self.ID ** -0.5 * self.IH ** -0.5)
                score = (torch.einsum('thd,jd->thj', iq, sh['idx_k']).relu() * wt[..., None]).sum(1)
                score = score.masked_fill(torch.arange(ncomp)[None, :] >= lens[:, None], float('-inf'))
                if l == self.cand_src:
                    nb = (ncomp + self.cand_bs - 1) // self.cand_bs
                    bsc = torch.nn.functional.pad(score, (0, nb * self.cand_bs - ncomp), value=float('-inf')).reshape(R, nb, self.cand_bs).amax(-1)
                    last = (lens - 1).clamp_min(0) // self.cand_bs
                    bsc = bsc.masked_fill(torch.arange(nb)[None, :] == last[:, None], float('inf'))
                    top = bsc.topk(min(self.cand_top, nb), -1)
                    keep = torch.zeros_like(bsc, dtype=torch.bool).scatter_(-1, top.indices, top.values > float('-inf'))
                    sh['cand'] = keep.repeat_interleave(self.cand_bs, -1)[:, :ncomp]
                elif self.cand_src is not None and l > self.cand_src and 'cand' in sh:
                    score = score.masked_fill(~sh['cand'], float('-inf'))
                k = min(self.ITOP, ncomp)
                idx = score.topk(k, -1).indices.sort(-1).values
                sh['topk'] = torch.where(idx < lens[:, None], idx, -1)
                tap(f'indexer_topk-{l}', sh['topk'].float())
            elif l in self.idx_src:
                sh['topk'] = torch.full((R, 0), -1, dtype=torch.long)
            ti = sh['topk']
            keys.append(comp)
            mask.append(ti >= 0)
            sel.append(ti.clamp_min(0) + T)
        allk = torch.cat(keys, 0)
        si, mk = torch.cat(sel, 1), torch.cat(mask, 1)
        kk = allk[si]                                          # [T, n, HD]
        s = torch.einsum('thd,tnd->thn', q, kk) * self.HD ** -0.5
        s = s.masked_fill(~mk[:, None, :], float('-inf'))
        sink = self.w(l, 'attn_sinks.weight')
        m = torch.maximum(s.amax(-1), sink[None, :])
        p = torch.exp(s - m[..., None])
        den = p.sum(-1) + torch.exp(sink[None, :] - m)
        o = torch.einsum('thn,tnd->thd', p, kk) / den[..., None]
        o[..., -self.RD:] = rope(o[..., -self.RD:], cs, qrows, inverse=True)
        tap(f'attn_derope-{l}', o)
        o = fp8_sim(o.reshape(R, self.OG, -1))
        wo_a = self.w(l, 'attn_output_a.weight').reshape(self.OG, self.OL, -1)
        oa = torch.einsum('tgd,grd->tgr', o, wo_a).reshape(R, -1)
        out = oa @ self.w(l, 'attn_output_b.weight').T
        return out

    # ---- MoE
    def moe(self, l, x, tap, blobs):
        T = x.shape[0]
        wg = self.w(l, 'ffn_gate_inp.weight')
        ne = wg.shape[0]
        nu = self.NU if l < self.L else self.g.arch('mtp.expert_used_count')
        logits = x @ wg.T
        tap(f'ffn_moe_logits-{l}', logits)
        sc = torch.nn.functional.softplus(logits).sqrt()
        bias = self.w(l, 'exp_probs_b.bias')
        ids = (sc + bias).topk(nu, -1).indices
        wts = sc.gather(1, ids)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * self.scale
        sg = x @ self.w(l, 'ffn_gate_shexp.weight').T
        su = x @ self.w(l, 'ffn_up_shexp.weight').T
        y = (torch.nn.functional.silu(sg.clamp(max=self.lim)) * su.clamp(-self.lim, self.lim)) @ self.w(l, 'ffn_down_shexp.weight').T
        tap(f'ffn_shexp-{l}', y)
        blob = blobs(l)
        cb = blob.codebook()
        for e in sorted(set(ids.flatten().tolist())):
            rows, slot = torch.where(ids == e)
            gate = torch.from_numpy(blob.matrix(e, 0, cb))
            up = torch.from_numpy(blob.matrix(e, 1, cb))
            down = torch.from_numpy(blob.matrix(e, 2, cb))
            xe = x[rows]
            h = torch.nn.functional.silu((xe @ gate.T).clamp(max=self.lim)) * (xe @ up.T).clamp(-self.lim, self.lim)
            y.index_add_(0, rows, (h * wts[rows, slot][:, None]) @ down.T)
        return y, ids

    # ---- the engram (before layer l)
    def engram_apply(self, l, h, hashes, tap):
        li = self.eng_layers.index(l)
        T = h.shape[0]
        rows = torch.from_numpy(self.engram.rows(li, hashes[:, li].reshape(-1))).reshape(T, -1)
        kv = rows @ self.w(l, 'engram_wkv.weight').T
        key, val = kv[:, :self.HC * self.D].reshape(T, self.HC, self.D), kv[:, self.HC * self.D:]
        wt = self.w(l, 'engram_q.weight') * self.w(l, 'engram_k.weight')
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * wt * key).sum(-1) * rstd * self.D ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        tap(f'engram_gate-{l}', gate)
        return h + gate[..., None] * val[:, None, :]

    def hc_mixes(self, l, x, half):
        T = x.shape[0]
        flat = x.reshape(T, -1)
        mixes = (flat @ self.w(l, f'hc_{half}_fn.weight').T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        return sinkhorn(mixes, self.w(l, f'hc_{half}_scale.weight'), self.w(l, f'hc_{half}_base.weight'), self.HC, self.hc_iters, self.hc_eps)

    def block(self, l, h, pre_mix, sh, tap, blobs, qrows=None):
        """One block (Block.forward): the attention collapsed by the previous sublayer's mix, the FFN by this
        attention's. h [T, 4, D] -> (h, ffn_pre) of the rows qrows (all by default; every row's keys are made)"""
        if qrows is None:
            qrows = torch.arange(h.shape[0])
        attn_pre, attn_post, attn_comb = self.hc_mixes(l, h, 'attn')
        x = (pre_mix[..., None] * h).sum(1)
        x = rms(x, self.w(l, 'attn_norm.weight'), self.eps)
        tap(f'attn_norm-{l}', x[qrows])
        a = self.attention(l, x, sh, tap, qrows)
        tap(f'attn_out-{l}', a)
        h, attn_pre, attn_post, attn_comb = h[qrows], attn_pre[qrows], attn_post[qrows], attn_comb[qrows]
        h = attn_post[..., None] * a[:, None, :] + (attn_comb[..., None] * h[:, :, None, :]).sum(1)
        tap(f'hc_attn_post-{l}', h)
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(l, h, 'ffn')
        x = (attn_pre[..., None] * h).sum(1)
        x = rms(x, self.w(l, 'ffn_norm.weight'), self.eps)
        tap(f'ffn_norm-{l}', x)
        y, ids = self.moe(l, x, tap, blobs)
        tap(f'ffn_out-{l}', y)
        h = ffn_post[..., None] * y[:, None, :] + (ffn_comb[..., None] * h[:, :, None, :]).sum(1)
        return h, ffn_pre, ids


class Taps:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.index = open(os.path.join(out, 'index.tsv'), 'w')

    def __call__(self, name, t):
        a = t.detach().to(F).contiguous().numpy()
        a.tofile(os.path.join(self.out, name + '.f32'))
        d = list(a.shape)[::-1] + [1] * 4
        self.index.write('\t'.join([name, 'f32'] + [str(x) for x in d[:4]]) + '\n')
        self.index.flush()


def teacher_input(dirname, name, shape):
    a = np.fromfile(os.path.join(dirname, name + '.f32'), np.float32)
    return torch.from_numpy(a[:int(np.prod(shape))].reshape(shape).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('engram')
    ap.add_argument('out')
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--teacher', default=None)
    ap.add_argument('--from', dest='first', type=int, default=0, help='the first layer (with --teacher)')
    ap.add_argument('--rows', type=int, default=0, help='only the last ROWS query rows (with --teacher)')
    args = ap.parse_args()
    r = Ref(args.model, None if args.engram == '-' else args.engram)
    toks = [int(t) for t in args.tokens.split()]
    T = len(toks)
    tap = Taps(args.out)
    with open(os.path.join(args.out, 'tokens.txt'), 'w') as f:
        f.write(' '.join(map(str, toks)) + '\n')
    nl = args.layers or r.L
    g = r.g
    hashes = engram_hash(toks, g.get('engram.token_map'), g.get('engram.multipliers'), g.get('engram.primes'), g.get('engram.offsets'),
                         int(g.get('engram.token_map')[g.arch('engram.pad_id_compressed')]), g.arch('engram.max_ngram_size'))
    np.savetxt(os.path.join(args.out, 'engram_rows.txt'), hashes.reshape(T, -1), fmt='%d')
    emb = torch.from_numpy(g.rows('token_embd.weight', toks))
    tap('inp_embd', emb)
    h = emb[:, None, :].repeat(1, r.HC, 1)
    pre_mix = torch.zeros(T, r.HC)
    pre_mix[:, 0] = 1
    sh = {}
    cache = {}

    def blobs(l):
        if l not in cache:
            cache.clear()
            cache[l] = VqBlob(g.raw(r.name(l, 'ffn_exps_vq.blob')))
        return cache[l]
    main_h = []
    t0 = time.time()
    qrows = torch.arange(T - args.rows, T) if args.rows else None
    for l in range(args.first, nl):
        if args.teacher:
            h = teacher_input(args.teacher, f'l_in-{l}', (T, r.HC, r.D))
            if l > 0:
                pre_mix = teacher_input(args.teacher, f'pre_mix-{l}', (T, r.HC))
        tap(f'l_in-{l}', h)
        if l > 0:
            tap(f'pre_mix-{l}', pre_mix)
        if r.engram and l in r.eng_layers:
            h = r.engram_apply(l, h, torch.from_numpy(hashes), tap)
            tap(f'hc_engram-{l}', h)
        if l in r.mtp_targets:
            main_h.append(h.mean(1))
        h, pre_mix, ids = r.block(l, h, pre_mix, sh, tap, blobs, qrows)
        tap(f'l_out-{l}', h)
        tap(f'experts-{l}', ids.float())
        print(f'layer {l}: {time.time() - t0:.1f} s', flush=True)
    if nl == r.L:
        x = (pre_mix[..., None] * h).sum(1)
        x = rms(x, torch.from_numpy(g.get('output_norm.weight')), r.eps)
        tap('result_norm', x[-1:])
        head = torch.from_numpy(g.get('output.weight'))
        logits = x @ head.T
        tap('result_output', logits[-1:])
        tap('result_output_all', logits)
        nxt = int(logits[-1].argmax())
        with open(os.path.join(args.out, 'next.txt'), 'w') as f:
            f.write(f'{nxt}\n')
        print('next', nxt, 'top5', logits[-1].topk(5).indices.tolist())
        if main_h:
            tap('mtp_main_hidden', torch.cat(main_h, -1))


if __name__ == '__main__':
    main()
