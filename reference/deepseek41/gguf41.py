"""DeepSeek-V4.1-Flash VQ GGUF (YoungAi's vq8sh14-q4k-mtpnative) read on the CPU: the header, every stored type this
file has dequantized to float32 (F32, BF16, I32/I64, q4_K, fp8_32x32, the DQVL/DQV3 VQ expert blobs), and the Engram
n-gram hash and table rows. The reference (ds41ref.py) builds the model's blocks on top of this; nothing here needs a GPU.

The VQ decode follows YoungAi's vq_fmt.h / ds4-dfm-rs's vq.rs (the CPU oracle): value[row][col] =
codebook[idx(row, col / 8)][col % 8] * gain[row], the codebook E4M3 (one per layer, at cb_off in the blob), the index a
12-bit little-endian field of the row's own stream plus (13-bit layers) a 13th bit from the row's plane."""
import mmap
import os
import struct

import numpy as np

# ggml type codes this file uses
F32, Q4K, I32, I64, BF16, VQBLOB, FP8B32 = 0, 12, 26, 27, 30, 42, 44


def e4m3_lut():
    """E4M3FN byte -> float32 (0x7f / 0xff NaN, exp 0 the subnormal m * 2^-9)."""
    out = np.zeros(256, np.float32)
    for b in range(256):
        a, s = b & 0x7F, -1.0 if b & 0x80 else 1.0
        if a == 0x7F:
            out[b] = np.nan
            continue
        e, m = a >> 3, a & 7
        out[b] = s * (m * 2.0 ** -9 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
    return out


E4M3 = e4m3_lut()


def e8m0(b):
    """E8M0 scale bytes -> float32 2^(e - 127) (e = 0: 2^-127)."""
    b = np.asarray(b, np.uint32)
    return np.where(b == 0, np.uint32(0x00400000), b << 23).astype(np.uint32).view(np.float32)


class Gguf:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        rd = self._rd
        magic, ver = rd('<4sI')
        assert magic == b'GGUF' and ver == 3, (magic, ver)
        nt, nkv = rd('<QQ')
        self.meta = {}
        for _ in range(nkv):
            k = self._str()
            (t,) = rd('<I')
            self.meta[k] = self._val(t)
        align = self.meta.get('general.alignment', 32)
        raw = []
        for _ in range(nt):
            name = self._str()
            (nd,) = rd('<I')
            dims = rd('<' + 'Q' * nd)
            ty, off = rd('<IQ')
            raw.append((name, list(reversed(dims)), ty, off))
        base = (self.pos + align - 1) // align * align
        self.t = {}
        for name, shape, ty, off in raw:
            n = int(np.prod(shape))
            nb = {F32: 4 * n, I32: 4 * n, I64: 8 * n, BF16: 2 * n, Q4K: n // 256 * 144, VQBLOB: n, FP8B32: n // 1024 * 1025}[ty]
            self.t[name] = (shape, ty, base + off, nb)

    pos = 0

    def _rd(self, fmt):
        n = struct.calcsize(fmt)
        v = struct.unpack_from(fmt, self.mm, self.pos)
        self.pos += n
        return v

    def _str(self):
        (n,) = self._rd('<Q')
        s = self.mm[self.pos:self.pos + n].decode('utf-8', 'replace')
        self.pos += n
        return s

    def _val(self, t):
        fm = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if t == 8:
            return self._str()
        if t == 9:
            at, n = self._rd('<IQ')
            return [self._val(at) for _ in range(n)]
        return self._rd('<' + fm[t])[0]

    def arch(self, k, default=None):
        return self.meta.get('deepseek4.' + k, default)

    def has(self, name):
        return name in self.t

    def raw(self, name):
        shape, ty, off, nb = self.t[name]
        return np.frombuffer(self.mm, np.uint8, nb, off)

    def get(self, name):
        """The tensor as float32 (integers as int64), row-major [outer.., inner]."""
        shape, ty, off, nb = self.t[name]
        r = self.raw(name)
        if ty == F32:
            return r.view(np.float32).reshape(shape).copy()
        if ty == BF16:
            return (r.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if ty == I32:
            return r.view(np.int32).astype(np.int64).reshape(shape)
        if ty == I64:
            return r.view(np.int64).reshape(shape).copy()
        if ty == Q4K:
            return dequant_q4k(r).reshape(shape)
        if ty == FP8B32:
            rows, cols = int(np.prod(shape[:-1])), shape[-1]
            return dequant_fp8b32(r, rows, cols).reshape(shape)
        raise ValueError(f'{name}: type {ty}')

    def rows(self, name, ids):
        """Rows `ids` of a [n, cols] q4_K / BF16 / F32 matrix (the embedding), float32."""
        shape, ty, off, nb = self.t[name]
        cols = shape[-1]
        r = self.raw(name)
        rb = {Q4K: cols // 256 * 144, BF16: 2 * cols, F32: 4 * cols}[ty]
        out = np.zeros((len(ids), cols), np.float32)
        for k, i in enumerate(ids):
            b = r[int(i) * rb:(int(i) + 1) * rb]
            if ty == Q4K:
                out[k] = dequant_q4k(b)
            elif ty == BF16:
                out[k] = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                out[k] = b.view(np.float32)
        return out


def dequant_q4k(b):
    """ggml's q4_K: blocks of 256 - d, dmin (f16), 12 bytes of 6-bit scales / mins, 128 bytes of 4-bit codes."""
    b = np.asarray(b).reshape(-1, 144)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
    q = b[:, 4:16].astype(np.int32)
    sc = np.zeros((b.shape[0], 8), np.int32)
    mn = np.zeros((b.shape[0], 8), np.int32)
    for j in range(8):
        if j < 4:
            sc[:, j] = q[:, j] & 63
            mn[:, j] = q[:, j + 4] & 63
        else:
            sc[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
            mn[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    qs = b[:, 16:144].reshape(-1, 4, 32)
    out = np.zeros((b.shape[0], 8, 32), np.float32)
    for k in range(4):
        lo, hi = (qs[:, k] & 0xF).astype(np.float32), (qs[:, k] >> 4).astype(np.float32)
        for j, v in ((2 * k, lo), (2 * k + 1, hi)):
            out[:, j] = d[:, None] * sc[:, j, None] * v - dmin[:, None] * mn[:, j, None]
    return out.reshape(-1)


def dequant_fp8b32(b, rows, cols):
    """fp8_32x32: an E4M3 plane [rows][cols], then an E8M0 scale a 32 x 32 tile [ceil(rows/32)][ceil(cols/32)]."""
    plane = np.asarray(b[:rows * cols]).reshape(rows, cols)
    tr, tc = (rows + 31) // 32, (cols + 31) // 32
    s = e8m0(np.asarray(b[rows * cols:rows * cols + tr * tc]).reshape(tr, tc))
    return E4M3[plane] * np.repeat(np.repeat(s, 32, 0), 32, 1)[:rows, :cols]


class VqBlob:
    """A layer's (or a draft tower's) `ffn_exps_vq.blob`: 'DQVL', version, layer, experts, a slot table of nexp x 3
    payload offsets (w1 gate, w3 up, w2 down), the payloads ('DQV3': dim 8, nc, rows, cols, flags, 12, cb_off, gains,
    the 12-bit stream, the 13th-bit plane)."""

    def __init__(self, raw):
        self.b = raw
        magic, ver, layer, nexp = struct.unpack_from('<4I', raw, 0)
        assert magic == 0x4C565144 and ver == 3, (hex(magic), ver)
        self.layer, self.nexp = layer, nexp
        self.slots = np.frombuffer(raw, np.uint64, nexp * 3, 16).reshape(nexp, 3)

    def header(self, e, which):
        off = int(self.slots[e, which])
        magic, dim, nc, rows, cols, flags, mnb, cb_off = struct.unpack_from('<IHHIIIIQ', self.b, off)
        assert magic == 0x33565144 and dim == 8 and mnb == 12, (hex(magic), dim, mnb)
        return dict(off=off, nc=nc if nc else 65536, rows=rows, cols=cols, flags=flags, cb_off=cb_off)

    def codebook(self, e=0, which=0):
        h = self.header(e, which)
        return E4M3[np.asarray(self.b[h['cb_off']:h['cb_off'] + h['nc'] * 8])].reshape(h['nc'], 8)

    def indices(self, e, which):
        h = self.header(e, which)
        rows, cols, nc = h['rows'], h['cols'], h['nc']
        nidx = cols // 8
        mrow = (nidx * 12 + 7) // 8
        g0 = h['off'] + 32
        gains = np.frombuffer(self.b, np.float16, rows, g0).astype(np.float32)
        s0 = g0 + rows * 2
        stream = np.asarray(self.b[s0:s0 + rows * mrow + 4]).astype(np.uint32)
        bit = np.arange(nidx) * 12
        at = (np.arange(rows) * mrow)[:, None] + (bit >> 3)[None, :]
        w = stream[at] | (stream[at + 1] << 8) | (stream[at + 2] << 16)
        idx = (w >> (bit & 7)[None, :]) & 0xFFF
        nbit = max(1, int(np.ceil(np.log2(nc))))
        if nbit > 12:
            assert h['flags'] & 2
            prow = (nidx + 7) // 8
            plane = np.asarray(self.b[s0 + rows * mrow:s0 + rows * mrow + rows * prow]).reshape(rows, prow)
            i = np.arange(nidx)
            idx = idx | (((plane[:, i >> 3] >> (i & 7)) & 1).astype(np.uint32) << 12)
        return idx, gains

    def matrix(self, e, which, cb=None):
        """Expert e's matrix `which` (0 gate, 1 up, 2 down) as float32 [rows][cols]."""
        if cb is None:
            cb = self.codebook(e, which)
        idx, gains = self.indices(e, which)
        rows = idx.shape[0]
        return (cb[idx] * gains[:, None, None]).reshape(rows, -1)

    def payload_span(self, e):
        """(first byte, bytes) of expert e's three payloads, when they lie back to back in gate, up, down order."""
        offs = [int(x) for x in self.slots[e]]
        return offs


def engram_hash(tokens, token_map, mult, primes, offsets, pad_id, max_ngram=4):
    """The rows each position's n-grams hash to, per engram layer: [len(tokens), layers, (max_ngram - 1) x heads]
    (DeepSeek's NgramHashState.forward over a whole sequence from position 0; no dead tokens)."""
    comp = np.asarray(token_map)[np.asarray(tokens)].astype(np.int64)
    n = len(comp)
    toks = np.full((n, max_ngram), pad_id, np.int64)
    for s in range(max_ngram):
        toks[s:, s] = comp[:n - s] if s else comp
    layers = mult.shape[0]
    heads = primes.shape[2]
    out = np.zeros((n, layers, (max_ngram - 1) * heads), np.int64)
    for li in range(layers):
        prod = toks * mult[li][None, :]
        rolling = prod[:, 0]
        hs = []
        for i in range(1, max_ngram):
            rolling = np.bitwise_xor(rolling, prod[:, i])
            hs.append(rolling[:, None] % primes[li, i - 1][None, :])
        out[:, li] = np.concatenate(hs, 1) + offsets[li][None, :]
    return out


class Engram:
    """The two official 1017 shards' tables, read in rows: weight e4m3 [N][256] at weight_offset, scale e8m0 [N][8] at
    scale_offset (the GGUF's metadata; the shard file named by its table_path's base name in `engram_dir`)."""

    def __init__(self, g, engram_dir):
        self.layers = g.arch('engram.layer_ids')
        self.hd = g.arch('engram.head_dim')
        self.files = []
        for i in range(len(self.layers)):
            p = os.path.join(engram_dir, os.path.basename(g.arch(f'engram.{i}.table_path')))
            self.files.append((open(p, 'rb'), g.arch(f'engram.{i}.weight_offset'), g.arch(f'engram.{i}.scale_offset')))

    def rows(self, li, ids):
        """Layer li's table rows ids [n] -> float32 [n][head_dim] (e4m3 x e8m0 a 32, rounded to bf16 as the model does)."""
        f, wo, so = self.files[li]
        hd = self.hd
        out = np.zeros((len(ids), hd), np.float32)
        for k, r in enumerate(np.asarray(ids).reshape(-1)):
            w = np.frombuffer(os.pread(f.fileno(), hd, wo + int(r) * hd), np.uint8)
            s = np.frombuffer(os.pread(f.fileno(), hd // 32, so + int(r) * (hd // 32)), np.uint8)
            out[k] = E4M3[w] * np.repeat(e8m0(s), 32)
        return bf16_round(out)


def bf16_round(x):
    """float32 -> the nearest bf16 (ties to even), as float32."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


if __name__ == '__main__':
    import sys
    g = Gguf(sys.argv[1])
    print('tensors', len(g.t))
    for name in ['blk.0.ffn_exps_vq.blob', 'blk.14.ffn_exps_vq.blob', 'mtp.0.ffn_exps_vq.blob']:
        b = VqBlob(g.raw(name))
        print(name, 'layer', b.layer, 'nexp', b.nexp)
        for e in (0, 1, b.nexp - 1):
            hs = [b.header(e, w) for w in range(3)]
            print('  expert', e, [(h['off'], h['nc'], h['rows'], h['cols'], h['flags'], h['cb_off']) for h in hs])
        m = b.matrix(0, 0)
        print('  gate0 rms', float(np.sqrt((m * m).mean())), m[0, :8])
