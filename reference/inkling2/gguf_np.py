"""A GGUF's tensors as numpy arrays (F32, F16, BF16, Q8_0, NVFP4), for checking the inkling2 engine's stages by hand
against a llama.cpp dump (reference/inkling2/ink-dump.cpp). Shapes are numpy's: outermost first ([rows, cols])."""
import struct
import numpy as np

TYPES = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 30: ('BF16', 1, 2), 8: ('Q8_0', 32, 34), 40: ('NVFP4', 64, 36)}


class Gguf:
    def __init__(self, path):
        self.f = open(path, 'rb')
        r = self.f
        def u32(): return struct.unpack('<I', r.read(4))[0]
        def u64(): return struct.unpack('<Q', r.read(8))[0]
        def s(): return r.read(u64()).decode('utf-8', 'replace')
        sc = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
        def val(t):
            if t in sc:
                f = sc[t]
                return struct.unpack(f, r.read(struct.calcsize(f)))[0]
            if t == 8:
                return s()
            et, n = u32(), u64()
            return [val(et) for _ in range(n)]
        assert r.read(4) == b'GGUF'
        u32()
        nt, nkv = u64(), u64()
        self.kv = {}
        for _ in range(nkv):
            k = s()
            self.kv[k] = val(u32())
        self.t = {}
        for _ in range(nt):
            name = s()
            nd = u32()
            dims = [u64() for _ in range(nd)]
            ty = u32()
            off = u64()
            self.t[name] = (dims, ty, off)
        align = self.kv.get('general.alignment', 32)
        self.data = (r.tell() + align - 1) // align * align

    def get(self, name):
        dims, ty, off = self.t[name]
        tn, vb, bb = TYPES[ty]
        n = int(np.prod(dims))
        self.f.seek(self.data + off)
        raw = np.frombuffer(self.f.read(n // vb * bb), dtype=np.uint8)
        if tn == 'F32':
            v = raw.view('<f4').astype(np.float32)
        elif tn == 'F16':
            v = raw.view('<f2').astype(np.float32)
        elif tn == 'BF16':
            v = (raw.view('<u2').astype(np.uint32) << 16).view(np.float32)
        elif tn == 'Q8_0':
            b = raw.reshape(-1, 34)
            d = b[:, :2].copy().view('<f2').astype(np.float32)
            v = (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)
        else:
            b = raw.reshape(-1, 36)
            e = (b[:, :4] >> 3) & 15
            m = (b[:, :4] & 7).astype(np.float32)
            d = np.where(e == 0, m * 2.0 ** -10, (8 + m) * 2.0 ** (e.astype(np.float32) - 11))
            d = np.where((b[:, :4] == 0) | (b[:, :4] == 0x7F), 0.0, d).astype(np.float32)
            tab = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
            q = b[:, 4:].reshape(-1, 4, 8)
            lo, hi = tab[q & 15], tab[q >> 4]
            v = (np.concatenate([lo, hi], axis=2) * d[:, :, None]).reshape(-1)
        return v.reshape(list(reversed(dims)))


def get_rows(g, name, rows):
    """rows of a 2-D tensor (an embedding's tokens) without reading the rest"""
    import io
    dims, ty, off = g.t[name]
    tn, vb, bb = TYPES[ty]
    rb = dims[0] // vb * bb
    out = []
    for r in rows:
        g.f.seek(g.data + off + r * rb)
        sub = Gguf.__new__(Gguf)
        sub.f, sub.data = io.BytesIO(g.f.read(rb)), 0
        sub.t = {'x': ([dims[0]], ty, 0)}
        out.append(sub.get('x'))
    return np.stack(out)


def rms(x, w=None, eps=1e-6):
    y = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
    return (y * w if w is not None else y).astype(np.float32)


def mix(states, query, eps=1e-6):
    """attention over the depth: states [S, T, C], query [Q, C] -> [Q, T, C]"""
    n = np.stack([rms(s, eps=eps) for s in states])
    sc = np.einsum('stc,qc->qts', n, query) / np.sqrt(n.shape[-1])
    p = np.exp(sc - sc.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum('qts,stc->qtc', p, n)


def load(dir_, name, cols):
    return np.fromfile(f'{dir_}/{name}.f32', dtype=np.float32).reshape(-1, cols)


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))
