# ggufstrip.py IN OUT REGEX: copy a GGUF shard dropping tensors whose name matches REGEX (header rewritten, data copied)
import sys, struct, re
src, dst, rx = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
f = open(src, 'rb')
def rd(fmt):
    n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))
def rstr():
    (n,) = rd('<Q'); return f.read(n)
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skipval(t):
    if t in SZ: f.read(SZ[t])
    elif t == 8: rstr()
    elif t == 9:
        (et,) = rd('<I'); (n,) = rd('<Q')
        for _ in range(n): skipval(et)
    else: raise SystemExit('type %d' % t)
magic = f.read(4); ver, nt, nkv = rd('<IQQ')
assert magic == b'GGUF' and ver == 3
kv_start = f.tell()
align = 32
for _ in range(nkv):
    k = rstr(); (t,) = rd('<I')
    if k == b'general.alignment': (align,) = rd('<I')
    else: skipval(t)
kv_end = f.tell()
infos = []
for _ in range(nt):
    s = f.tell(); name = rstr(); (nd,) = rd('<I'); dims = rd('<%dQ' % nd); ty, off = rd('<IQ')
    infos.append((name, dims, ty, off))
hdr_end = f.tell()
data0 = (hdr_end + align - 1) // align * align
keep = [i for i in infos if not rx.search(i[0].decode())]
print('tensors', nt, '->', len(keep))
f.seek(kv_start); kvbytes = f.read(kv_end - kv_start)
import os
if os.environ.get('DRY'): sys.exit(0)
f.seek(0, 2); total = f.tell() - data0
order = sorted(range(len(infos)), key=lambda i: infos[i][3])
size = {}
for n, i in enumerate(order):
    nxt = infos[order[n + 1]][3] if n + 1 < len(order) else total
    size[i] = nxt - infos[i][3]
newoff = {}; cur = 0
for i in order:
    if rx.search(infos[i][0].decode()): continue
    newoff[i] = cur; cur += size[i]
f.seek(kv_start); kvbytes = f.read(kv_end - kv_start)
keepi = [i for i in range(len(infos)) if i in newoff]
h = b'GGUF' + struct.pack('<IQQ', 3, len(keepi), nkv) + kvbytes
for i in keepi:
    name, dims, ty, off = infos[i]
    h += struct.pack('<Q', len(name)) + name + struct.pack('<I', len(dims)) + struct.pack('<%dQ' % len(dims), *dims) + struct.pack('<IQ', ty, newoff[i])
h += b'\0' * ((-len(h)) % align)
out = open(dst, 'wb'); out.write(h)
for i in order:
    if i not in newoff: continue
    f.seek(data0 + infos[i][3]); left = size[i]
    while left:
        b = f.read(min(left, 1 << 26)); out.write(b); left -= len(b)
out.close()
