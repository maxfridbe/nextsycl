# m3kv.py IN OUT KEY VALUE: copy a small GGUF with one integer KV changed (same width)
import sys, struct
src, dst, key, val = sys.argv[1], sys.argv[2], sys.argv[3].encode(), int(sys.argv[4])
b = bytearray(open(src, 'rb').read()); f = open(src, 'rb')
def rd(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
def rstr():
    (n,) = rd('<Q'); return f.read(n)
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skipval(t):
    if t in SZ: f.read(SZ[t])
    elif t == 8: rstr()
    elif t == 9:
        (et,) = rd('<I'); (n,) = rd('<Q')
        for _ in range(n): skipval(et)
f.read(4); rd('<I'); nt, nkv = rd('<QQ')
for _ in range(nkv):
    k = rstr(); (t,) = rd('<I'); at = f.tell()
    if k == key:
        fmt = {4:'<I',5:'<i',10:'<Q',11:'<q'}[t]; struct.pack_into(fmt, b, at, val); print('set', k, t, val); break
    skipval(t)
open(dst, 'wb').write(b)
