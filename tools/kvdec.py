import numpy as np, sys
P = 1088

def dec(blob, slot=1, nh=4):
    out = []
    base = slot * P
    for h in range(nh):
        off = base + h * 272
        for b in range(8):
            blk = blob[off + b * 34: off + b * 34 + 34]
            d = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float64)
            q = np.frombuffer(blk[2:], dtype=np.int8).astype(np.float64)
            out.append(q * d)
    return np.concatenate(out)

ka = open('tmp/a16.bin.k', 'rb').read()
kb = open('tmp/b16.bin.k', 'rb').read()
va = open('tmp/a16.bin.v', 'rb').read()
vb = open('tmp/b16.bin.v', 'rb').read()
for nm, x, y in (('K', ka, kb), ('V', va, vb)):
    a = dec(x); b = dec(y)
    d = np.abs(a - b)
    print(nm, 'slot1: |a| %.5f |b| %.5f  maxdiff %.5f  cos %.7f' % (
        np.linalg.norm(a), np.linalg.norm(b), d.max(),
        float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))))
    # per-head cosine
    cs = []
    for h in range(4):
        aa = a[h * 256:(h + 1) * 256]; bb = b[h * 256:(h + 1) * 256]
        cs.append(round(float(aa @ bb / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-30)), 5))
    print('   per-head cos', cs)
    # slot0 sanity
    a0 = dec(x, 0); b0 = dec(y, 0)
    print('   slot0 maxdiff %.8f' % np.abs(a0 - b0).max())
