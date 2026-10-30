import numpy as np
P = 1088


def dec(blob, slot, nh=4):
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


def kv(f, off):
    a = np.fromfile(f, dtype=np.float32)
    return a[off:off + 544].tobytes()


K = 31232
V = 31776
files = {'c1': 'tmp/c1.bin', 'a16': 'tmp/a16.bin', 'b2': 'tmp/b16.bin', 'd2': 'tmp/d2.bin'}
data = {k: (kv(v, K), kv(v, V)) for k, v in files.items()}
keys = list(data)
for nm, sl in (('K', 0), ('V', 1)):
    print('---', nm)
    for a in keys:
        for b in keys:
            if a >= b:
                continue
            x = dec(data[a][sl], 0)
            y = dec(data[b][sl], 0)
            c = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
            print('  %s vs %s cos %.5f' % (a, b, c))
print('--- cross slot: b2 slot0 vs b2 slot1')
for nm, sl in (('K', 0), ('V', 1)):
    x = dec(data['b2'][sl], 0)
    y = dec(data['b2'][sl], 1)
    print(nm, 'cos %.5f' % float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)))
print('--- a16 slot0 vs b2 slot1')
for nm, sl in (('K', 0), ('V', 1)):
    x = dec(data['a16'][sl], 0)
    y = dec(data['b2'][sl], 1)
    print(nm, 'cos %.5f' % float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)))
