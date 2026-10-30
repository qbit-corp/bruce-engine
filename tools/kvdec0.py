import numpy as np
P = 1088


def dec(blob, slot=0, nh=4):
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


a = np.fromfile('tmp/a16.bin', dtype=np.float32)
b = np.fromfile('tmp/b16.bin', dtype=np.float32)
for nm, off in (('K', 31232), ('V', 31776)):
    ka = a[off:off + 544].tobytes()
    kb = b[off:off + 544].tobytes()
    x = dec(ka, 0)
    y = dec(kb, 0)
    print(nm, 'slot0 |a| %.5f |b| %.5f maxdiff %.5f cos %.6f' % (
        np.linalg.norm(x), np.linalg.norm(y), np.abs(x - y).max(),
        float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))))
