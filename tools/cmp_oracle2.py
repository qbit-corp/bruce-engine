import struct, sys, numpy as np

op, xp, lp = sys.argv[1], sys.argv[2], sys.argv[3]
tok = int(sys.argv[4]) if len(sys.argv) > 4 else -1

od = open(op, "rb").read()
n_tok, n_virt, n_embd, n_vocab = struct.unpack("<4i", od[8:24])
base = 24
x = np.frombuffer(od, dtype=np.float32, count=n_tok * n_virt * n_embd, offset=base).reshape(n_tok, n_virt, n_embd)
lg = np.frombuffer(od, dtype=np.float32, count=n_tok * n_vocab, offset=base + n_tok * n_virt * n_embd * 4).reshape(n_tok, n_vocab)
t = tok if tok >= 0 else n_tok - 1
eng = np.fromfile(xp, dtype=np.float32).reshape(n_virt, n_embd)
print("oracle tokens=%d layers=%d, comparing token %d" % (n_tok, n_virt, t))

rows = []
for l in range(n_virt):
    a, b = x[t, l], eng[l]
    d = a - b
    rms = float(np.sqrt(np.mean(d * d)))
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rows.append((cos, l, rms, float(np.abs(d).max())))
print("first 6 and last 3 layers:")
for cos, l, rms, mx in rows[:6] + rows[-3:]:
    print("  layer %3d cos %.7f rms %.3e max %.3e" % (l, cos, rms, mx))
print("worst 8 by (1-cos):")
for cos, l, rms, mx in sorted(rows)[:8]:
    print("  layer %3d cos %.7f rms %.3e max %.3e" % (l, cos, rms, mx))

def logsoftmax(z):
    m = z.max()
    return z - (m + np.log(np.exp(z - m, dtype=np.float64).sum()))

o = logsoftmax(lg[t].astype(np.float64))
e = logsoftmax(np.fromfile(lp, dtype=np.float32).astype(np.float64))
oo = np.argsort(-o)[:8]
print("oracle top8:", [(int(i), round(float(o[i]), 4)) for i in oo])
print("engine top8:", [(int(i), round(float(e[i]), 4)) for i in np.argsort(-e)[:8]])
print("engine at oracle top8:", [(int(i), round(float(e[i]), 4), round(float(e[i] - o[i]), 4)) for i in oo])
