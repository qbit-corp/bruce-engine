import struct, sys, numpy as np

od = open("tmp/oracle5.bin", "rb").read()
magic = od[:8]
n_tok, n_virt, n_embd, n_vocab = struct.unpack("<4i", od[8:24])
base = 24
x = np.frombuffer(od, dtype=np.float32, count=n_tok * n_virt * n_embd, offset=base).reshape(n_tok, n_virt, n_embd)
lg = np.frombuffer(od, dtype=np.float32, count=n_tok * n_vocab, offset=base + n_tok * n_virt * n_embd * 4).reshape(n_tok, n_vocab)
print("oracle dump:", n_tok, n_virt, n_embd, n_vocab)

eng = np.fromfile("tmp/x5.bin", dtype=np.float32).reshape(n_virt, n_embd)
t = n_tok - 1
print("layer  blk  maxdiff      rms        rel_rms   cos")
worst = []
for l in range(n_virt):
    d = x[t, l] - eng[l]
    rms = float(np.sqrt(np.mean(d * d)))
    nr = float(np.linalg.norm(x[t, l]))
    rel = rms / max(nr / np.sqrt(n_embd), 1e-9)
    cos = float(np.dot(x[t, l], eng[l]) / (np.linalg.norm(x[t, l]) * np.linalg.norm(eng[l]) + 1e-30))
    worst.append((rel, l, rms, float(np.abs(d).max()), cos, nr))
    if l < 4 or l >= n_virt - 4 or rel > 1e-3:
        print("%4d %4d  %.6e  %.6e  %.3e  %.7f" % (l, l, np.abs(d).max(), rms, rel, cos))
print("\nworst 12 layers by rel rms:")
for rel, l, rms, mx, cos, nr in sorted(worst, reverse=True)[:12]:
    print("  layer %3d rel %.4e rms %.4e max %.4e cos %.7f norm %.4f" % (l, rel, rms, mx, cos, nr))

def logsoftmax(z):
    m = z.max()
    return z - (m + np.log(np.exp(z - m, dtype=np.float64).sum()))

o = logsoftmax(lg[t].astype(np.float64))
e_lg = np.fromfile("tmp/logits5.bin", dtype=np.float32).astype(np.float64)
e = logsoftmax(e_lg)
oo = np.argsort(-o)[:8]
print("\noracle top8:", [(int(i), round(float(o[i]), 4)) for i in oo])
print("engine top8:", [(int(i), round(float(e[i]), 4)) for i in np.argsort(-e)[:8]])
print("engine values at oracle top8:", [(int(i), round(float(e[i]), 4), round(float(e[i] - o[i]), 4)) for i in oo])
