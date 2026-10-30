import struct, sys
import numpy as np

op, xp = sys.argv[1], sys.argv[2]
tok = int(sys.argv[3]) if len(sys.argv) > 3 else -1

od = open(op, "rb").read()
nt, nv, ne, nvo = struct.unpack("<4i", od[8:24])
x = np.frombuffer(od, dtype=np.float32, count=nt * nv * ne, offset=24).reshape(nt, nv, ne)
e = np.fromfile(xp, dtype=np.float32).reshape(nv, ne)
t = tok if tok >= 0 else nt - 1
cs = []
for l in range(nv):
    a, b = x[t, l], e[l]
    cs.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))
print("tokens=%d compare token %d: min cos %.6f at layer %d, l0 %.6f, l6 %.6f, l110 %.6f" %
      (nt, t, min(cs), int(np.argmin(cs)), cs[0], cs[6], cs[-2]))
if len(sys.argv) > 4:
    print("all:", " ".join("%d:%.5f" % (i, c) for i, c in enumerate(cs)))
