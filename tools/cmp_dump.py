import sys
import numpy as np

a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 2560)
b = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, 2560)
cs = []
for l in range(min(len(a), len(b))):
    x, y = a[l], b[l]
    cs.append(float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)))
print("layers %d  min cos %.6f at %d  l0 %.6f l1 %.6f l2 %.6f" %
      (len(cs), min(cs), int(np.argmin(cs)), cs[0], cs[1], cs[2]))
if len(sys.argv) > 3:
    print(" ".join("%d:%.5f" % (i, c) for i, c in enumerate(cs)))
