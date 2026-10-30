import struct, sys

def load(p):
    b = open(p, 'rb').read()
    n = len(b) // 4
    return struct.unpack('%df' % n, b[:n * 4])

a = load(sys.argv[1] if len(sys.argv) > 2 else 'tmp/lg_tc.bin')
b = load(sys.argv[2] if len(sys.argv) > 2 else 'tmp/lg_ex.bin')
print('vocab', len(a), len(b))
if len(a) != len(b):
    sys.exit('size mismatch')

import math
md = max(abs(x - y) for x, y in zip(a, b))
scale = max(abs(y) for y in b)
print('max |diff| %.6f  (logit max %.3f)' % (md, scale))

def top(v, k):
    idx = sorted(range(len(v)), key=lambda i: -v[i])[:k]
    return idx

t1, t2 = top(a, 1)[0], top(b, 1)[0]
print('argmax tc=%d exact=%d  %s' % (t1, t2, 'MATCH' if t1 == t2 else 'DIFFER'))
for k in (5, 10, 50):
    o1, o2 = set(top(a, k)), set(top(b, k))
    print('top-%d overlap %d/%d' % (k, len(o1 & o2), k))
# rank correlation of the top-50 sets and gap at the top
s1 = sorted(a, reverse=True)
print('top-2 gap tc %.4f exact %.4f' % (s1[0] - s1[1], sorted(b, reverse=True)[1] and (sorted(b, reverse=True)[0] - sorted(b, reverse=True)[1])))
