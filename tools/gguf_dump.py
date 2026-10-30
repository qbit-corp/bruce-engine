import os
import struct, sys, json

PATH = os.environ.get("BRUCE_MODEL", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
f = open(PATH, 'rb', buffering=1 << 20)

def rd(fmt, n=1):
    sz = struct.calcsize(fmt) * n
    d = f.read(sz)
    if len(d) != sz:
        raise EOFError
    return struct.unpack('<' + fmt * n, d) if n > 1 else struct.unpack('<' + fmt, d)[0]

def rstr():
    n = rd('Q')
    return f.read(n).decode('utf-8', 'replace')

def rval(t):
    if t == 8: return rstr()
    if t == 9:
        at = rd('I'); n = rd('Q')
        if at == 8:
            v = [rstr() for _ in range(n)]
        else:
            fmts = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: 'B', 10: 'Q', 11: 'q', 12: 'd'}
            v = list(rd(fmts[at], n))
        if len(v) > 12:
            v = v[:12] + ['... len=%d' % n]
        return v
    if t == 0: return rd('B')
    if t == 1: return rd('b')
    if t == 2: return rd('H')
    if t == 3: return rd('h')
    if t == 4: return rd('I')
    if t == 5: return rd('i')
    if t == 6: return rd('f')
    if t == 7: return bool(rd('B'))
    if t == 10: return rd('Q')
    if t == 11: return rd('q')
    if t == 12: return rd('d')
    raise ValueError(t)

magic = rd('I'); ver = rd('I'); nT = rd('Q'); nK = rd('Q')
kv = {}
for i in range(nK):
    k = rstr(); t = rd('I')
    print('K', i, k, 't', t, file=sys.stderr, flush=True)
    kv[k] = rval(t)

for k, v in kv.items():
    if k != 'dory.rope.profile':
        print(k, '=', v)

# tensor infos
types = {0: ('F32', 4), 1: ('F16', 2), 2: ('Q4_0', 0), 3: ('Q4_1', 0), 6: ('Q5_0', 0), 7: ('Q5_1', 0),
         8: ('Q8_0', 0), 9: ('Q8_1', 0), 10: ('Q2_K', 0), 11: ('Q3_K', 0), 12: ('Q4_K', 0),
         13: ('Q5_K', 0), 14: ('Q6_K', 0), 15: ('Q8_K', 0), 16: ('IQ2_XXS', 0), 17: ('IQ2_XS', 0),
         18: ('IQ3_XXS', 0), 19: ('IQ1_S', 0), 20: ('IQ4_NL', 0), 21: ('IQ3_S', 0), 22: ('IQ2_S', 0),
         23: ('IQ4_XS', 0), 24: ('I8', 1), 25: ('I16', 2), 26: ('I32', 4), 27: ('I64', 8), 28: ('F64', 8),
         29: ('IQ1_M', 0), 30: ('BF16', 2), 34: ('TQ1_0', 0), 35: ('TQ2_0', 0), 39: ('MXFP4', 0)}

tensors = []
for i in range(nT):
    name = rstr()
    nd = rd('I')
    dims = [rd('Q') for _ in range(nd)]
    tt = rd('I')
    off = rd('Q')
    tensors.append((name, dims, types.get(tt, ('?%d' % tt, 0)), off))

print('\n=== TENSORS ===')
by_prefix = {}
for name, dims, tt, off in tensors:
    print(name, dims, tt[0])
