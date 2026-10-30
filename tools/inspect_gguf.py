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
    kv[k] = rval(t)

for k in ['dory.rope.profile', 'dory.layer_pattern']:
    print(k, '=', kv.get(k))

for k, v in kv.items():
    if isinstance(v, str) and len(v) > 300:
        v = v[:300] + '...'
    if isinstance(v, list) and len(v) > 16:
        v = v[:16] + ['... len=%d' % len(v)]
    if k not in ('dory.rope.profile', 'dory.layer_pattern', 'tokenizer.ggml.tokens',
                 'tokenizer.ggml.token_type', 'tokenizer.ggml.merges'):
        print(k, '=', v)

types = {0: ('F32', 4), 1: ('F16', 2), 8: ('Q8_0', 0), 13: ('Q5_K', 0), 14: ('Q6_K', 0), 30: ('BF16', 2)}
tensors = []
for i in range(nT):
    name = rstr()
    nd = rd('I')
    dims = [rd('Q') for _ in range(nd)]
    tt = rd('I')
    off = rd('Q')
    tensors.append((name, dims, tt, off))

nbytes = {0: 4, 1: 2, 8: 34/32, 13: 176/256, 14: 210/256, 30: 2}
tot = 0
by = {}
for name, dims, tt, off in tensors:
    n = 1
    for d in dims: n *= d
    b = int(n * nbytes[tt])
    tot += b
    suf = name.split('.', 2)[-1] if name.startswith('blk.') else name
    key = (suf, types[tt][0])
    by[key] = by.get(key, 0) + b
print('\ntotal tensor bytes', tot)
for (suf, ty), b in sorted(by.items(), key=lambda x: -x[1]):
    print('%-40s %-6s %8.1f MB' % (suf, ty, b / 1e6))
print('\nper-tensor present:')
seen = {}
for name, dims, tt, off in tensors:
    if name.startswith('blk.'):
        b = int(name.split('.')[1])
        if b not in seen:
            seen[b] = (name, dims, types.get(tt, (str(tt), 0))[0])
for b in sorted(seen):
    print(b, seen[b])
print('\ntokenizer:')
print('model', kv.get('tokenizer.ggml.model'), 'pre', kv.get('tokenizer.ggml.pre'))
print('bos', kv.get('tokenizer.ggml.bos_token_id'), 'eos', kv.get('tokenizer.ggml.eos_token_id'),
      'unk', kv.get('tokenizer.ggml.unknown_token_id'), 'pad', kv.get('tokenizer.ggml.padding_token_id'),
      'add_bos', kv.get('tokenizer.ggml.add_bos_token'), 'add_eos', kv.get('tokenizer.ggml.add_eos_token'))
print('ntok', len(kv['tokenizer.ggml.tokens']), 'nmerges', len(kv['tokenizer.ggml.merges']))
print('chat_template:')
print(kv.get('tokenizer.chat_template', '')[:4000])
