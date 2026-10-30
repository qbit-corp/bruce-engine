import struct, sys, os, math
import numpy as np

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

tensors = {}
order = []
for i in range(nT):
    name = rstr()
    nd = rd('I')
    dims = [rd('Q') for _ in range(nd)]
    tt = rd('I')
    off = rd('Q')
    tensors[name] = (dims, tt, off)

align = kv.get('general.alignment', 32)
pos = f.tell()
data_start = (pos + align - 1) // align * align
f.seek(0, 2)
file_size = f.tell()

def raw(name):
    dims, tt, off = tensors[name]
    n = 1
    for d in dims: n *= d
    if tt == 30:
        nb = n * 2
    elif tt == 13:
        nb = n // 256 * 176
    elif tt == 14:
        nb = n // 256 * 210
    else:
        raise ValueError('type %d for %s' % (tt, name))
    f.seek(data_start + off)
    return f.read(nb), dims, tt

def to_f32(name):
    buf, dims, tt = raw(name)
    N, K = (dims[1], dims[0]) if len(dims) > 1 else (1, dims[0])
    if tt == 30:
        return (np.frombuffer(buf, dtype='<u2').reshape(N, K).astype(np.uint32) << 16).view(np.float32).copy()
    if tt == 13:
        nsb = K // 256
        dt = np.dtype([('d', '<f2'), ('dmin', '<f2'), ('scales', 'u1', 12), ('qh', 'u1', 32), ('qs', 'u1', 128)])
        a = np.frombuffer(buf, dtype=dt, count=N * nsb).reshape(N, nsb)
        d = a['d'].astype(np.float32); dmin = a['dmin'].astype(np.float32)
        sc = a['scales']
        s = np.zeros((N, nsb, 8), np.float32); m = np.zeros((N, nsb, 8), np.float32)
        for j in range(4):
            s[:, :, j] = sc[:, :, j] & 63
            m[:, :, j] = sc[:, :, j + 4] & 63
        for j in range(4, 8):
            s[:, :, j] = (sc[:, :, j + 4] & 0xF) | ((sc[:, :, j - 4] >> 6) << 4)
            m[:, :, j] = (sc[:, :, j + 4] >> 4) | ((sc[:, :, j] >> 6) << 4)
        qsb = a['qs'].reshape(N, nsb, 4, 32)
        hq = a['qh'].reshape(N, nsb, 32)
        out = np.empty((N, nsb, 8, 32), np.float32)
        for g in range(4):
            for j in range(2):
                q = (qsb[:, :, g, :] >> (4 * j)) & 0xF
                q = q | (((hq >> (2 * g + j)) & 1) << 4)
                out[:, :, 2 * g + j, :] = d[:, :, None] * s[:, :, 2 * g + j][:, :, None] * q \
                    - dmin[:, :, None] * m[:, :, 2 * g + j][:, :, None]
        return out.reshape(N, K)
    if tt == 14:
        nsb = K // 256
        dt = np.dtype([('ql', 'u1', 128), ('qh', 'u1', 64), ('sc', 'i1', 16), ('d', '<f2')])
        a = np.frombuffer(buf, dtype=dt, count=N * nsb).reshape(N, nsb)
        d = a['d'].astype(np.float32)
        ql = a['ql'].reshape(N, nsb, 2, 64)
        qh = a['qh'].reshape(N, nsb, 2, 32)
        sc = a['sc'].reshape(N, nsb, 2, 8).astype(np.float32)
        out = np.empty((N, nsb, 2, 4, 32), np.float32)
        idx = (np.arange(32) // 16)
        for h in range(2):
            for qd in range(4):
                off = (qd & 1) * 32
                q = (ql[:, :, h, off:off + 32] >> (4 * (qd >> 1))) & 0xF
                q = q | (((qh[:, :, h, :] >> (2 * qd)) & 3) << 4)
                scl = sc[:, :, h, :][:, :, idx + 2 * qd]
                out[:, :, h, qd, :] = d[:, :, None] * scl * (q.astype(np.float32) - 32)
        return out.reshape(N, K)
    raise ValueError('bad type')

N_EMBD, N_HEAD, N_KV, HD, N_FF, VOCAB, SWA = 2560, 16, 4, 256, 7168, 131072, 1024
EPS = 1e-6
INIT_STD = 0.01769999973475933
ALPHA_INIT = 0.05000000074505806
GATE_INIT = 50.59644317626953

def l2norm(a):
    a = np.asarray(a, np.float32)
    n = a.shape[-1]
    ss = np.sum(a * a, dtype=np.float32)
    return a * ((1.0 / np.sqrt(ss / n + EPS / n)) * (1.0 / np.sqrt(n)))

def l2norm_rows(a):
    n = a.shape[-1]
    ss = np.sum(a * a, axis=-1, dtype=np.float32)
    return a * ((1.0 / np.sqrt(ss / n + EPS / n)) * (1.0 / np.sqrt(n)))[..., None]

def scaled(name, n, fn):
    return fn(to_f32(name).reshape(-1).astype(np.float32))

def emb_row(t):
    dims, tt, off = tensors['token_embd.weight']
    f.seek(data_start + off + t * N_EMBD * 2)
    b = f.read(N_EMBD * 2)
    return (np.frombuffer(b, dtype='<u2').astype(np.uint32) << 16).view(np.float32).copy()

def q8_deq(x):
    x = x.reshape(-1, 32)
    amax = np.max(np.abs(x), axis=1)
    d = amax / 127.0
    d = d.astype(np.float16).astype(np.float32)
    id_ = np.where(d != 0, 1.0 / d, 0.0)
    q = np.rint(x * id_[:, None]).astype(np.int8)
    return (q.astype(np.float32) * d[:, None]).reshape(-1)

def rope(vec, pos_, theta, n_heads):
    # GGML_ROPE_TYPE_NEOX: pairs (i, i + HD/2)
    out = np.empty_like(vec)
    for h in range(n_heads):
        base = h * HD
        for i in range(HD // 2):
            th = pos_ * (theta ** (-2.0 * i / HD))
            c, s = math.cos(th), math.sin(th)
            x0 = vec[base + i]; x1 = vec[base + i + HD // 2]
            out[base + i] = x0 * c - x1 * s
            out[base + i + HD // 2] = x0 * s + x1 * c
    return out

pattern = kv['dory.layer_pattern']
profile = kv['dory.rope.profile']
LAYER_ORDER = list(range(16)) + [16 + i for _ in range(2) for i in range(40)] + list(range(56, 72))

def run(ids, nlayers, debug_block=-1, xdump=None, tapsdump=None, loops=2):
    x = emb_row(ids[0])
    input_h = None
    n_virtual = 16 + 40 * loops + 16
    order = LAYER_ORDER
    rec_end = 16 + 40 * loops
    cache = {}
    xdump_f = None
    xd = None
    if xdump:
        xd = np.zeros((n_virtual, N_EMBD), np.float32)
    taps = None
    for pi, tok in enumerate(ids):
        if pi > 0:
            x = emb_row(tok)
        for vi in range(min(nlayers, n_virtual)):
            b = order[vi]
            if vi == 16:
                input_h = x.copy()
            if 16 <= vi < rec_end and (vi - 16) % 40 == 0:
                x = l2norm(x + input_h)
            resid = x.copy()
            pc = pattern[b]
            alpha = scaled('blk.%d.residual_alpha.weight' % b, N_EMBD, lambda v: np.abs(v * (ALPHA_INIT / INIT_STD)))
            if pc == '-':
                Wg = to_f32('blk.%d.ffn_gate.weight' % b)
                Wu = to_f32('blk.%d.ffn_up.weight' % b)
                Wd = to_f32('blk.%d.ffn_down.weight' % b)
                sg = scaled('blk.%d.ffn_gate_scale.weight' % b, N_FF, lambda v: v * np.sqrt(N_EMBD))
                su = scaled('blk.%d.ffn_up_scale.weight' % b, N_FF, lambda v: v * np.sqrt(N_EMBD))
                g = (Wg @ x) * sg
                u = (Wu @ x) * su
                t = l2norm((g / (1.0 + np.exp(-g))) * u)
                upd = Wd @ t
            else:
                Wq = to_f32('blk.%d.attn_q.weight' % b)
                Wk = to_f32('blk.%d.attn_k.weight' % b)
                Wv = to_f32('blk.%d.attn_v.weight' % b)
                Wo = to_f32('blk.%d.attn_output.weight' % b)
                Wgate = to_f32('blk.%d.attn_gate.weight' % b)
                sqk = scaled('blk.%d.attn_qk_scale.weight' % b, N_HEAD * HD,
                             lambda v: (v * (np.sqrt(HD) / INIT_STD)) ** 2)
                sgate = scaled('blk.%d.attn_gate_scale.weight' % b, N_HEAD * HD,
                               lambda v: (v * (np.sqrt(GATE_INIT) / INIT_STD)) ** 2)
                q = Wq @ x
                k = Wk @ x
                v = Wv @ x
                gate = Wgate @ x
                theta = 1e7 if profile[b] == 1 else 1e4
                q = rope(q, pi, theta, N_HEAD)
                k = rope(k, pi, theta, N_KV)
                q = l2norm_rows(q.reshape(N_HEAD, HD)).reshape(-1) * sqk
                k = l2norm_rows(k.reshape(N_KV, HD)).reshape(-1)
                kq = q8_deq(k)
                vq = q8_deq(v)
                key = b
                if key not in cache: cache[key] = ([], [])
                cache[key][0].append(kq)
                cache[key][1].append(vq)
                if pc == '*':
                    lo = 0
                else:
                    lo = max(0, pi + 1 - SWA)
                out = np.zeros((N_HEAD, HD), np.float32)
                for h in range(N_HEAD):
                    kh = h // (N_HEAD // N_KV)
                    ks = np.stack(cache[key][0][lo:pi + 1])[:, kh * HD:(kh + 1) * HD]
                    vs = np.stack(cache[key][1][lo:pi + 1])[:, kh * HD:(kh + 1) * HD]
                    sc = (ks @ q[h * HD:(h + 1) * HD]) * (1.0 / 16.0)
                    sc = sc - sc.max()
                    w = np.exp(sc)
                    w /= w.sum()
                    out[h] = w @ vs
                aout = out.reshape(-1)
                gt = gate * sgate
                attn_in = l2norm(aout / (1.0 + np.exp(-gt)))
                upd = Wo @ attn_in
                if b == debug_block and tapsdump:
                    taps = dict(q=(Wq @ x), k=(Wk @ x), v=(Wv @ x), gate=gate,
                                aout=aout, ain=attn_in, proj=upd)
            a = l2norm(resid)
            bb = l2norm(upd)
            x = l2norm(a + (bb - a) * alpha)
            if 16 <= vi < rec_end and (vi + 1 - 16) % 40 == 0:
                x = l2norm(x)
            if xdump:
                xd[vi] = x
    return x, xd, taps

if __name__ == '__main__':
    ids = [int(v) for v in sys.argv[1].split(',')]
    nlayers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    dbg = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    xpath = sys.argv[4] if len(sys.argv) > 4 else 'tmp/x1.bin'
    x, xd, taps = run(ids, nlayers, dbg, xdump=True, tapsdump=True)
    eng = None
    if os.path.exists(xpath):
        eng = np.fromfile(xpath, dtype=np.float32).reshape(-1, N_EMBD)
    if eng is not None:
        for vi in range(min(nlayers, eng.shape[0])):
            d = np.abs(eng[vi] - xd[vi])
            print('layer %3d (blk %2d %s): maxdiff %.6f rms %.6f  ref_norm %.5f eng_norm %.5f' % (
                vi, LAYER_ORDER[vi], pattern[LAYER_ORDER[vi]], d.max(), np.sqrt(np.mean(d * d)),
                np.linalg.norm(xd[vi]), np.linalg.norm(eng[vi])))
    if taps is not None and os.path.exists('tmp/taps0.bin'):
        buf = np.fromfile('tmp/taps0.bin', dtype=np.float32)
        TAP = dict(q=(0, 4096), k=(4096, 1024), v=(5120, 1024), gate=(6144, 4096),
                   aout=(10240, 4096), ain=(14336, 4096), proj=(18432, 2560))
        for name, a in taps.items():
            o, n = TAP[name]
            e = buf[o:o + n]
            d = np.abs(e - a)
            print('tap %-5s maxdiff %.6f rms %.6f ref_norm %.5f eng_norm %.5f' % (
                name, d.max(), np.sqrt(np.mean(d * d)), np.linalg.norm(a), np.linalg.norm(e)))
