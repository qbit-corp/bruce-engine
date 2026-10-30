import os
import sys, struct, collections

P = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BRUCE_MODEL", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

class R:
    def __init__(s, f): s.f = f
    def u32(s): return struct.unpack("<I", s.f.read(4))[0]
    def u64(s): return struct.unpack("<Q", s.f.read(8))[0]
    def i32(s): return struct.unpack("<i", s.f.read(4))[0]
    def i64(s): return struct.unpack("<q", s.f.read(8))[0]
    def f32(s): return struct.unpack("<f", s.f.read(4))[0]
    def f64(s): return struct.unpack("<d", s.f.read(8))[0]
    def s(s):
        n = s.u64(); b = s.f.read(n); return b.decode("utf-8", "replace")
    def val(s, t):
        if t == T_U8: return struct.unpack("<B", s.f.read(1))[0]
        if t == T_I8: return struct.unpack("<b", s.f.read(1))[0]
        if t == T_U16: return struct.unpack("<H", s.f.read(2))[0]
        if t == T_I16: return struct.unpack("<h", s.f.read(2))[0]
        if t == T_U32: return s.u32()
        if t == T_I32: return s.i32()
        if t == T_F32: return s.f32()
        if t == T_BOOL: return bool(s.f.read(1)[0])
        if t == T_STR: return s.s()
        if t == T_U64: return s.u64()
        if t == T_I64: return s.i64()
        if t == T_F64: return s.f64()
        if t == T_ARR:
            at, n = s.u32(), s.u64()
            return [s.val(at) for _ in range(n)]
        raise ValueError(t)

f = open(P, "rb")
r = R(f)
assert f.read(4) == b"GGUF"
ver = r.u32(); nt = r.u64(); nkv = r.u64()
print("version", ver, "tensors", nt, "kv", nkv)
kv = {}
for _ in range(nkv):
    k = r.s(); t = r.u32(); kv[k] = r.val(t)
for k in sorted(kv):
    if not isinstance(kv[k], list) or len(kv[k]) < 12:
        print(" ", k, "=", kv[k])

TS = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
      10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
      17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
      23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
      30: "BF16", 34: "TQ1_0", 35: "TQ2_0"}

# block sizes / bytes for type sizing (n elements must be divisible)
BS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
      9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
      15: (256, 292), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 30: (1, 2)}

tens = []
for _ in range(nt):
    name = r.s(); nd = r.u32()
    dims = [r.u64() for _ in range(nd)]
    t = r.u32(); off = r.u64()
    tens.append((name, dims, t, off))

by_type = collections.Counter()
byname = collections.Counter()
for name, dims, t, off in tens:
    n = 1
    for d in dims: n *= d
    bs, bb = BS[t]
    b = n // bs * bb
    by_type[TS.get(t, t)] += b
    byname[name.rsplit(".", 1)[0]] += b

tot = sum(by_type.values())
print("\nby type:")
for k, v in by_type.most_common():
    print("  %-8s %12d  %6.2f MB  %5.1f%%" % (k, v, v / 1e6, 100.0 * v / tot))
print("  TOTAL    %12d  %6.2f MB" % (tot, tot / 1e6))
print("\nby tensor group:")
for k, v in byname.most_common(12):
    print("  %-28s %10.2f MB" % (k, v / 1e6))
