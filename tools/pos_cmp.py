"""Compare full logit vectors at a given generation index between the engine and the
sequential llama.cpp oracle dumper.

Sequence compared: prompt + first k generated tokens (which are identical in both).
The oracle dumps raw logits for every prefix token; the engine's "--dump-logits x.late"
after `--gen k+1` holds the distribution that predicts token k.

Usage: python tools/pos_cmp.py <prompt-name> <k>
"""
import os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
ENGINE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
ORACLE = os.path.join(ROOT, "tools", "oracle_dump.exe" if os.name == "nt" else "oracle_dump")
BIN = os.environ.get("LLAMA_BIN", os.path.join(ROOT, "tools", "doryref", "build-cuda", "bin"))
VOCAB = 131072

name = sys.argv[1]
k = int(sys.argv[2])

prompt = [int(x) for x in open("tmp/ids_%s.txt" % name).read().replace(",", " ").split()]
gen = [int(x) for x in open("tmp/out_%s_plain.txt" % name).read().split()]
seq = prompt + gen[:k]

env = dict(os.environ)
env["PATH"] = BIN + ";" + env["PATH"]
r = subprocess.run([ORACLE, MODEL, "tmp/pos_oracle.bin", ",".join(map(str, seq))],
                   capture_output=True, text=True, env=env)
if r.returncode != 0:
    print(r.stdout[-2000:], r.stderr[-2000:])
    sys.exit(1)

raw = open("tmp/pos_oracle.bin", "rb").read()
assert raw[:8] == b"DORYDUMP"
ntok, nvirt, nembd, nvocab = np.frombuffer(raw[8:24], dtype=np.int32)
off = 24 + ntok * nvirt * nembd * 4
logs = np.frombuffer(raw[off:off + ntok * nvocab * 4], dtype=np.float32).reshape(int(ntok), int(nvocab))
ora = logs[-1].copy()

r2 = subprocess.run([ENGINE, "--ctx", "4096", "--token-file", "tmp/ids_%s.txt" % name,
                     "--gen", str(k + 1), "--greedy", "--print-ids",
                     "--dump-logits", "tmp/pos_eng.bin"], capture_output=True, text=True)
eng_ids = [int(x) for x in r2.stdout.split()]
assert eng_ids[:k] == gen[:k], (eng_ids[:k], gen[:k])
eng = np.fromfile("tmp/pos_eng.bin.late", dtype=np.float32)

d = eng - ora
dc = d - d.mean()
print("engine vs oracle logits at position %d (after %d generated tokens)" % (k, k))
print("  raw diff: mean %+.4f  std %.4f  max|d| %.4f" % (d.mean(), d.std(), np.abs(d).max()))
print("  centered: std %.4f  max|d| %.4f" % (dc.std(), np.abs(dc).max()))
cos = float(eng @ ora / (np.linalg.norm(eng) * np.linalg.norm(ora)))
print("  cosine %.8f" % cos)
to = np.argsort(-ora)[:5]
te = np.argsort(-eng)[:5]
print("  oracle top5:", [(int(i), round(float(ora[i]), 4)) for i in to])
print("  engine top5:", [(int(i), round(float(eng[i]), 4)) for i in te])
print("  oracle gaps to top1:", [round(float(ora[to[0]] - ora[i]), 4) for i in to])
print("  engine gaps to top1:", [round(float(eng[te[0]] - eng[i]), 4) for i in te])
rank = int((ora > ora[te[0]]).sum())
print("  engine top1 rank in oracle: %d (oracle logit %.4f, top %.4f)"
      % (rank, ora[te[0]], ora[to[0]]))
print("  argmax equal:", int(te[0]) == int(to[0]))
