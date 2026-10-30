import json, subprocess, struct, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
GGUF = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
oracle = json.load(open("tmp/prefix_oracle.json"))
os.makedirs("tmp", exist_ok=True)

for p, o in oracle.items():
    ids = o["ids"]
    out = os.path.abspath("tmp/lg_cmp.bin")
    r = subprocess.run([ENGINE, "--gguf", GGUF, "--ctx", "8192",
                        "--tokens", ",".join(map(str, ids)), "--dump-logits", out,
                        "--gen", "0", "--top-logits", "0"], capture_output=True, text=True)
    if r.returncode != 0:
        print("engine failed", r.stdout, r.stderr[-1500:])
        sys.exit(1)
    lg = struct.unpack("<%df" % 131072, open(out, "rb").read())
    mx = max(lg)
    lse = math.log(sum(math.exp(v - mx) for v in lg)) + mx
    order = sorted(range(131072), key=lambda i: -lg[i])[:5]
    srv = {t[0]: t[1] for t in o["tops"]}
    rows = [(i, round(lg[i] - lse, 4), srv.get(i)) for i in order]
    print("PREFIX %r" % p)
    for i, v, s in rows:
        print("   id %6d engine %8.4f oracle %s delta %s" % (
            i, v, ("%8.4f" % s) if s is not None else "      --",
            ("%+.4f" % (v - s)) if s is not None else "   --"))
