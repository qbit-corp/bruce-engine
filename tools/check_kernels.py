"""Model-level A/B gate for the load-time Q5_K pre-unpack.

Runs the same 16K prompt through the engine twice -- raw Q5_K MMQ vs the
pre-unpacked 144 B kernel -- dumps the final logits, and reports the same checks
used in README section 4b: argmax, top-5 / top-50 overlap and max |delta logit|.

Usage (GPU exclusive, from the repository root):
    python tools/check_kernels.py [--ctx 16384] [--prompt tmp/ids_16k.txt]
"""
import argparse
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine", "bruce.exe")


def slice_prompt(path, n):
    if os.path.exists(path):
        return path
    src = os.path.join(ROOT, "tmp", "ids_codereal32k.txt")
    ids = open(src).read().split()[:n]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write("\n".join(ids))
    return path


def run(prompt, ctx, out, extra):
    cmd = [ENGINE, "--ctx", str(ctx), "--token-file", prompt, "--gen", "0",
           "--no-live", "--dump-logits", out] + extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("engine failed: %s" % " ".join(cmd))
    return r.stderr


def load(p):
    b = open(p, "rb").read()
    return struct.unpack("<%df" % (len(b) // 4), b)


def top(v, k):
    return sorted(range(len(v)), key=lambda i: -v[i])[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--prompt", default=os.path.join(ROOT, "tmp", "ids_16k.txt"))
    args = ap.parse_args()
    prompt = slice_prompt(args.prompt, args.ctx)
    ref = os.path.join(ROOT, "tmp", "chk_ref.bin")
    new = os.path.join(ROOT, "tmp", "chk_new.bin")

    print("reference: --q5k-prepack off (raw 176 B MMQ)")
    print(run(prompt, args.ctx, ref, ["--q5k-prepack", "off"]).strip().splitlines()[-1])
    print("candidate: --q5k-prepack on  (144 B pre-unpack)")
    print(run(prompt, args.ctx, new, ["--q5k-prepack", "on"]).strip().splitlines()[-1])

    a, b = load(ref), load(new)
    if len(a) != len(b):
        raise SystemExit("logit vector size mismatch")
    md = max(abs(x - y) for x, y in zip(a, b))
    print("max |delta logit| %.6f" % md)
    print("argmax %d / %d  %s" % (top(a, 1)[0], top(b, 1)[0],
                                  "MATCH" if top(a, 1) == top(b, 1) else "DIFFER"))
    for k in (5, 50):
        o = len(set(top(a, k)) & set(top(b, k)))
        print("top-%d overlap %d/%d" % (k, o, k))
    ok = top(a, 1) == top(b, 1) and len(set(top(a, 5)) & set(top(b, 5))) >= 4 \
        and len(set(top(a, 50)) & set(top(b, 50))) >= 49
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
