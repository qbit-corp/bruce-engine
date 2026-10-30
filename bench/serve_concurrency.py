"""Concurrency regression for the bruce server's multi-slot (continuous batching) mode.

Starts the server, then runs N identical-length code prompts in parallel for
N = 1, 2, 4, 8 and reports per-request and aggregate decode throughput.  The
single-request rows are the "did batching hurt solo speed" check; the multi rows
are the batching win.

    python bench/serve_concurrency.py [--ctx 32768] [--gen 256] [--depth 512]

Requires the GGUF at the usual path (override with --model).
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRUCE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
DEFAULT_MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
PORT = 18240


def post(obj, timeout=3600):
    data = json.dumps(obj, separators=(",", ":")).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health():
    with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=5) as r:
        return json.loads(r.read())


def start(model, ctx, slots, log_path):
    log = open(log_path, "wb")
    p = subprocess.Popen([BRUCE, "--model", model, "--serve", "--host", "127.0.0.1",
                          "--port", str(PORT), "--ctx", str(ctx), "--slots", str(slots),
                          "--n-batch", "512", "--n-predict", "-1", "--no-live"],
                         stdout=log, stderr=log)
    for _ in range(900):
        try:
            health()
            return p, log
        except Exception:
            time.sleep(0.5)
    p.kill()
    raise SystemExit("server did not start (see %s)" % log_path)


def stop(p, log):
    p.terminate()
    try:
        p.wait(timeout=30)
    except Exception:
        p.kill()
    log.close()
    time.sleep(3)


def run(prompts):
    out = [None] * len(prompts)

    def one(i):
        out[i] = post({"prompt": prompts[i], "n_predict": ARGS.gen, "temperature": 0.0,
                       "cache_prompt": False, "ignore_eos": True})

    ths = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    total = sum(b["tokens_predicted"] for b in out)
    per = [b["timings"]["predicted_per_second"] for b in out]
    pre = [b["timings"]["prompt_per_second"] for b in out]
    return wall, total, per, pre


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--depth", type=int, default=512)
    ARGS = ap.parse_args()

    ids = json.load(open(os.path.join(ROOT, "bench", "corpus_code_ids.json")))
    prompts = [ids[i * (ARGS.depth + 100):(i * (ARGS.depth + 100)) + ARGS.depth]
               for i in range(ARGS.slots)]
    p, log = start(ARGS.model, ARGS.ctx, ARGS.slots,
                   os.path.join(ROOT, "tmp", "serve_concurrency.log"))
    try:
        print("ctx %d (%d/slot), prompt %d tokens, gen %d, greedy"
              % (ARGS.ctx, ARGS.ctx // ARGS.slots, ARGS.depth, ARGS.gen))
        print("  n   wall    aggregate   per-request decode (tok/s)   prefill (tok/s, queued)")
        for n in (1, 2, 4, 8):
            if n > ARGS.slots:
                continue
            wall, total, per, pre = run(prompts[:n])
            print("%3d  %5.2fs  %7.1f     %-28s  %.0f"
                  % (n, wall, total / wall,
                     " ".join("%.1f" % x for x in per), sum(pre) / len(pre)))
    finally:
        stop(p, log)
