import json, subprocess, time, urllib.request, sys, os, struct, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
ENGINE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
PORT = 18086
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
TEXT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
UB = int(sys.argv[4]) if len(sys.argv) > 4 else 1024

def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()

cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", str(CTX), "-fa", "on", "-np", "1",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
       "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
       "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", str(BATCH), "-ub", str(UB)]
print("starting server...", flush=True)
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_validate.log"), "wb")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
try:
    t0 = time.time()
    while True:
        try:
            get("/health")
            break
        except Exception:
            if time.time() - t0 > 300:
                raise RuntimeError("server did not start")
            time.sleep(1.0)
    print("server up in %.1fs" % (time.time() - t0), flush=True)

    tok = post("/tokenize", {"content": TEXT})
    ids = tok["tokens"]
    print("tokens:", ids, tok.get("pieces", [])[:20], flush=True)

    resp = post("/completion", {"prompt": TEXT, "n_predict": 1, "temperature": 0.0,
                                "logprobs": 5, "top_k": 1, "seed": 1, "cache_prompt": False})
    cp = resp.get("completion_probabilities", [])
    if not cp:
        print(json.dumps(resp)[:2000])
        sys.exit(1)
    tops = cp[-1]["top_logprobs"]
    print("server top5:", [(t["id"], round(t["logprob"], 4), t.get("token")) for t in tops], flush=True)

    os.makedirs("tmp", exist_ok=True)
    out = os.path.join(ROOT, "tmp", "logits.bin")
    args = [ENGINE, "--ctx", str(min(CTX, 8192)), "--tokens", ",".join(map(str, ids)),
            "--dump-logits", out, "--top-logits", "5", "--gen", "0"]
    r = subprocess.run(args, capture_output=True, text=True)
    print(r.stdout, flush=True)
    print(r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        sys.exit(1)

    lg = struct.unpack("<%df" % (131072), open(out, "rb").read())
    mx = max(lg)
    z = sum(math.exp(v - mx) for v in lg)
    lse = math.log(z) + mx
    order = sorted(range(131072), key=lambda i: -lg[i])[:5]
    srvmap = {t["id"]: t["logprob"] for t in tops}
    print("engine top5:", [(i, round(lg[i] - lse, 4)) for i in order], flush=True)
    print("delta:", [(i, round((lg[i] - lse) - srvmap[i], 4) if i in srvmap else None) for i in order], flush=True)
finally:
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()
    log.close()
