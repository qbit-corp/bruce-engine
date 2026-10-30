import json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18100

cmd = [ENGINE, "--model", MODEL, "--serve", "--host", "127.0.0.1", "--port", str(PORT),
       "--ctx", "262144", "--loops", "2", "--n-batch", "512", "--n-predict", "-1",
       "--draft", "0", "--thinking", "--no-live"]
print(" ".join(cmd), flush=True)
log = open(os.path.join(ROOT, "bench", "smoke_bruce.log"), "wb")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()


def post(path, obj, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


try:
    t0 = time.time()
    while True:
        try:
            print("health:", get("/health"), flush=True)
            break
        except Exception:
            if time.time() - t0 > 180:
                raise
            time.sleep(1)
    print("up in %.1fs" % (time.time() - t0), flush=True)
    ids = post("/tokenize", {"content": "The capital of France is", "add_special": False})["tokens"]
    ids = [t["id"] if isinstance(t, dict) else int(t) for t in ids]
    print("ids", ids, flush=True)
    r = post("/completion", {"prompt": ids, "n_predict": 16, "temperature": 0, "cache_prompt": False,
                             "ignore_eos": True})
    print(json.dumps({k: r.get(k) for k in ("tokens_evaluated", "tokens_predicted", "tokens_cached",
                                             "timings", "generation_settings")}, indent=2), flush=True)
    print("content:", repr(r.get("content", ""))[:300], flush=True)
finally:
    srv.terminate()
    try:
        srv.wait(timeout=20)
    except Exception:
        srv.kill()
    log.close()
