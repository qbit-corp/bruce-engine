import json, subprocess, time, urllib.request, sys, os

SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get("BRUCE_MODEL", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
PORT = int(sys.argv[1])
UB = int(sys.argv[2])
TEXT = open(sys.argv[3], encoding='utf-8').read()

def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()

cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", "8192", "-fa", "on", "-np", "1",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
       "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
       "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", str(max(UB, 2)), "-ub", str(UB)]
log = open(os.path.join("tmp", "srv_ub%d.log" % UB), "wb")
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
    resp = post("/completion", {"prompt": TEXT, "n_predict": 1, "temperature": 0.0,
                                "logprobs": 5, "top_k": 1, "seed": 1, "cache_prompt": False})
    cp = resp["completion_probabilities"]
    t = resp.get("timings", {})
    tops = [(x["id"], round(x["logprob"], 5)) for x in cp[-1]["top_logprobs"]]
    print("ub=%d tokens=%d prompt_ms=%.1f tops=%s" % (
        UB, t.get("prompt_n"), t.get("prompt_ms", -1), tops), flush=True)
finally:
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()
    log.close()
