import json, os, subprocess, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "bench")
SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
PORT = 18102

TEXT_SUFFIX = ("\n\nWrite the next section in the same style: a vivid, carefully paced essay about "
               "how ordinary people experienced the fall of the Roman Empire. About 500 words.\n")
CODE_SUFFIX = ("\n\n// Task: implement a lock-free single-producer single-consumer ring buffer in modern C++. "
               "Provide the complete header and explain the memory-ordering choices.\n")

cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", "4096", "-fa", "on", "-np", "1",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
       "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
       "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", "2048", "-ub", "1024",
       "--load-mode", "none", "-t", "1", "--cache-ram", "0"]
print(" ".join(cmd), flush=True)
log = open(os.path.join(BENCH, "tokenize.log"), "wb")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()


def post(path, obj, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ids_of(text):
    r = post("/tokenize", {"content": text, "add_special": False, "parse_special": True})
    return [t["id"] if isinstance(t, dict) else int(t) for t in r["tokens"]]


try:
    t0 = time.time()
    while True:
        try:
            get("/health")
            break
        except Exception:
            if time.time() - t0 > 180:
                raise
            time.sleep(1)
    print("server up in %.1fs" % (time.time() - t0), flush=True)
    for name, path in [("text", "text_corpus.txt"), ("code", "code_corpus.txt")]:
        text = open(os.path.join(BENCH, path), encoding="utf-8").read()
        print("tokenizing %s (%d chars)..." % (name, len(text)), flush=True)
        t0 = time.time()
        ids = ids_of(text)
        print("  %d tokens in %.1fs" % (len(ids), time.time() - t0), flush=True)
        with open(os.path.join(BENCH, "corpus_%s_ids.json" % name), "w") as f:
            json.dump(ids, f)
    for name, suffix in [("text", TEXT_SUFFIX), ("code", CODE_SUFFIX)]:
        ids = ids_of(suffix)
        print("%s suffix: %d tokens" % (name, len(ids)), flush=True)
        with open(os.path.join(BENCH, "%s_suffix_ids.json" % name), "w") as f:
            json.dump(ids, f)
        print("  first ids:", ids[:20], flush=True)
finally:
    srv.terminate()
    try:
        srv.wait(timeout=20)
    except Exception:
        srv.kill()
    log.close()
