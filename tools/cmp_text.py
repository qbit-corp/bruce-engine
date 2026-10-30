"""Print the oracle completion and the engine's saved completion side by side."""
import json, subprocess, time, urllib.request, os, sys

SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get("BRUCE_MODEL", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
PORT = 18090
from tok import PROMPTS

names = sys.argv[1:] or list(PROMPTS.keys())
N = 160


def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()


cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", "4096", "-fa", "on", "-np", "1",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
       "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
       "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", "2048", "-ub", "1024"]
log = open(os.path.join("tmp", "server_text.log"), "wb")
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
    for name in names:
        resp = post("/completion", {"prompt": PROMPTS[name], "n_predict": N, "temperature": 0.0,
                                    "top_k": 1, "seed": 1, "cache_prompt": False, "n_probs": 1,
                                    "repeat_penalty": 1.0})
        srv_ids = [c["id"] for c in resp.get("completion_probabilities", [])]
        print("################ %s" % name)
        print("--- oracle (%d tok) ---" % len(srv_ids))
        print(repr(resp.get("content", ""))[:1200])
        try:
            eng_ids = [int(x) for x in open("tmp/out_%s_plain.txt" % name).read().split()]
            print("--- engine (%d tok) ---" % len(eng_ids))
            print(repr(post("/detokenize", {"tokens": eng_ids}).get("content", ""))[:1200])
        except FileNotFoundError:
            print("(no engine output saved)")
        n = min(len(srv_ids), len(eng_ids)) if 'eng_ids' in dir() else 0
        if n:
            first = next((i for i in range(n) if srv_ids[i] != eng_ids[i]), None)
            print("--- first token diff: %s ---" % first)
            if first is not None:
                print("   oracle:", srv_ids[first:first + 12])
                print("   engine:", eng_ids[first:first + 12])
                print("   oracle top:", [(c["id"], round(c["logprob"], 4)) for c in resp["completion_probabilities"][first]["top_logprobs"]])
finally:
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()
    log.close()
