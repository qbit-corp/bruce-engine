"""Reference decode speed at depth, measured on the llama.cpp PR 30306 server.

Starts the same server as run_dory.bat (same flags, separate port), then for each
depth D sends a single /completion request whose prompt is exactly D token ids and
asks for 128 tokens.  The server reports prefill and generation timing separately,
so `predicted_per_second` is the batch-1 decode speed at that KV depth.
"""
import json, os, subprocess, sys, time, urllib.request

SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get("BRUCE_MODEL", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
PORT = 18091
DEPTHS = [16384, 32768, 65536, 131072]
NP = 128


def post(path, obj, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()


def main():
    ids = [int(x) for x in open("tmp/ids_depth.txt").read().split()]
    assert len(ids) >= max(DEPTHS), len(ids)
    cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", "262144", "-fa", "on", "-np", "1",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "-b", "2048", "-ub", "1024",
           "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
           "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
           "--load-mode", "none", "-t", "1", "--cache-ram", "0",
           "--host", "127.0.0.1", "--port", str(PORT), "--no-webui"]
    log = open("tmp/server_depth.log", "wb")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        t0 = time.time()
        while True:
            try:
                get("/health")
                break
            except Exception:
                if time.time() - t0 > 600:
                    raise RuntimeError("server did not start")
                time.sleep(1.0)
        print("server up in %.1fs" % (time.time() - t0), flush=True)
        for d in DEPTHS:
            t0 = time.time()
            r = post("/completion", {"prompt": ids[:d], "n_predict": NP, "temperature": 0.0,
                                     "top_k": 1, "ignore_eos": True, "cache_prompt": False,
                                     "n_probs": 0})
            tm = r.get("timings", {})
            rows.append((d, tm.get("prompt_n"), tm.get("prompt_per_second"),
                         tm.get("predicted_n"), tm.get("predicted_per_second")))
            print("depth %6d  prompt %6d @ %7.1f tok/s   gen %3d @ %6.2f tok/s (%.2f ms/tok)  wall %.0fs"
                  % (d, tm.get("prompt_n", 0), tm.get("prompt_per_second", 0.0),
                     tm.get("predicted_n", 0), tm.get("predicted_per_second", 0.0),
                     1000.0 / max(1e-9, tm.get("predicted_per_second", 1.0)), time.time() - t0),
                  flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=15)
        except Exception:
            srv.kill()
        log.close()
    with open("tmp/ref_depth.json", "w") as f:
        json.dump([{"depth": d, "prompt_n": pn, "prefill_tok_s": ps, "gen_n": gn, "gen_tok_s": gs}
                   for d, pn, ps, gn, gs in rows], f, indent=1)


main()
