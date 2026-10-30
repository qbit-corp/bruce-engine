"""Full decode benchmark + oracle equivalence check.

Runs the engine (GPU exclusive) on every prompt in tok.py, plain and speculative,
then starts the llama.cpp server once to (a) obtain the oracle greedy continuation
and (b) detokenize.  Prints a summary table.

Usage: python tools/bench_all.py [n_predict]
"""
import json, subprocess, time, urllib.request, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
ENGINE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
PORT = 18089
CTX = 4096

from tok import PROMPTS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
NAMES = list(PROMPTS.keys())


def run_engine(name, extra):
    r = subprocess.run([ENGINE, "--ctx", str(CTX), "--token-file", "tmp/ids_%s.txt" % name,
                        "--gen", str(N), "--greedy", "--print-ids"] + extra,
                       capture_output=True, text=True)
    ids = [int(x) for x in r.stdout.split()]
    d = {"ids": ids, "raw": r.stderr}
    for line in r.stderr.splitlines():
        if "gen " in line and "tok/s" in line:
            d["tps"] = float(line.split("(")[1].split(" tok/s")[0])
        if "prefill " in line and "tok/s" in line:
            d["prefill_tps"] = float(line.split("(")[1].split(" tok/s")[0])
        if "spec:" in line:
            d["spec"] = line.split("spec: ")[1]
            import re
            d["tpr"] = float(re.search(r"([\d.]+) tokens/round", line).group(1))
    return d


def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()


res = {}
for name in NAMES:
    print("=== %s ===" % name, flush=True)
    plain = run_engine(name, [])
    print("  plain %.2f tok/s (prefill %.0f tok/s)" % (plain["tps"], plain["prefill_tps"]), flush=True)
    spec = run_engine(name, ["--draft", "7"])
    print("  spec  %.2f tok/s  %s" % (spec["tps"], spec.get("spec", "")), flush=True)
    n = min(len(plain["ids"]), len(spec["ids"]))
    eq = sum(1 for i in range(n) if plain["ids"][i] == spec["ids"][i])
    print("  spec/plain token equality: %d/%d" % (eq, n), flush=True)
    res[name] = {"plain": plain, "spec": spec}

cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", str(CTX), "-fa", "on", "-np", "1",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
       "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
       "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", "2048", "-ub", "1024"]
log = open(os.path.join("tmp", "server_bench.log"), "wb")
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
    print("\nserver up in %.1fs" % (time.time() - t0), flush=True)

    print("\n%-6s %8s %8s %7s %9s %9s %s" %
          ("prompt", "plain", "spec", "tok/rnd", "spec/plain", "oracle", "gen tok/s"))
    for name in NAMES:
        resp = post("/completion", {"prompt": PROMPTS[name], "n_predict": N, "temperature": 0.0,
                                    "top_k": 1, "seed": 1, "cache_prompt": False, "n_probs": 1,
                                    "repeat_penalty": 1.0})
        srv_ids = [c["id"] for c in resp.get("completion_probabilities", [])]
        p = res[name]["plain"]["ids"]
        s = res[name]["spec"]["ids"]
        n1 = min(len(srv_ids), len(p))
        eq1 = sum(1 for i in range(n1) if srv_ids[i] == p[i])
        n2 = min(len(srv_ids), len(s))
        eq2 = sum(1 for i in range(n2) if srv_ids[i] == s[i])
        print("%-6s %8.2f %8.2f %7.2f %9s %9s  %.2f" %
              (name, res[name]["plain"]["tps"], res[name]["spec"]["tps"],
               res[name]["spec"].get("tpr", 0.0),
               "%d/%d" % (eq1, n1), "%d/%d" % (eq2, n2),
               res[name]["plain"]["tps"]), flush=True)
        open("tmp/out_%s_plain.txt" % name, "w").write(" ".join(map(str, p)))
        open("tmp/out_%s_spec.txt" % name, "w").write(" ".join(map(str, s)))

    for name in NAMES:
        p = res[name]["plain"]["ids"]
        print("\n=== %s (engine plain) ===" % name)
        print(post("/detokenize", {"tokens": p}).get("content", "")[:500])
finally:
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()
    log.close()
