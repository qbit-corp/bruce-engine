"""Final benchmark + validation suite.

Phase 1 (engine only, GPU exclusive):
  - plain greedy and speculative greedy on every prompt
  - temperature/top-p sampling run on two prompts
Phase 2 (llama.cpp server once):
  - oracle greedy continuation for every prompt -> token-level equality
  - detokenize engine outputs
"""
import json, re, subprocess, time, urllib.request, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
ENGINE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
PORT = 18091
CTX = 4096
N = 128
NAMES = ["code", "chat", "prose", "long"]
from tok import PROMPTS


def run_engine(name, extra, gen=N):
    r = subprocess.run([ENGINE, "--ctx", str(CTX), "--token-file", "tmp/ids_%s.txt" % name,
                        "--gen", str(gen), "--print-ids"] + extra, capture_output=True, text=True)
    ids = [int(x) for x in r.stdout.split()]
    d = {"ids": ids}
    for line in r.stderr.splitlines():
        m = re.search(r"gen \d+ tokens in [\d.]+ ms \(([\d.]+) tok/s\)", line)
        if m:
            d["tps"] = float(m.group(1))
        m = re.search(r"([\d.]+) tokens/round", line)
        if m:
            d["tpr"] = float(m.group(1))
        if "spec:" in line:
            d["spec"] = line.split("spec: ")[1]
        m = re.search(r"prefill \d+ tokens in [\d.]+ ms \(([\d.]+) tok/s\)", line)
        if m:
            d["prefill"] = float(m.group(1))
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
    plain = run_engine(name, ["--greedy"])
    spec = run_engine(name, ["--greedy", "--draft", "7"])
    res[name] = {"plain": plain, "spec": spec}
    n = min(len(plain["ids"]), len(spec["ids"]))
    eq = sum(1 for i in range(n) if plain["ids"][i] == spec["ids"][i])
    print("%-6s plain %6.2f tok/s | spec %6.2f tok/s (%.2f tok/round) | spec==plain %d/%d"
          % (name, plain["tps"], spec["tps"], spec.get("tpr", 0), eq, n), flush=True)

samp = {}
for name in ["chat", "prose"]:
    s = run_engine(name, ["--temp", "0.8", "--top-p", "0.9"], gen=96)
    samp[name] = s
    print("%-6s temp0.8/topp0.9 %6.2f tok/s, %d tokens" % (name, s["tps"], len(s["ids"])), flush=True)

# 16K breakthrough benchmark: prefill rate and the 4-row verify round.
p16 = os.path.join("tmp", "ids_16k.txt")
if not os.path.exists(p16):
    src = open(os.path.join("tmp", "ids_codereal32k.txt")).read().split()
    open(p16, "w").write("\n".join(src[:16384]))
r = subprocess.run([ENGINE, "--ctx", "16384", "--token-file", p16, "--gen", "0", "--no-live"],
                   capture_output=True, text=True)
m = re.search(r"prefill \d+ tokens in [\d.]+ ms \(([\d.]+) tok/s\)", r.stderr)
pref16 = float(m.group(1)) if m else 0.0
r = subprocess.run([ENGINE, "--ctx", "16384", "--bench", "20", "--bench-at", "16384",
                    "--bench-rows", "4"], capture_output=True, text=True)
m = re.search(r"([\d.]+) ms/round", r.stdout)
round4 = float(m.group(1)) if m else 0.0
print("16K    prefill %6.2f tok/s | 4-row round %.2f ms (%.1f tok/s at 1.43 tok/round)"
      % (pref16, round4, 1430.0 / round4 if round4 else 0.0), flush=True)

cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", str(CTX), "-fa", "on", "-np", "1",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
       "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
       "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", "2048", "-ub", "1024"]
log = open(os.path.join("tmp", "server_final.log"), "wb")
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

    print("\n%-6s %8s %8s %7s %12s %12s" % ("prompt", "plain", "spec", "tok/rnd", "plain==oracle", "spec==oracle"))
    for name in NAMES:
        resp = post("/completion", {"prompt": PROMPTS[name], "n_predict": N, "temperature": 0.0,
                                    "top_k": 1, "seed": 1, "cache_prompt": False, "n_probs": 1,
                                    "repeat_penalty": 1.0})
        srv_ids = [c["id"] for c in resp.get("completion_probabilities", [])]
        p = res[name]["plain"]["ids"]
        s = res[name]["spec"]["ids"]
        def eqn(a, b):
            n = min(len(a), len(b))
            return "%d/%d" % (sum(1 for i in range(n) if a[i] == b[i]), n)
        print("%-6s %8.2f %8.2f %7.2f %12s %12s" %
              (name, res[name]["plain"]["tps"], res[name]["spec"]["tps"],
               res[name]["spec"].get("tpr", 0), eqn(p, srv_ids), eqn(s, srv_ids)), flush=True)
        open("tmp/out_%s_plain.txt" % name, "w").write(" ".join(map(str, p)))
        open("tmp/out_%s_spec.txt" % name, "w").write(" ".join(map(str, s)))

    print("\n================ generated text (plain greedy) ================")
    for name in NAMES:
        print("\n--- %s ---" % name)
        print(post("/detokenize", {"tokens": res[name]["plain"]["ids"]}).get("content", "")[:420])
    print("\n================ sampled text (temp 0.8, top-p 0.9) ================")
    for name in samp:
        print("\n--- %s ---" % name)
        print(post("/detokenize", {"tokens": samp[name]["ids"]}).get("content", "")[:420])
finally:
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()
    log.close()
