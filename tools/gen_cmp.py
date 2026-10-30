"""Compare greedy generation of the engine against llama.cpp server on the same prompt.

Order matters: the engine runs FIRST (GPU exclusive), then the oracle server is started,
so both never share the GPU.

Usage: python tools/gen_cmp.py <prompt-name> [n_predict] [--spec]
Reads tmp/ids_<name>.txt for the prompt tokens, whose source text lives in tools/tok.py.
"""
import json, subprocess, time, urllib.request, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
ENGINE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
PORT = 18088

from tok import PROMPTS  # reuse the same prompt texts

name = sys.argv[1] if len(sys.argv) > 1 else "code"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
SPEC = "--spec" in sys.argv


def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()


def run_engine(extra):
    r = subprocess.run([ENGINE, "--ctx", "4096", "--token-file", "tmp/ids_%s.txt" % name,
                        "--gen", str(N), "--greedy", "--print-ids"] + extra,
                       capture_output=True, text=True)
    ids = [int(x) for x in r.stdout.split()]
    lines = [l for l in r.stderr.splitlines() if "gen " in l or "prefill" in l or "spec:" in l]
    return ids, lines


t_eng = time.time()
eng_plain, log_plain = run_engine([])
print("engine plain :", log_plain)
if SPEC:
    eng_spec, log_spec = run_engine(["--draft", "7"])
    print("engine spec  :", log_spec)
    n = min(len(eng_plain), len(eng_spec))
    same = sum(1 for i in range(n) if eng_plain[i] == eng_spec[i])
    first = next((i for i in range(n) if eng_plain[i] != eng_spec[i]), None)
    print("spec vs plain: %d/%d equal, first diff at %s" % (same, n, first))

cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", "4096", "-fa", "on", "-np", "1",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
       "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
       "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", "2048", "-ub", "1024"]
log = open(os.path.join("tmp", "server_gen.log"), "wb")
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

    tok = post("/tokenize", {"content": PROMPTS[name], "add_special": False})
    srv_prompt = [t["id"] if isinstance(t, dict) else int(t) for t in tok["tokens"]]
    eng_prompt = [int(x) for x in open("tmp/ids_%s.txt" % name).read().replace(",", " ").split()]
    print("prompt tokens: %d (engine) vs %d (server), equal=%s"
          % (len(eng_prompt), len(srv_prompt), eng_prompt == srv_prompt))

    resp = post("/completion", {"prompt": PROMPTS[name], "n_predict": N, "temperature": 0.0,
                                "top_k": 1, "seed": 1, "cache_prompt": False, "n_probs": 1,
                                "repeat_penalty": 1.0})
    cp = resp.get("completion_probabilities", [])
    srv_ids = [c["id"] for c in cp]

    n = min(len(srv_ids), len(eng_plain))
    same = sum(1 for i in range(n) if srv_ids[i] == eng_plain[i])
    first = next((i for i in range(n) if srv_ids[i] != eng_plain[i]), None)
    print("plain vs server: %d/%d equal, first diff at %s" % (same, n, first))
    if SPEC:
        n2 = min(len(srv_ids), len(eng_spec))
        same2 = sum(1 for i in range(n2) if srv_ids[i] == eng_spec[i])
        first2 = next((i for i in range(n2) if srv_ids[i] != eng_spec[i]), None)
        print("spec  vs server: %d/%d equal, first diff at %s" % (same2, n2, first2))

    print("\n=== server text ===")
    print(resp.get("content", "")[:700])
    print("\n=== engine (plain) text ===")
    print(post("/detokenize", {"tokens": eng_plain}).get("content", "")[:700])
    if SPEC:
        print("\n=== engine (spec) text ===")
        print(post("/detokenize", {"tokens": eng_spec}).get("content", "")[:700])
finally:
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()
    log.close()
