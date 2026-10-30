"""Fair Bruce vs llama.cpp benchmark on the same GGUF, 262144-token context.

Both engines are driven through their own OpenAI/llama.cpp-compatible HTTP servers.
The prompt is a token-id array, so both receive exactly the same tokens.
Primary comparison: greedy decode, no speculation, cache_prompt=false.
Secondary: ngram/prompt-lookup speculation with a comparable draft length.
"""
import argparse, json, os, platform, statistics, subprocess, sys, time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "bench")
LOGS = os.path.join(BENCH, "logs")
MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
BRUCE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
LLAMA = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
CTX = 262144
PORT_BRUCE = 18110
PORT_LLAMA = 18111
N_PREDICT = 256

DEPTHS = [(4096, 3), (32768, 2), (256000, 1)]
SPEC_DEPTHS = [(4096, 2), (32768, 2)]
PROMPTS = ["text", "code"]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm,"
             "temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
        vals = [x.strip() for x in out.splitlines()[0].split(",")]
        return dict(mem_used_mib=int(float(vals[0])), mem_total_mib=int(float(vals[1])),
                    util=int(float(vals[2])), clock_mhz=int(float(vals[3])),
                    temp_c=int(float(vals[4])), power_w=float(vals[5]))
    except Exception as e:
        return {"error": str(e)}


def wait_vram_free(threshold=1500, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        g = gpu_info()
        if "error" in g or g["mem_used_mib"] <= threshold:
            return True
        time.sleep(2)
    return False


def get_json(port, path, timeout=60):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=timeout) as r:
        return json.loads(r.read())


def post_json(port, path, obj, timeout=3600):
    data = json.dumps(obj, separators=(",", ":")).encode()
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    wall = time.perf_counter() - t0
    return json.loads(body), wall, len(data)


def bruce_cmd(port, draft):
    return [BRUCE, "--model", MODEL, "--serve", "--host", "127.0.0.1", "--port", str(port),
            "--ctx", str(CTX), "--loops", "2", "--n-batch", "512", "--n-predict", "-1",
            "--slots", "1", "--draft", str(draft), "--thinking",
            "--temp", "1.0", "--top-p", "0.95", "--top-k", "0", "--min-p", "0.0",
            "--repeat-penalty", "1.0", "--repeat-last-n", "64"]


def llama_cmd(port, spec=False, ubatch=1024):
    cmd = [LLAMA, "-m", MODEL, "-ngl", "999", "-c", str(CTX), "-fa", "on", "--jinja",
           "-np", "1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "-b", "2048", "-ub", str(ubatch),
           "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
           "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
           "--load-mode", "none", "-t", "1", "--cache-ram", "0",
           "--temp", "1.0", "--top-p", "0.95", "--top-k", "0", "--min-p", "0.0",
           "--repeat-penalty", "1.0", "--reasoning", "on",
           "--host", "127.0.0.1", "--port", str(port)]
    if spec:
        # Comparable to Bruce --draft 7 --draft-ngram 4: 4-token lookup, 7-token draft.
        cmd += ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "4",
                "--spec-ngram-simple-size-m", "7", "--spec-ngram-simple-min-hits", "1"]
    return cmd


class Server:
    def __init__(self, engine, tag, port, spec=False, ubatch=1024):
        self.engine = engine
        self.tag = tag
        self.port = port
        self.spec = spec
        self.ubatch = ubatch
        self.proc = None
        self.logf = None
        self.logpath = os.path.join(LOGS, "%s_%s.log" % (engine, tag))

    def start(self):
        os.makedirs(LOGS, exist_ok=True)
        cmd = bruce_cmd(self.port, 7 if self.spec else 0) if self.engine == "bruce" \
            else llama_cmd(self.port, self.spec, self.ubatch)
        log("starting %s (%s): %s" % (self.engine, self.tag, " ".join(cmd)))
        self.logf = open(self.logpath, "wb")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        t0 = time.time()
        while time.time() - t0 < 300:
            if self.proc.poll() is not None:
                raise RuntimeError("%s exited during startup (see %s)" % (self.engine, self.logpath))
            try:
                h = get_json(self.port, "/health", timeout=5)
                if h.get("status") in ("ok", "loading model"):
                    if h.get("status") == "ok":
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("%s did not become healthy in 300s" % self.engine)
        log("%s up in %.1fs, gpu=%s" % (self.engine, time.time() - t0, gpu_info()))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=30)
        if self.logf:
            self.logf.close()
            self.logf = None
        wait_vram_free()
        log("%s stopped, gpu=%s" % (self.engine, gpu_info()))


def make_prompt(corpus, suffix, depth):
    if depth > len(corpus) + len(suffix):
        raise ValueError("corpus too short for depth %d" % depth)
    n = depth - len(suffix)
    p = corpus[:n] + suffix
    assert len(p) == depth, (len(p), depth)
    return p


def request(port, prompt, spec):
    body = {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "repeat_last_n": 64,
        "seed": 1,
        "cache_prompt": False,
        "ignore_eos": True,
    }
    if spec:
        # Bruce accepts draft per request; llama.cpp ignores unknown keys.
        body["draft"] = 7
    r, wall, nbytes = post_json(port, "/completion", body)
    t = r.get("timings", {})
    out = {
        "prompt_n": r.get("tokens_evaluated", t.get("prompt_n")),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": t.get("prompt_per_second"),
        "predicted_n": r.get("tokens_predicted", t.get("predicted_n")),
        "predicted_ms": t.get("predicted_ms"),
        "predicted_tps": t.get("predicted_per_second"),
        "ttft_ms": t.get("ttft_ms"),
        "cache_n": t.get("cache_n", r.get("tokens_cached")),
        "total_ms": t.get("total_ms"),
        "wall_s": wall,
        "body_mib": nbytes / (1024 * 1024),
        "content_head": (r.get("content") or "")[:240],
        "gpu_after": gpu_info(),
    }
    return out


def run_suite(engine, tag, spec, depths, corpus, suffix, results_path, ubatch=1024):
    port = PORT_BRUCE if engine == "bruce" else PORT_LLAMA
    srv = Server(engine, tag, port, spec=spec, ubatch=ubatch)
    results = []
    meta = {
        "engine": engine, "tag": tag, "spec": spec, "ctx": CTX, "n_predict": N_PREDICT,
        "model": MODEL, "depths": depths, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(), "gpu": gpu_info(),
        "cmd": bruce_cmd(port, 7 if spec else 0) if engine == "bruce"
        else llama_cmd(port, spec, ubatch),
    }
    try:
        srv.start()
        # Warm up weights, CUDA graphs, and GPU clocks on a 4K prompt.
        warm = make_prompt(corpus["text"], suffix["text"], 4096)
        log("warmup %s..." % engine)
        w = request(port, warm, spec)
        log("warmup done: prefill %.0f tok/s, decode %.2f tok/s, %.1fs"
            % (w["prompt_tps"], w["predicted_tps"], w["wall_s"]))
        for depth, reps in depths:
            for pname in PROMPTS:
                prompt = make_prompt(corpus[pname], suffix[pname], depth)
                for rep in range(reps):
                    log("%s %s %s depth=%d rep=%d/%d ..." % (engine, tag, pname, depth, rep + 1, reps))
                    g0 = gpu_info()
                    try:
                        r = request(port, prompt, spec)
                        rec = {"prompt": pname, "depth": depth, "rep": rep,
                               "gpu_before": g0, **r}
                        log("  prefill %d tok in %.1f s = %.1f tok/s | decode %d tok in %.1f s "
                            "= %.2f tok/s | wall %.1f s | gpu %s"
                            % (r["prompt_n"], r["prompt_ms"] / 1000, r["prompt_tps"],
                               r["predicted_n"], r["predicted_ms"] / 1000, r["predicted_tps"],
                               r["wall_s"], r["gpu_after"]))
                    except Exception as e:
                        rec = {"prompt": pname, "depth": depth, "rep": rep,
                               "gpu_before": g0, "error": repr(e)}
                        log("  ERROR: %r" % (e,))
                    results.append(rec)
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump({"meta": meta, "results": results}, f, indent=2)
                    # Save a copy of the generated text for sanity checks.
                    if "content_head" in rec:
                        with open(os.path.join(BENCH, "samples_%s_%s.txt" % (engine, tag)),
                                  "a", encoding="utf-8") as f:
                            f.write("\n=== %s depth=%d rep=%d ===\n%s\n"
                                    % (pname, depth, rep, rec["content_head"]))
    finally:
        srv.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["bruce", "llama", "both"], default="both")
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--depths", default="", help="comma list, optionally depth:reps")
    ap.add_argument("--ubatch", type=int, default=1024)
    ap.add_argument("--spec", action="store_true")
    args = ap.parse_args()

    os.makedirs(BENCH, exist_ok=True)
    with open(os.path.join(BENCH, "corpus_text_ids.json")) as f:
        text_ids = json.load(f)
    with open(os.path.join(BENCH, "corpus_code_ids.json")) as f:
        code_ids = json.load(f)
    with open(os.path.join(BENCH, "text_suffix_ids.json")) as f:
        text_suffix = json.load(f)
    with open(os.path.join(BENCH, "code_suffix_ids.json")) as f:
        code_suffix = json.load(f)
    corpus = {"text": text_ids, "code": code_ids}
    suffix = {"text": text_suffix, "code": code_suffix}
    log("corpora: text=%d code=%d tokens; suffixes text=%d code=%d"
        % (len(text_ids), len(code_ids), len(text_suffix), len(code_suffix)))

    if args.depths:
        depths = []
        for x in args.depths.split(","):
            if ":" in x:
                d, r = x.split(":")
                depths.append((int(d), int(r)))
            else:
                depths.append((int(x), 1))
    else:
        depths = SPEC_DEPTHS if args.spec else DEPTHS
    engines = ["bruce", "llama"] if args.engine == "both" else [args.engine]
    for i, engine in enumerate(engines):
        tag = args.tag
        path = os.path.join(BENCH, "results_%s_%s.json" % (engine, tag))
        log("================ %s / %s ================" % (engine, tag))
        run_suite(engine, tag, args.spec, depths, corpus, suffix, path, ubatch=args.ubatch)
        if i + 1 < len(engines):
            log("cooling down 90s...")
            time.sleep(90)
    log("done")


if __name__ == "__main__":
    main()
