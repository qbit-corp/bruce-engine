"""Long-context, single-slot prefill/decode benchmark: Bruce vs llama.cpp.

Both engines serve the same Dory 1.0 4B Q5_K_M GGUF through their HTTP APIs and
receive exactly the same prompt token-id arrays.  For every context point the
KV pool is sized to ``prompt_depth + generation_headroom`` so the measured
context is the actual prompt depth (the llama-bench convention), and the server
is restarted so neither engine keeps a warm cache between points.

Default sweep: 8K, 16K, 32K, 64K, 128K, 256K prompt depth.
Default protocol: 1 slot, greedy, no speculation, cache_prompt=false,
256 generated tokens, text and code prompts.

Results are written incrementally to ``benchmarks/raw/results_<engine>.json``
and server logs to ``benchmarks/logs/``.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "bench")
LOGS = os.path.join(HERE, "logs")
RAW = os.path.join(HERE, "raw")

MODEL = os.environ.get(
    "BRUCE_MODEL",
    os.path.join(ROOT, "models", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"))
BRUCE = os.path.join(ROOT, "engine", "bruce.exe" if os.name == "nt" else "bruce")
LLAMA = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")

PORT_BRUCE = 18210
PORT_LLAMA = 18211

DEPTHS = [8192, 16384, 32768, 65536, 131072, 256000]
# Prompt depth is exact; the KV pool gets only enough extra room for generation.
GEN_HEADROOM = 256
N_PREDICT = 256
PROMPTS = ["text", "code"]
DEFAULT_REPS = {8192: 2, 16384: 2, 32768: 2, 65536: 1,
                131072: 1, 256000: 1}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,"
             "clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
        vals = [x.strip() for x in out.splitlines()[0].split(",")]
        return dict(mem_used_mib=int(float(vals[0])), mem_total_mib=int(float(vals[1])),
                    util=int(float(vals[2])), clock_mhz=int(float(vals[3])),
                    temp_c=int(float(vals[4])), power_w=float(vals[5]))
    except Exception as e:
        return {"error": str(e)}


def wait_vram_free(threshold=1500, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        g = gpu_info()
        if "error" in g or g["mem_used_mib"] <= threshold:
            return True
        time.sleep(2)
    return False


def get_json(port, path, timeout=60):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path),
                                timeout=timeout) as r:
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


def bruce_cmd(port, ctx):
    return [BRUCE, "--model", MODEL, "--serve", "--host", "127.0.0.1",
            "--port", str(port), "--ctx", str(ctx), "--loops", "2",
            "--n-batch", "512", "--n-predict", "-1", "--slots", "1",
            "--draft", "0", "--thinking", "--no-live",
            "--temp", "1.0", "--top-p", "0.95", "--top-k", "0", "--min-p", "0.0",
            "--repeat-penalty", "1.0", "--repeat-last-n", "64"]


def llama_cmd(port, ctx):
    return [LLAMA, "-m", MODEL, "-ngl", "999", "-c", str(ctx), "-fa", "on",
            "--jinja", "-np", "1",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "-b", "2048", "-ub", "1024",
            "--override-kv",
            "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
            "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,"
                   "output_scale.*=CUDA0",
            "--load-mode", "none", "-t", "1", "--cache-ram", "0",
            "--temp", "1.0", "--top-p", "0.95", "--top-k", "0", "--min-p", "0.0",
            "--repeat-penalty", "1.0", "--reasoning", "on",
            "--host", "127.0.0.1", "--port", str(port)]


def cmd_for(engine, port, ctx):
    return bruce_cmd(port, ctx) if engine == "bruce" else llama_cmd(port, ctx)


class Server:
    def __init__(self, engine, ctx):
        self.engine = engine
        self.ctx = ctx
        self.port = PORT_BRUCE if engine == "bruce" else PORT_LLAMA
        self.proc = None
        self.logf = None
        self.logpath = os.path.join(LOGS, "%s_ctx%d.log" % (engine, ctx))
        self.launch_s = None

    def start(self):
        os.makedirs(LOGS, exist_ok=True)
        cmd = cmd_for(self.engine, self.port, self.ctx)
        log("starting %s ctx=%d" % (self.engine, self.ctx))
        self.logf = open(self.logpath, "wb")
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        while time.time() - t0 < 300:
            if self.proc.poll() is not None:
                raise RuntimeError("%s exited during startup (see %s)"
                                   % (self.engine, self.logpath))
            try:
                h = get_json(self.port, "/health", timeout=5)
                if h.get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("%s did not become healthy in 300s" % self.engine)
        self.launch_s = time.time() - t0
        log("%s ctx=%d up in %.1fs, gpu=%s"
            % (self.engine, self.ctx, self.launch_s, gpu_info()))

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
        log("%s ctx=%d stopped, gpu=%s" % (self.engine, self.ctx, gpu_info()))


def make_prompt(corpus, suffix, depth):
    if depth > len(corpus) + len(suffix):
        raise ValueError("corpus too short for depth %d" % depth)
    n = depth - len(suffix)
    p = corpus[:n] + suffix
    assert len(p) == depth, (len(p), depth)
    return p


def request(port, prompt, n_predict):
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "repeat_last_n": 64,
        "seed": 1,
        "cache_prompt": False,
        "ignore_eos": True,
        "draft": 0,
    }
    r, wall, nbytes = post_json(port, "/completion", body)
    t = r.get("timings", {})
    return {
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
        "content_head": (r.get("content") or "")[:160],
    }


def load_results(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "results": []}


def save_results(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def run_engine(args, engine, depths, reps, corpora, suffixes, data):
    path = os.path.join(RAW, "results_%s.json" % engine)
    meta = data["meta"]
    meta.setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    meta.setdefault("gpu_at_start", gpu_info())
    meta["engine"] = engine
    meta["platform"] = platform.platform()
    meta["model"] = MODEL
    meta["depths"] = sorted(set(meta.get("depths", [])) | set(depths))
    meta["n_predict"] = args.n_predict
    meta["gen_headroom"] = args.headroom
    meta["prompts"] = sorted(set(meta.get("prompts", [])) | set(args.prompts))
    meta["reps"] = {**meta.get("reps", {}),
                    **{str(d): reps[d] for d in depths}}
    meta["executable"] = BRUCE if engine == "bruce" else LLAMA
    meta.setdefault("commands", {})
    for d in depths:
        meta["commands"][str(d + args.headroom)] = cmd_for(
            engine, PORT_BRUCE if engine == "bruce" else PORT_LLAMA,
            d + args.headroom)
    save_results(path, data)

    for depth in depths:
        ctx = depth + args.headroom
        srv = Server(engine, ctx)
        try:
            srv.start()
            # Warm weights, CUDA graphs and clocks without paying a full-depth
            # prompt before every measured point.
            warm_depth = min(2048, depth)
            warm = make_prompt(corpora["text"], suffixes["text"], warm_depth)
            log("warmup %s ctx=%d with %d tokens..." % (engine, ctx, warm_depth))
            w = request(srv.port, warm, min(32, args.n_predict))
            log("warmup done: prefill %.0f tok/s, decode %.2f tok/s, %.1fs"
                % (w["prompt_tps"], w["predicted_tps"], w["wall_s"]))
            time.sleep(3)

            for pname in args.prompts:
                prompt = make_prompt(corpora[pname], suffixes[pname], depth)
                for rep in range(reps[depth]):
                    log("%s ctx=%d %s depth=%d rep=%d/%d ..."
                        % (engine, ctx, pname, depth, rep + 1, reps[depth]))
                    g0 = gpu_info()
                    try:
                        r = request(srv.port, prompt, args.n_predict)
                        g1 = gpu_info()
                        rec = {
                            "engine": engine, "prompt": pname, "depth": depth,
                            "context": ctx, "rep": rep, "reps": reps[depth],
                            "launch_s": srv.launch_s, "gpu_before": g0,
                            "gpu_after": g1, **r,
                        }
                        log("  prefill %s tok in %.1f s = %.1f tok/s | "
                            "decode %s tok in %.1f s = %.2f tok/s | "
                            "ttft %.1f ms | wall %.1f s | gpu %s"
                            % (r["prompt_n"], (r["prompt_ms"] or 0) / 1000,
                               r["prompt_tps"] or 0, r["predicted_n"],
                               (r["predicted_ms"] or 0) / 1000,
                               r["predicted_tps"] or 0, r["ttft_ms"] or 0,
                               r["wall_s"], g1))
                    except Exception as e:
                        rec = {"engine": engine, "prompt": pname, "depth": depth,
                               "context": ctx, "rep": rep, "reps": reps[depth],
                               "gpu_before": g0, "error": repr(e)}
                        log("  ERROR: %r" % (e,))
                    data["results"].append(rec)
                    save_results(path, data)
        finally:
            srv.stop()
        time.sleep(args.cooldown)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["bruce", "llama", "both"], default="both")
    ap.add_argument("--depths", default="",
                    help="comma list, optionally depth:reps")
    ap.add_argument("--prompts", default="text,code")
    ap.add_argument("--n-predict", type=int, default=N_PREDICT)
    ap.add_argument("--headroom", type=int, default=GEN_HEADROOM,
                    help="KV tokens reserved for generation beyond prompt depth")
    ap.add_argument("--cooldown", type=float, default=20.0,
                    help="seconds between server restarts / engines")
    ap.add_argument("--fresh", action="store_true",
                    help="archive existing raw/results_*.json before starting")
    args = ap.parse_args()

    if args.depths:
        depths, reps = [], {}
        for x in args.depths.split(","):
            if ":" in x:
                d, r = x.split(":")
                d, r = int(d), int(r)
            else:
                d, r = int(x), DEFAULT_REPS.get(int(x), 1)
            depths.append(d)
            reps[d] = r
    else:
        depths = DEPTHS
        reps = dict(DEFAULT_REPS)
    args.prompts = [p for p in args.prompts.split(",") if p]

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(BENCH, "corpus_text_ids.json")) as f:
        text_ids = json.load(f)
    with open(os.path.join(BENCH, "corpus_code_ids.json")) as f:
        code_ids = json.load(f)
    with open(os.path.join(BENCH, "text_suffix_ids.json")) as f:
        text_suffix = json.load(f)
    with open(os.path.join(BENCH, "code_suffix_ids.json")) as f:
        code_suffix = json.load(f)
    corpora = {"text": text_ids, "code": code_ids}
    suffixes = {"text": text_suffix, "code": code_suffix}
    log("corpora: text=%d code=%d tokens; suffixes text=%d code=%d"
        % (len(text_ids), len(code_ids), len(text_suffix), len(code_suffix)))

    engines = ["bruce", "llama"] if args.engine == "both" else [args.engine]
    if args.fresh:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = os.path.join(RAW, "archive")
        os.makedirs(archive, exist_ok=True)
        for engine in engines:
            path = os.path.join(RAW, "results_%s.json" % engine)
            if os.path.exists(path):
                os.replace(path, os.path.join(
                    archive, "results_%s_%s.json" % (engine, stamp)))
                log("archived existing %s results" % engine)
    data = {engine: load_results(os.path.join(RAW, "results_%s.json" % engine))
            for engine in engines}

    for i, depth in enumerate(depths):
        # Alternate which engine runs first at each depth so neither side always
        # sees the hotter GPU.
        order = engines if i % 2 == 0 else list(reversed(engines))
        for engine in order:
            log("================ %s / ctx %d ================" % (engine, depth))
            run_engine(args, engine, [depth], {depth: reps[depth]},
                       corpora, suffixes, data[engine])
    log("done")


if __name__ == "__main__":
    main()
