import json, subprocess, time, urllib.request, os, sys

SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get("BRUCE_MODEL", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
PORT = 18087

PROMPTS = {
    "code": (
        "def merge_sorted(a, b):\n"
        "    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n"
        "    result = []\n"
        "    i = j = 0\n"
        "    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n"
        "            result.append(a[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            result.append(b[j])\n"
        "            j += 1\n"
        "    result.extend(a[i:])\n"
        "    result.extend(b[j:])\n"
        "    return result\n"
        "\n"
        "def binary_search(arr, target):\n"
        "    \"\"\"Return the index of target in a sorted list, or -1.\"\"\"\n"
    ),
    "chat": (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Explain in simple terms how a hash table works and why lookups are fast.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "prose": (
        "The history of the Roman Empire is a story of ambition, reform, and decline. "
        "At its height the empire stretched from Britain to Mesopotamia, and its roads, "
        "laws, and cities shaped the world that followed. Historians still argue about "
        "the causes of its fall: fiscal strain, plague, migration, and the sheer difficulty "
        "of governing such a vast territory all played a part. "
    ),
    "long": (
        "You are an expert C++ programmer. Write a complete, efficient implementation of a "
        "lock-free single-producer single-consumer ring buffer with the following requirements:\n"
        "- fixed capacity chosen at construction time\n"
        "- a single producer thread calls push() and a single consumer thread calls pop()\n"
        "- no locks, no dynamic allocation in push/pop\n"
        "- correct memory ordering so that data written before push() is visible after pop()\n"
        "Provide the full header and explain the memory ordering choices.\n"
    ),
}


def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()


cmd = [SERVER, "-m", MODEL, "-ngl", "999", "-c", "4096", "-fa", "on", "-np", "1",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--override-kv", "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop",
       "-ot", "blk.*(scale|alpha).*=CUDA0,token_embd.*=CUDA0,output_scale.*=CUDA0",
       "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", "2048", "-ub", "1024"]
log = open(os.path.join("tmp", "server_tok.log"), "wb")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
os.makedirs("tmp", exist_ok=True)
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
    for name, text in PROMPTS.items():
        tok = post("/tokenize", {"content": text, "add_special": False, "with_pieces": True})
        ids = [t["id"] if isinstance(t, dict) else int(t) for t in tok["tokens"]]
        pieces = tok.get("pieces") or [t.get("piece", "") if isinstance(t, dict) else "" for t in tok["tokens"]]
        open("tmp/ids_%s.txt" % name, "w").write(" ".join(map(str, ids)))
        print("%-6s %4d tokens" % (name, len(ids)))
        if pieces:
            print("   first pieces:", [p.get("piece", p) if isinstance(p, dict) else p for p in pieces[:16]])
finally:
    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()
    log.close()
