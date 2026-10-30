"""Tokenize arbitrary text into token-id files using llama-server's /tokenize.

The server is started CPU-only (no VRAM) because we only need the vocab.

usage: python tools/tok_text.py <out_name> <target_tokens> <file...>
Concatenates the files (with a file banner) until ~target_tokens ids are produced.
"""
import json, os, subprocess, sys, time, urllib.request

SERVER = os.environ.get("LLAMA_SERVER", "llama-server.exe" if os.name == "nt" else "llama-server")
MODEL = os.environ.get("BRUCE_MODEL", "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
PORT = 18089


def post(path, obj):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=60) as r:
        return r.read().decode()


def main():
    name, target = sys.argv[1], int(sys.argv[2])
    files = sys.argv[3:]
    os.makedirs("tmp", exist_ok=True)
    cmd = [SERVER, "-m", MODEL, "-ngl", "0", "-c", "512", "-np", "1",
           "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "-b", "512", "-ub", "512"]
    log = open(os.path.join("tmp", "server_tok_text.log"), "wb")
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
        ids = []
        used = []
        for path in files:
            if len(ids) >= target:
                break
            text = open(path, encoding="utf-8", errors="replace").read()
            banner = "\n\n// ===== %s =====\n" % os.path.basename(path)
            # chunk on line boundaries to keep BPE seams realistic
            chunks, cur = [], banner
            for line in text.splitlines(keepends=True):
                cur += line
                if len(cur) > 40000:
                    chunks.append(cur)
                    cur = ""
            if cur:
                chunks.append(cur)
            got = 0
            for ch in chunks:
                if len(ids) >= target:
                    break
                tok = post("/tokenize", {"content": ch, "add_special": False})
                ids.extend(int(t) for t in tok["tokens"])
                got += len(tok["tokens"])
            used.append((os.path.basename(path), got))
        ids = ids[:target]
        open("tmp/ids_%s.txt" % name, "w").write(" ".join(map(str, ids)))
        print("%s: %d tokens from %d files" % (name, len(ids), len(used)))
        for fn, n in used:
            print("   %8d  %s" % (n, fn))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()
        log.close()


main()
