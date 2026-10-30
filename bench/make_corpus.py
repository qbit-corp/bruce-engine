"""Build deterministic text/code corpora and tokenize them once.

Text: plain-text extracts from a fixed list of English Wikipedia articles.
Code: concatenated C++/CUDA sources from the pinned llama.cpp checkout.
The resulting token-id arrays are shared by both benchmark servers.
"""
import json, os, re, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "bench")
LLAMA = os.environ.get("LLAMA_CPP", os.path.join(ROOT, "tools", "doryref"))
UA = {"User-Agent": "Bruce-bench/1.0 (local benchmark)"}

TEXT_ARTICLES = [
    "World War II", "United States", "India", "World War I", "Russian Empire",
    "Roman Empire", "History of the Roman Empire", "Physics", "Evolution",
    "Association football", "Second Sino-Japanese War", "Soviet Union",
    "Nazi Germany", "Cold War", "China", "France", "Germany", "United Kingdom",
    "Japan", "Italy", "Spain", "Brazil", "Africa", "Asia", "Europe",
    "Mathematics", "Computer science", "Artificial intelligence",
    "Machine learning", "Climate change", "Industrial Revolution",
    "Renaissance", "Middle Ages", "Byzantine Empire", "Ottoman Empire",
    "Mongol Empire", "Age of Discovery", "Scientific Revolution",
    "French Revolution", "American Civil War", "Ancient Rome", "Ancient Greece",
    "Mesopotamia", "History of China", "History of India", "History of Japan",
    "History of Africa", "History of Europe", "History of the United States",
    "History of the Middle East", "Philosophy", "Economics", "Psychology",
    "Sociology", "Anthropology", "Linguistics", "Art", "Music", "Literature",
    "Architecture", "Medicine", "Engineering", "Chemistry", "Biology",
    "Astronomy", "Geology", "Oceanography", "Meteorology", "Statistics",
]

CODE_FILES = [
    "src/llama-context.cpp", "src/llama-model.cpp", "src/llama-vocab.cpp",
    "src/llama-sampler.cpp", "src/llama-graph.cpp", "src/llama-kv-cache.cpp",
    "src/llama-arch.cpp", "src/llama-memory.cpp", "src/llama-batch.cpp",
    "src/llama.cpp", "src/unicode.cpp", "src/unicode-data.cpp",
    "src/models/models.h",
    "common/arg.cpp", "common/speculative.cpp", "common/sampling.cpp",
    "common/common.cpp", "common/chat.cpp", "common/json-schema-to-grammar.cpp",
    "include/llama.h",
    "ggml/src/ggml-cuda/ggml-cuda.cu", "ggml/src/ggml-cuda/fattn-mma-f16.cuh",
    "ggml/src/ggml-cuda/fattn-tile.cu", "ggml/src/ggml-cuda/mmq.cu",
    "ggml/src/ggml-cuda/mmvq.cu", "ggml/src/ggml-cuda/dequantize.cuh",
    "ggml/src/ggml-cuda/common.cuh", "ggml/src/ggml-cuda/vecdotq.cuh",
    "ggml/src/ggml-cpu/ggml-cpu.c", "ggml/src/ggml-cpu/ops.cpp",
    "ggml/src/ggml-backend.cpp", "ggml/src/ggml.c", "ggml/include/ggml.h",
]


def fetch_text():
    out = []
    total = 0
    for i, title in enumerate(TEXT_ARTICLES):
        params = {
            "action": "query", "format": "json", "prop": "extracts",
            "explaintext": 1, "redirects": 1, "titles": title,
        }
        try:
            r = requests.get("https://en.wikipedia.org/w/api.php", params=params,
                             headers=UA, timeout=60)
            r.raise_for_status()
            pages = r.json()["query"]["pages"]
            page = next(iter(pages.values()))
            text = page.get("extract", "")
        except Exception as e:
            print("  skip %-38s %s" % (title, e), flush=True)
            continue
        text = text.replace("\r\n", "\n").strip()
        if len(text) < 2000:
            print("  skip %-38s too short (%d)" % (title, len(text)), flush=True)
            continue
        out.append("\n\n## %s\n\n%s\n" % (title, text))
        total += len(text)
        print("  %-38s %7d chars  (total %7d)" % (title, len(text), total), flush=True)
        if total > 1_600_000:
            break
    return "".join(out)


def build_code():
    parts = []
    total = 0
    for rel in CODE_FILES:
        path = os.path.join(LLAMA, rel)
        if not os.path.exists(path):
            print("  missing", rel, flush=True)
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        parts.append("\n\n// ===== %s =====\n\n%s\n" % (rel, text))
        total += len(text)
        print("  %-45s %8d chars (total %8d)" % (rel, len(text), total), flush=True)
    return "".join(parts)


def main():
    os.makedirs(BENCH, exist_ok=True)
    text_path = os.path.join(BENCH, "text_corpus.txt")
    code_path = os.path.join(BENCH, "code_corpus.txt")
    if os.path.exists(text_path) and os.path.getsize(text_path) > 1_000_000:
        print("text corpus exists, skipping download", flush=True)
    else:
        print("fetching Wikipedia text...", flush=True)
        text = fetch_text()
        # Drop the Gutenberg-style navigation tail risk: keep only clean paragraphs.
        text = re.sub(r"\n{3,}", "\n\n", text)
        with open(text_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print("building code corpus...", flush=True)
    code = build_code()
    with open(code_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(code)
    for p in (text_path, code_path):
        print("%-60s %d bytes" % (p, os.path.getsize(p)), flush=True)

    # Tokenize both corpora through the llama.cpp server started by tokenize_corpora.py.


if __name__ == "__main__":
    main()
