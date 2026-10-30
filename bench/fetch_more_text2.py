import os, time, requests

BENCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BENCH, "text_corpus.txt")
TITLES = ["Italy", "United Kingdom", "Brazil", "Spain", "Africa", "Mathematics"]
UA = {"User-Agent": "Bruce-bench/1.0 (local benchmark)"}
text = open(OUT, encoding="utf-8").read()
have = {line[3:].strip() for line in text.splitlines() if line.startswith("## ")}
for title in TITLES:
    if title in have:
        continue
    r = requests.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": title,
    }, headers=UA, timeout=60)
    if r.status_code != 200:
        print("failed", title, r.status_code, flush=True)
        time.sleep(15)
        continue
    t = next(iter(r.json()["query"]["pages"].values())).get("extract", "").strip()
    if len(t) < 2000:
        continue
    text += "\n\n## %s\n\n%s\n" % (title, t)
    print("  %-16s %7d chars, corpus %7d chars" % (title, len(t), len(text)), flush=True)
    time.sleep(5)
    if len(text) > 1_250_000:
        break
open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("text corpus now %d bytes" % os.path.getsize(OUT), flush=True)
