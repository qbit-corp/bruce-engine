import os, time, requests

BENCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BENCH, "text_corpus.txt")
TITLES = ["Soviet Union", "Cold War", "China", "Japan", "Germany", "France",
          "United Kingdom", "Italy", "Spain", "Brazil", "Africa", "Mathematics"]
UA = {"User-Agent": "Bruce-bench/1.0 (local benchmark)"}

text = open(OUT, encoding="utf-8").read()
have = {line[3:].strip() for line in text.splitlines() if line.startswith("## ")}
added = 0
for title in TITLES:
    if title in have:
        print("have", title, flush=True)
        continue
    for attempt in range(4):
        r = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "prop": "extracts",
            "explaintext": 1, "redirects": 1, "titles": title,
        }, headers=UA, timeout=60)
        if r.status_code == 200:
            break
        print("  %s -> %d, retrying" % (title, r.status_code), flush=True)
        time.sleep(20)
    else:
        print("  FAILED", title, flush=True)
        continue
    page = next(iter(r.json()["query"]["pages"].values()))
    t = page.get("extract", "").replace("\r\n", "\n").strip()
    if len(t) < 2000:
        print("  short", title, len(t), flush=True)
        continue
    text += "\n\n## %s\n\n%s\n" % (title, t)
    added += len(t)
    print("  %-22s %7d chars (added %7d)" % (title, len(t), added), flush=True)
    time.sleep(6)
    if len(text) > 1_100_000:
        break
open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("text corpus now %d bytes" % os.path.getsize(OUT), flush=True)
