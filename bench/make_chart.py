import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(BENCH, "summary.csv"), encoding="utf-8")))
depths = [4096, 32768, 256000]
labels = ["4K", "32K", "256K"]


def avg(engine, depth, key):
    vals = [float(r[key]) for r in rows
            if (r["engine"] == engine or (engine == "bruce" and r["engine"] != "llama"))
            and r["tag"] == "plain" and int(r["depth"]) == depth]
    return sum(vals) / len(vals) if vals else 0.0


d_pre = [avg("bruce", d, "prefill_tps") for d in depths]
l_pre = [avg("llama", d, "prefill_tps") for d in depths]
d_dec = [avg("bruce", d, "decode_tps") for d in depths]
l_dec = [avg("llama", d, "decode_tps") for d in depths]

bg = "#fafaf9"
bruce = "#0f766e"
llama = "#c2410c"
ink = "#1c1917"
muted = "#78716c"

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=bg)
plt.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.13, wspace=0.22)

for ax, dv, lv, title, unit in [
    (axes[0], d_pre, l_pre, "Prompt processing (prefill)", "tokens/s"),
    (axes[1], d_dec, l_dec, "Token generation (decode)", "tokens/s"),
]:
    ax.set_facecolor(bg)
    x = np.arange(len(depths))
    w = 0.34
    b1 = ax.bar(x - w / 2, dv, w, color=bruce, label="Bruce", zorder=3)
    b2 = ax.bar(x + w / 2, lv, w, color=llama, label="llama.cpp", zorder=3)
    for bars, vals in [(b1, dv), (b2, lv)]:
        for b, v in zip(bars, vals):
            ax.annotate("%.1f" % v if v < 100 else "%.0f" % v,
                        (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10.5, color=ink, fontweight="medium")
    for i, (a, b) in enumerate(zip(dv, lv)):
        ax.annotate("%.2fx" % (a / b), (i, max(a, b) * 1.27), ha="center", va="bottom",
                    fontsize=10.5, color=ink, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28", fc=bg, ec="#d6d3d1", lw=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, color=ink)
    ax.set_ylabel(unit, fontsize=10.5, color=muted)
    ax.set_title(title, fontsize=14, color=ink, fontweight="bold", loc="left", pad=14)
    ax.grid(axis="y", color="#e7e5e4", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d3d1")
    ax.tick_params(colors=muted, length=0)
    ax.set_ylim(0, max(max(dv), max(lv)) * 1.30)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: "%.0f" % v))

handles = [plt.Rectangle((0, 0), 1, 1, color=bruce),
           plt.Rectangle((0, 0), 1, 1, color=llama)]
fig.legend(handles, ["Bruce", "llama.cpp"], loc="upper right", ncol=2, frameon=False,
           fontsize=12, bbox_to_anchor=(0.985, 0.91))

fig.text(0.055, 0.955, "Nemotron Dory 1.0 4B Q5_K_M — 262,144-token context",
         fontsize=19, color=ink, fontweight="bold", ha="left")
fig.text(0.055, 0.905,
         "RTX 3060 12 GB  ·  q8_0 KV  ·  greedy  ·  no speculation  ·  same tokenized prompts  ·  256 tokens generated",
         fontsize=10.5, color=muted, ha="left")
fig.text(0.055, 0.035,
         "Server-to-server, cache off. Bars are the mean of text and code prompts. "
         "Numbers above pairs are Bruce/llama.cpp speedups.",
         fontsize=9.5, color=muted, ha="left")

out = os.path.join(BENCH, "benchmark.png")
fig.savefig(out, dpi=180, facecolor=bg)
print(out)
