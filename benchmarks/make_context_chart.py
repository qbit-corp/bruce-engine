"""Render the long-context benchmark charts from summary.json."""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "summary.json")
MACHINE = os.path.join(HERE, "machine.json")
LABEL = {8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K",
         131072: "128K", 256000: "256K"}

BG = "#fafaf9"
BRUCE = "#0f766e"
LLAMA = "#c2410c"
INK = "#1c1917"
MUTED = "#78716c"
GRID = "#e7e5e4"
RULE = "#d6d3d1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": RULE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
})


def load():
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)
    pts = sorted(d["combined"], key=lambda r: r["depth"])
    depths = [r["depth"] for r in pts]
    return d, pts, depths


def hardware_line():
    """Compact CPU / RAM / GPU / OS caption from the captured machine.json."""
    try:
        with open(MACHINE, encoding="utf-8-sig") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return ""

    gpu = [p.strip() for p in (m.get("gpu") or "").split(",")]
    gpu_txt = None
    if len(gpu) >= 2:
        name = gpu[0].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        gpu_txt = "%s %d GB" % (name, int(round(float(gpu[1]) / 1024)))
        extra = []
        if len(gpu) >= 3:
            extra.append("driver %s" % gpu[2])
        if len(gpu) >= 5:
            extra.append("%g W cap" % float(gpu[4]))
        if extra:
            gpu_txt += " (" + ", ".join(extra) + ")"

    os_txt = (m.get("os") or "").replace("Microsoft ", "")
    build = re.match(r"^(.*?)\s+10\.0\.(\d+)$", os_txt)
    if build:
        os_txt = "%s (build %s)" % (build.group(1), build.group(2))

    cpu_txt = (m.get("cpu") or "").replace(" with Radeon Graphics", "")
    cores, threads = m.get("cpu_cores"), m.get("cpu_threads")
    if cpu_txt and cores and threads:
        cpu_txt += " (%dC/%dT)" % (int(cores), int(threads))

    cuda_txt = None
    mm = re.search(r"release\s+([0-9.]+)", m.get("cuda") or "")
    if mm:
        cuda_txt = "CUDA %s" % mm.group(1)

    ram_txt = None
    if m.get("ram_installed_gb"):
        ram_txt = "%g GB" % float(m["ram_installed_gb"])
        if m.get("ram_gb"):
            ram_txt += " (%g GB visible)" % float(m["ram_gb"])
    elif m.get("ram_gb"):
        ram_txt = "%g GB" % float(m["ram_gb"])

    parts = []
    if cpu_txt:
        parts.append("CPU  " + cpu_txt)
    if ram_txt:
        parts.append("RAM  " + ram_txt)
    if gpu_txt:
        parts.append("GPU  " + gpu_txt)
    if os_txt:
        parts.append("OS  " + os_txt)
    if cuda_txt:
        parts.append(cuda_txt)
    return "   ·   ".join(parts)


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(length=0, labelsize=10.5)


def line_panel(ax, x, labels, dv, lv, title, ylabel):
    style(ax)
    ax.plot(x, dv, color=BRUCE, lw=2.4, marker="o", ms=6.5, mec=BG, mew=1.4,
            label="Bruce", zorder=4)
    ax.plot(x, lv, color=LLAMA, lw=2.4, marker="s", ms=6.0, mec=BG, mew=1.4,
            label="llama.cpp", zorder=4)
    for i, (a, b) in enumerate(zip(dv, lv)):
        hi, lo = (a, b) if a >= b else (b, a)
        ax.annotate("%.1f" % a if a < 100 else "%.0f" % a, (x[i], a),
                    xytext=(0, 9 if a >= b else -14), textcoords="offset points",
                    ha="center", va="bottom" if a >= b else "top",
                    fontsize=8.7, color=BRUCE, fontweight="bold", zorder=5)
        ax.annotate("%.1f" % b if b < 100 else "%.0f" % b, (x[i], b),
                    xytext=(0, -14 if a >= b else 9), textcoords="offset points",
                    ha="center", va="top" if a >= b else "bottom",
                    fontsize=8.7, color=LLAMA, fontweight="bold", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=13.5, color=INK, fontweight="bold", loc="left", pad=11)
    ax.set_ylim(0, max(max(dv), max(lv)) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: "%.0f" % v))


def speedup_panel(ax, x, labels, pre, dec):
    style(ax)
    ax.plot(x, pre, color="#115e59", lw=2.2, marker="o", ms=6.0, mec=BG, mew=1.3,
            label="prefill", zorder=4)
    ax.plot(x, dec, color="#d97706", lw=2.2, marker="D", ms=5.6, mec=BG, mew=1.3,
            label="decode", zorder=4)
    med_pre = float(np.median(pre))
    med_dec = float(np.median(dec))
    ax.annotate("median  %.2fx prefill  ·  %.2fx decode" % (med_pre, med_dec),
                (0.02, 0.04), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=8.6, color=MUTED)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, color=INK)
    ax.set_ylabel("Bruce / llama.cpp", fontsize=10.5)
    ax.set_title("Bruce speedup", fontsize=13.5, color=INK, fontweight="bold",
                 loc="left", pad=11)
    hi = max(max(pre), max(dec))
    lo = min(1.0, min(min(pre), min(dec))) - 0.04
    ax.set_ylim(lo, hi + 0.12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: "%.1fx" % v))
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=2)


def chart_lines(d, pts, depths):
    labels = [LABEL.get(x, "%.0fK" % (x / 1000)) for x in depths]
    x = np.arange(len(depths))
    dv = [r["bruce"]["prefill_tps"] for r in pts]
    lv = [r["llama"]["prefill_tps"] for r in pts]
    dd = [r["bruce"]["decode_tps"] for r in pts]
    ld = [r["llama"]["decode_tps"] for r in pts]
    pre = [r["prefill_speedup"] for r in pts]
    dec = [r["decode_speedup"] for r in pts]

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.7), facecolor=BG)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.745, bottom=0.14, wspace=0.25)

    line_panel(axes[0], x, labels, dv, lv, "Prompt processing (prefill)", "tokens/s")
    line_panel(axes[1], x, labels, dd, ld, "Token generation (decode)", "tokens/s")
    speedup_panel(axes[2], x, labels, pre, dec)

    handles = [plt.Line2D([], [], color=BRUCE, lw=2.4, marker="o", ms=6.5),
               plt.Line2D([], [], color=LLAMA, lw=2.4, marker="s", ms=6.0)]
    fig.legend(handles, ["Bruce", "llama.cpp"], loc="upper right", ncol=2,
               frameon=False, fontsize=11, bbox_to_anchor=(0.99, 0.945))

    fig.text(0.05, 0.955, "Bruce vs llama.cpp: prefill and decode vs context",
             fontsize=18, color=INK, fontweight="bold", ha="left")
    fig.text(0.05, 0.900,
             "Nemotron Dory 1.0 4B Q5_K_M  ·  single slot  ·  "
             "q8_0 KV  ·  greedy  ·  no speculation  ·  256 tokens generated",
             fontsize=10.2, color=MUTED, ha="left")
    hw = hardware_line()
    if hw:
        fig.text(0.05, 0.858, hw, fontsize=9.6, color=MUTED, ha="left")
    fig.text(0.05, 0.030,
             "Server-to-server via /completion, cache_prompt=false, one fresh process per point. "
             "Context = prompt depth + 256 tokens generation headroom (256K = 256,000 prompt tokens). "
             "Lines are the mean of the text and code prompts.",
             fontsize=9.0, color=MUTED, ha="left")

    out = os.path.join(HERE, "benchmark_context.png")
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(out)


def chart_bars(d, pts, depths):
    labels = [LABEL.get(x, "%.0fK" % (x / 1000)) for x in depths]
    x = np.arange(len(depths))
    dv = np.array([r["bruce"]["prefill_tps"] for r in pts])
    lv = np.array([r["llama"]["prefill_tps"] for r in pts])
    dd = np.array([r["bruce"]["decode_tps"] for r in pts])
    ld = np.array([r["llama"]["decode_tps"] for r in pts])
    pre = [r["prefill_speedup"] for r in pts]
    dec = [r["decode_speedup"] for r in pts]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.3), facecolor=BG)
    plt.subplots_adjust(left=0.055, right=0.985, top=0.755, bottom=0.135, wspace=0.22)

    for ax, a, b, title, sp in [
        (axes[0], dv, lv, "Prompt processing (prefill)", pre),
        (axes[1], dd, ld, "Token generation (decode)", dec),
    ]:
        style(ax)
        w = 0.34
        b1 = ax.bar(x - w / 2, a, w, color=BRUCE, label="Bruce", zorder=3)
        b2 = ax.bar(x + w / 2, b, w, color=LLAMA, label="llama.cpp", zorder=3)
        for bars, vals in ((b1, a), (b2, b)):
            for bar, v in zip(bars, vals):
                ax.annotate("%.1f" % v if v < 100 else "%.0f" % v,
                            (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=9.2, color=INK, zorder=5)
        for i, (aa, bb, s) in enumerate(zip(a, b, sp)):
            ax.annotate("%.2fx" % s, (i, max(aa, bb) * 1.18), ha="center", va="bottom",
                        fontsize=9.5, color=INK, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=RULE, lw=0.7),
                        zorder=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11.5, color=INK)
        ax.set_ylabel("tokens/s", fontsize=10.5)
        ax.set_title(title, fontsize=13.5, color=INK, fontweight="bold", loc="left",
                     pad=11)
        ax.set_ylim(0, max(max(a), max(b)) * 1.34)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: "%.0f" % v))

    handles = [plt.Rectangle((0, 0), 1, 1, color=BRUCE),
               plt.Rectangle((0, 0), 1, 1, color=LLAMA)]
    fig.legend(handles, ["Bruce", "llama.cpp"], loc="upper right", ncol=2,
               frameon=False, fontsize=11.5, bbox_to_anchor=(0.985, 0.945))
    fig.text(0.055, 0.955, "Bruce vs llama.cpp: long-context prefill and decode",
             fontsize=18, color=INK, fontweight="bold", ha="left")
    fig.text(0.055, 0.900,
             "Nemotron Dory 1.0 4B Q5_K_M  ·  single slot  ·  "
             "q8_0 KV  ·  greedy  ·  no speculation  ·  256 tokens generated",
             fontsize=10.2, color=MUTED, ha="left")
    hw = hardware_line()
    if hw:
        fig.text(0.055, 0.858, hw, fontsize=9.6, color=MUTED, ha="left")
    fig.text(0.055, 0.028,
             "Server-to-server, cache off. Bars are the mean of text and code prompts. "
             "Numbers above pairs are Bruce/llama.cpp speedups. 256K = 256,000 prompt tokens.",
             fontsize=9.2, color=MUTED, ha="left")

    out = os.path.join(HERE, "benchmark_context_bars.png")
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    d, pts, depths = load()
    chart_lines(d, pts, depths)
    chart_bars(d, pts, depths)


if __name__ == "__main__":
    main()
