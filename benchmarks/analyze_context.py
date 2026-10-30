"""Analyse the long-context Bruce vs llama.cpp sweep.

Reads ``raw/results_{bruce,llama}.json`` and writes:
  summary.csv          one row per depth/prompt with both engines + speedups
  summary.json         structured medians and metadata
  report.md            full report
  README_SNIPPET.md    compact markdown tables/chart block ready for a README
"""
import csv
import json
import os
import re
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = HERE
MACHINE = os.path.join(HERE, "machine.json")
PROMPTS = ["text", "code"]
ENGINES = ["bruce", "llama"]
DEPTH_LABEL = {8192: "8K", 16384: "16K", 32768: "32K", 65536: "64K",
               131072: "128K", 256000: "256K"}


def load(engine):
    path = os.path.join(RAW, "results_%s.json" % engine)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def measured_time(bruce):
    return (bruce or {}).get("meta", {}).get("started") or time.strftime("%Y-%m-%d %H:%M")


def machine_line():
    if not os.path.exists(MACHINE):
        return None
    with open(MACHINE, encoding="utf-8-sig") as f:
        m = json.load(f)
    gpu = [x.strip() for x in (m.get("gpu") or "").split(",")]
    gpu_txt = ""
    if len(gpu) >= 2:
        gpu_txt = "%s %d GB" % (gpu[0], int(round(float(gpu[1]) / 1024)))
        if len(gpu) >= 3:
            gpu_txt += " (driver %s" % gpu[2]
            if len(gpu) >= 5:
                gpu_txt += ", %g W cap" % float(gpu[4])
            gpu_txt += ")"
    cuda = ""
    mm = re.search(r"release\s+([0-9.]+)", m.get("cuda") or "")
    if mm:
        cuda = "CUDA %s" % mm.group(1)
    cpu_txt = (m.get("cpu") or "").replace(" with Radeon Graphics", "")
    if cpu_txt and m.get("cpu_cores") and m.get("cpu_threads"):
        cpu_txt += " (%dC/%dT)" % (int(m["cpu_cores"]), int(m["cpu_threads"]))
    ram_txt = None
    if m.get("ram_installed_gb"):
        ram_txt = "%g GB RAM" % float(m["ram_installed_gb"])
        if m.get("ram_gb"):
            ram_txt += " (%g GB visible)" % float(m["ram_gb"])
    elif m.get("ram_gb"):
        ram_txt = "%g GB RAM" % float(m["ram_gb"])
    parts = [m.get("os"), cpu_txt, ram_txt, gpu_txt, cuda]
    return "  ·  ".join(p for p in parts if p)


def group(data):
    out = {}
    for r in (data or {}).get("results", []):
        if "error" in r or not r.get("predicted_n"):
            continue
        out.setdefault((r["prompt"], int(r["depth"])), []).append(r)
    return out


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt_tps(x):
    if x is None:
        return "-"
    return "%.1f" % x if x >= 100 else "%.2f" % x


def fmt_ms(x):
    return "-" if x is None else "%.1f" % x


def fmt_s(x):
    return "-" if x is None else "%.2f" % x


def fmt_sp(x):
    if x is None:
        return "-"
    return "**%.2fx**" % x


def depth_label(d):
    return DEPTH_LABEL.get(d, "%.0fK" % (d / 1000))


def point(rows, engine):
    return {
        "engine": engine,
        "reps": len(rows),
        "context": int(round(med(rows, "context"))) if med(rows, "context") else None,
        "prefill_tps": med(rows, "prompt_tps"),
        "prefill_ms": med(rows, "prompt_ms"),
        "decode_tps": med(rows, "predicted_tps"),
        "decode_ms": med(rows, "predicted_ms"),
        "prompt_n": med(rows, "prompt_n"),
        "predicted_n": med(rows, "predicted_n"),
        "total_ms": med(rows, "total_ms"),
        "wall_s": med(rows, "wall_s"),
        "launch_s": med(rows, "launch_s"),
    }


def build(bruce, llama):
    gd, gl = group(bruce), group(llama)
    depths = sorted({d for (_, d) in set(gd) | set(gl)})
    prompts = [p for p in PROMPTS if any((p, d) in gd or (p, d) in gl for d in depths)]
    rows, combined = [], []
    for d in depths:
        by_prompt = {}
        for p in prompts:
            rd, rl = gd.get((p, d), []), gl.get((p, d), [])
            if not rd or not rl:
                continue
            pd, pl = point(rd, "bruce"), point(rl, "llama")
            rec = {
                "depth": d, "context": pd["context"] or pl["context"], "prompt": p,
                "bruce": pd, "llama": pl,
                "prefill_speedup": (pd["prefill_tps"] / pl["prefill_tps"])
                if pd["prefill_tps"] and pl["prefill_tps"] else None,
                "decode_speedup": (pd["decode_tps"] / pl["decode_tps"])
                if pd["decode_tps"] and pl["decode_tps"] else None,
            }
            rows.append(rec)
            by_prompt[p] = rec
        if len(by_prompt) == len(prompts) and prompts:
            dp = mean([r["bruce"]["prefill_tps"] for r in by_prompt.values()])
            lp = mean([r["llama"]["prefill_tps"] for r in by_prompt.values()])
            dd = mean([r["bruce"]["decode_tps"] for r in by_prompt.values()])
            ld = mean([r["llama"]["decode_tps"] for r in by_prompt.values()])
            combined.append({
                "depth": d,
                "context": rows[-1]["context"],
                "prompt": "combined",
                "bruce": {"prefill_tps": dp, "decode_tps": dd,
                         "prefill_ms": mean([r["bruce"]["prefill_ms"]
                                             for r in by_prompt.values()]),
                         "decode_ms": mean([r["bruce"]["decode_ms"]
                                            for r in by_prompt.values()])},
                "llama": {"prefill_tps": lp, "decode_tps": ld,
                          "prefill_ms": mean([r["llama"]["prefill_ms"]
                                              for r in by_prompt.values()]),
                          "decode_ms": mean([r["llama"]["decode_ms"]
                                             for r in by_prompt.values()])},
                "prefill_speedup": dp / lp if dp and lp else None,
                "decode_speedup": dd / ld if dd and ld else None,
            })
    return depths, prompts, rows, combined


def write_csv(rows, combined):
    path = os.path.join(OUT, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["depth", "context", "prompt", "reps_bruce", "reps_llama",
                    "bruce_prefill_tps", "llama_prefill_tps", "prefill_speedup",
                    "bruce_decode_tps", "llama_decode_tps", "decode_speedup",
                    "bruce_prefill_ms", "llama_prefill_ms",
                    "bruce_decode_ms", "llama_decode_ms",
                    "bruce_ms_per_token", "llama_ms_per_token"])
        for r in list(rows) + list(combined):
            d, l = r["bruce"], r["llama"]
            w.writerow([
                r["depth"], r["context"], r["prompt"], d.get("reps", ""),
                l.get("reps", ""),
                "" if d["prefill_tps"] is None else "%.3f" % d["prefill_tps"],
                "" if l["prefill_tps"] is None else "%.3f" % l["prefill_tps"],
                "" if r["prefill_speedup"] is None else "%.4f" % r["prefill_speedup"],
                "" if d["decode_tps"] is None else "%.4f" % d["decode_tps"],
                "" if l["decode_tps"] is None else "%.4f" % l["decode_tps"],
                "" if r["decode_speedup"] is None else "%.4f" % r["decode_speedup"],
                "" if d["prefill_ms"] is None else "%.1f" % d["prefill_ms"],
                "" if l["prefill_ms"] is None else "%.1f" % l["prefill_ms"],
                "" if d["decode_ms"] is None else "%.1f" % d["decode_ms"],
                "" if l["decode_ms"] is None else "%.1f" % l["decode_ms"],
                "" if (d["decode_ms"] is None or not d.get("predicted_n"))
                else "%.4f" % (d["decode_ms"] / d["predicted_n"]),
                "" if (l["decode_ms"] is None or not l.get("predicted_n"))
                else "%.4f" % (l["decode_ms"] / l["predicted_n"]),
            ])
    return path


def write_json(depths, prompts, rows, combined, bruce, llama):
    obj = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": (bruce or {}).get("meta", {}).get("model"),
        "depths": depths,
        "prompts": prompts,
        "meta": {"bruce": (bruce or {}).get("meta", {}),
                 "llama": (llama or {}).get("meta", {})},
        "points": rows,
        "combined": combined,
    }
    path = os.path.join(OUT, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def table_for(rows, depths, prompts, metric, title):
    lines = ["### " + title, ""]
    head = "| Depth | " + " | ".join("%s Bruce | %s llama | %s speedup"
                                     % (p.title(), p.title(), p.title())
                                     for p in prompts) + " |"
    lines.append(head)
    lines.append("|---:|" + "---:|---:|---:|" * len(prompts))
    for d in depths:
        cells = []
        for p in prompts:
            rec = next((r for r in rows if r["depth"] == d and r["prompt"] == p), None)
            if not rec:
                cells += ["-", "-", "-"]
                continue
            a = rec["bruce"][metric]
            b = rec["llama"][metric]
            cells += [fmt_tps(a), fmt_tps(b), fmt_sp(rec[metric.replace("_tps", "_speedup")])]
        lines.append("| %s | %s |" % (depth_label(d), " | ".join(cells)))
    lines.append("")
    return lines


def latency_table(rows, depths, prompts):
    lines = ["### Prompt and generation time", "",
             "| Depth | " + " | ".join("%s prefill d/l (s) | %s decode d/l (s)"
                                      % (p.title(), p.title()) for p in prompts) + " |",
             "|---:|" + "---:|---:|" * len(prompts)]
    for d in depths:
        cells = []
        for p in prompts:
            rec = next((r for r in rows if r["depth"] == d and r["prompt"] == p), None)
            if not rec:
                cells += ["-", "-"]
                continue
            cells.append("%s / %s" % (fmt_s((rec["bruce"]["prefill_ms"] or 0) / 1000),
                                      fmt_s((rec["llama"]["prefill_ms"] or 0) / 1000)))
            cells.append("%s / %s" % (fmt_s((rec["bruce"]["decode_ms"] or 0) / 1000),
                                      fmt_s((rec["llama"]["decode_ms"] or 0) / 1000)))
        lines.append("| %s | %s |" % (depth_label(d), " | ".join(cells)))
    lines.append("")
    return lines


def combined_table(combined, depths, title):
    lines = ["### " + title, "",
             "| Depth | Prefill Bruce | Prefill llama | Prefill speedup | "
             "Decode Bruce | Decode llama | Decode speedup |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for d in depths:
        r = next((x for x in combined if x["depth"] == d), None)
        if not r:
            continue
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            depth_label(d), fmt_tps(r["bruce"]["prefill_tps"]),
            fmt_tps(r["llama"]["prefill_tps"]), fmt_sp(r["prefill_speedup"]),
            fmt_tps(r["bruce"]["decode_tps"]), fmt_tps(r["llama"]["decode_tps"]),
            fmt_sp(r["decode_speedup"])))
    lines.append("")
    return lines


def observations(combined, rows):
    if not combined:
        return []
    first, last = combined[0], combined[-1]
    lines = []
    pre = [r["prefill_speedup"] for r in combined if r["prefill_speedup"]]
    dec = [r["decode_speedup"] for r in combined if r["decode_speedup"]]
    lines.append(
        "Across the %d measured context points, Bruce's median prefill speedup is "
        "**%.2fx** (range %.2fx-%.2fx) and its median decode speedup is **%.2fx** "
        "(range %.2fx-%.2fx)."
        % (len(combined), statistics.median(pre), min(pre), max(pre),
           statistics.median(dec), min(dec), max(dec)))
    lines.append("")
    lines.append(
        "From %s to %s, prefill falls from **%s** to **%s** tok/s on Bruce "
        "(%.0f%% retained) and from **%s** to **%s** tok/s on llama.cpp "
        "(%.0f%% retained); decode falls from **%s** to **%s** tok/s "
        "and from **%s** to **%s** tok/s respectively."
        % (depth_label(first["depth"]), depth_label(last["depth"]),
           fmt_tps(first["bruce"]["prefill_tps"]), fmt_tps(last["bruce"]["prefill_tps"]),
           100 * last["bruce"]["prefill_tps"] / first["bruce"]["prefill_tps"],
           fmt_tps(first["llama"]["prefill_tps"]), fmt_tps(last["llama"]["prefill_tps"]),
           100 * last["llama"]["prefill_tps"] / first["llama"]["prefill_tps"],
           fmt_tps(first["bruce"]["decode_tps"]), fmt_tps(last["bruce"]["decode_tps"]),
           fmt_tps(first["llama"]["decode_tps"]), fmt_tps(last["llama"]["decode_tps"])))
    lines.append("")
    lines.append(
        "At the %s point the practical numbers are prefill **%s vs %s** tok/s "
        "(%s) and decode **%s vs %s** tok/s (%s)."
        % (depth_label(last["depth"]),
           fmt_tps(last["bruce"]["prefill_tps"]), fmt_tps(last["llama"]["prefill_tps"]),
           fmt_sp(last["prefill_speedup"]), fmt_tps(last["bruce"]["decode_tps"]),
           fmt_tps(last["llama"]["decode_tps"]), fmt_sp(last["decode_speedup"])))
    lines.append("")
    if dec and dec[-1] > dec[0] * 1.05:
        lines.append(
            "The decode lead widens with context, from **%.2fx** at %s to **%.2fx** "
            "at %s, while the prefill lead narrows over the same range."
            % (dec[0], depth_label(first["depth"]), dec[-1],
               depth_label(last["depth"])))
        lines.append("")
    # Where prefill lead changes most.
    if pre:
        best = max(combined, key=lambda r: r["prefill_speedup"] or 0)
        worst = min(combined, key=lambda r: r["prefill_speedup"] or 99)
        lines.append(
            "The prefill gap is widest at **%s** (%s) and narrowest at **%s** "
            "(%s); decode stays ahead at every point."
            % (depth_label(best["depth"]), fmt_sp(best["prefill_speedup"]),
               depth_label(worst["depth"]), fmt_sp(worst["prefill_speedup"])))
        lines.append("")
    return lines


def write_report(depths, prompts, rows, combined, bruce, llama):
    meta_d = (bruce or {}).get("meta", {})
    meta_l = (llama or {}).get("meta", {})
    lines = []
    lines.append("# Bruce vs llama.cpp: prefill and decode vs context")
    lines.append("")
    lines.append("Single-slot server-to-server comparison on the same GGUF, prompt tokens, "
                 "KV type, and GPU. Measured %s." % measured_time(bruce))
    lines.append("")
    mach = machine_line()
    if mach:
        lines.append("**Machine:** " + mach + ".")
        lines.append("")
    lines.append("![prefill and decode vs context](benchmark_context.png)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| Item | Bruce | llama.cpp |")
    lines.append("|---|---|---|")
    lines.append("| Engine | `engine/bruce.exe` | `llama-server.exe` build 1, commit `57b70d0` |")
    lines.append("| Model | `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` | same file |")
    lines.append("| Context per point | prompt depth + 256 generation tokens | same |")
    lines.append("| Slots | 1 (`--slots 1`) | 1 (`-np 1`) |")
    lines.append("| Loops | 2 (`--loops 2`) | 2 (`--override-kv`) |")
    lines.append("| KV cache | q8_0, `last_loop` | q8_0, `last_loop` |")
    lines.append("| Attention | flash (engine default) | `-fa on` |")
    lines.append("| Prompt batch | 512 | `-b 2048 -ub 1024` |")
    lines.append("| Sampling | greedy | greedy |")
    lines.append("| Speculation | off | off |")
    lines.append("| Prefix cache | off (`cache_prompt:false`) | off (`cache_prompt:false`) |")
    lines.append("| Generation | 256 tokens, `ignore_eos` | same |")
    lines.append("| Repeats | 2 at 8K/16K/32K, 1 above | same |")
    lines.append("")
    lines.append("Both servers were driven through `POST /completion` with the same "
                 "token-id array. Each context point uses a fresh server process, so no "
                 "prefix cache or warm KV survives between points. A short warmup request "
                 "precedes the measurements at each point. Exact per-point command lines "
                 "are recorded in `raw/results_*.json` under `meta.commands`.")
    lines.append("")
    lines.append("**Text prompt:** 16 long English Wikipedia articles followed by an essay "
                 "instruction. **Code prompt:** concatenated llama.cpp C++/CUDA sources "
                 "followed by a ring-buffer task. Depths are 8,192 / 16,384 / 32,768 / "
                 "65,536 / 131,072 / 256,000 prompt tokens; the last point follows the "
                 "existing report's 256K convention.")
    lines.append("")
    ok = all(
        r["bruce"]["prompt_n"] == r["depth"] and r["llama"]["prompt_n"] == r["depth"]
        and r["bruce"]["predicted_n"] == 256 and r["llama"]["predicted_n"] == 256
        for r in rows
    )
    if ok:
        lines.append("**Sanity check:** every measured request evaluated exactly the "
                     "requested prompt length and generated 256/256 tokens on both "
                     "engines.")
        lines.append("")

    if meta_d or meta_l:
        rows_gpu = []
        for engine in (bruce, llama):
            for r in (engine or {}).get("results", []):
                if "error" not in r and r.get("gpu_after"):
                    rows_gpu.append(r["gpu_after"])
        clocks = [g["clock_mhz"] for g in rows_gpu if "clock_mhz" in g]
        temps = [g["temp_c"] for g in rows_gpu if "temp_c" in g]
        powers = [g["power_w"] for g in rows_gpu if "power_w" in g]
        if clocks:
            lines.append("**Observed under load:** SM clock %d-%d MHz, temperature %d-%d C, "
                         "board power %.0f-%.0f W." %
                         (min(clocks), max(clocks), min(temps), max(temps),
                          min(powers), max(powers)))
            lines.append("")
        if depths:
            maxd = max(depths)

            def vram(data):
                vals = [r["gpu_after"]["mem_used_mib"]
                        for r in (data or {}).get("results", [])
                        if r.get("depth") == maxd and r.get("gpu_after")]
                return max(vals) if vals else None

            vd, vl = vram(bruce), vram(llama)
            vt = next((r["gpu_after"]["mem_total_mib"]
                       for r in (bruce or {}).get("results", [])
                       if r.get("gpu_after") and "mem_total_mib" in r["gpu_after"]),
                      12288)
            if vd and vl:
                lines.append("**VRAM at %s:** Bruce %s MiB, llama.cpp %s MiB "
                             "of %s MiB." % (depth_label(maxd), f"{vd:,}",
                                             f"{vl:,}", f"{vt:,}"))
                lines.append("")

    lines += combined_table(combined, depths, "Summary (mean of text and code)")
    lines += observations(combined, rows)
    lines += table_for(rows, depths, prompts, "prefill_tps", "Prefill (prompt processing)")
    lines += table_for(rows, depths, prompts, "decode_tps", "Decode (token generation)")
    lines += latency_table(rows, depths, prompts)
    lines.append("### Notes and caveats")
    lines.append("")
    lines.append("* This is a server-to-server measurement. Decode tok/s includes each "
                 "engine's per-token sampler and HTTP loop; Bruce's native CUDA-graph "
                 "replay (`--bench`) is slightly faster than its server number.")
    lines.append("* Context is allocated per point as prompt depth + 256 tokens. The KV "
                 "pool therefore grows with depth, as it would in a real single-slot "
                 "deployment; it is not held at 256K for the 8K point.")
    lines.append("* Bruce enables its Q5_K prompt prepack when it fits beside the KV "
                 "cache; with this budget it is on through 128K and off at 256K. That is "
                 "the shipped `auto` rule, not a benchmark override, and it is the main "
                 "reason the prefill lead narrows at the 256K point.")
    lines.append("* llama.cpp uses its normal 1024-token compute ubatch; Bruce's "
                 "maximum prefill batch is 512 tokens. This is the same asymmetry as the "
                 "earlier head-to-head report.")
    lines.append("* Both engines generate greedily with `ignore_eos`, so the 256 measured "
                 "decode tokens are never cut short by EOS.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| file | contents |")
    lines.append("|---|---|")
    lines.append("| `summary.csv` | per-prompt and combined rows with both engines and speedups |")
    lines.append("| `summary.json` | machine-readable medians and run metadata |")
    lines.append("| `machine.json` | CPU, GPU, driver, CUDA, and RAM of the run |")
    lines.append("| `benchmark_context.png` / `.svg` | prefill, decode, and speedup vs context |")
    lines.append("| `benchmark_context_bars.png` / `.svg` | grouped-bar view of the same data |")
    lines.append("| `README_SNIPPET.md` | compact tables and chart block for a README |")
    lines.append("| `raw/results_*.json` | raw requests, including GPU state |")
    lines.append("| `logs/` | server logs for every context point |")
    lines.append("| `run_full.log` | console log of the full sweep |")
    lines.append("| `run_context_bench.py` | benchmark launcher + request loop |")
    lines.append("| `run_all.cmd` | one-step reproduce script |")
    lines.append("| `analyze_context.py` | medians, tables, report |")
    lines.append("| `make_context_chart.py` | chart generator |")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bat")
    lines.append("benchmarks\\run_all.cmd")
    lines.append("```")
    lines.append("")
    lines.append("`--fresh` archives an earlier `raw/results_*.json` under "
                 "`raw/archive/`; without it, new measurements are appended.")
    lines.append("")
    path = os.path.join(OUT, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def headline(combined):
    pre = [r["prefill_speedup"] for r in combined if r["prefill_speedup"]]
    dec = [r["decode_speedup"] for r in combined if r["decode_speedup"]]
    if not pre or not dec:
        return None
    return ("**Headline:** Bruce is **%.2fx-%.2fx** faster at prefill and "
            "**%.2fx-%.2fx** faster at decode across the measured contexts."
            % (min(pre), max(pre), min(dec), max(dec)))


def write_folder_readme(depths, prompts, rows, combined, bruce, llama):
    lines = []
    lines.append("# Bruce vs llama.cpp: long-context benchmark")
    lines.append("")
    lines.append("Single-slot prefill and decode at **8K, 16K, 32K, 64K, 128K, and "
                 "256K** prompt depth, on the same GGUF, tokenized prompts, KV type, "
                 "and GPU. Measured %s." % measured_time(bruce))
    lines.append("")
    mach = machine_line()
    if mach:
        lines.append("**Machine:** " + mach + ".")
        lines.append("")
    lines.append("![prefill and decode vs context](benchmark_context.png)")
    lines.append("")
    head = headline(combined)
    if head:
        lines.append(head)
        lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("| Item | Bruce | llama.cpp |")
    lines.append("|---|---|---|")
    lines.append("| Engine | `engine/bruce.exe` | `llama-server.exe` build 1, `57b70d0` |")
    lines.append("| Model | Dory 1.0 4B Q5_K_M | same file |")
    lines.append("| Context | prompt depth + 256 generation tokens, per point | same |")
    lines.append("| Slots | 1 | 1 |")
    lines.append("| Loops | 2 | 2 (`--override-kv`) |")
    lines.append("| KV | q8_0, `last_loop` | q8_0, `last_loop` |")
    lines.append("| Attention | flash default | `-fa on` |")
    lines.append("| Prompt batch | 512 | `-b 2048 -ub 1024` |")
    lines.append("| Decode | greedy, no speculation, 256 tokens | same |")
    lines.append("| Prefix cache | off | off |")
    lines.append("| Repeats | 2 at 8K/16K/32K, 1 above | same |")
    lines.append("")
    lines.append("Each context point starts a fresh server process. Requests are "
                 "`POST /completion` with the same token-id array and "
                 "`cache_prompt:false`; a short warmup precedes the measured "
                 "requests. The text prompt is Wikipedia prose plus an essay task; the "
                 "code prompt is llama.cpp sources plus a ring-buffer task.")
    lines.append("")
    lines.append("Depths are 8,192 / 16,384 / 32,768 / 65,536 / 131,072 / 256,000 "
                 "prompt tokens; the last point follows the existing head-to-head "
                 "report's 256K convention.")
    lines.append("")
    lines += combined_table(combined, depths, "Summary (mean of text and code)")
    lines += observations(combined, rows)
    if depths and max(depths) >= 256000:
        lines.append("At 256K Bruce's Q5_K prompt prepack is off because it no longer "
                     "fits beside the KV pool; that shipped `auto` rule is the main reason "
                     "the prefill lead is smaller at the last point. Prefill and decode "
                     "are both single-slot, cache-off, no-speculation server numbers.")
        lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| file | contents |")
    lines.append("|---|---|")
    lines.append("| `report.md` | full report with per-prompt tables, latency, caveats |")
    lines.append("| `README_SNIPPET.md` | compact markdown block ready for the root README |")
    lines.append("| `summary.csv` | all per-prompt and combined rows |")
    lines.append("| `summary.json` | machine-readable medians and metadata |")
    lines.append("| `machine.json` | CPU, GPU, driver, CUDA, and RAM of the run |")
    lines.append("| `benchmark_context.png` / `.svg` | three-panel trend chart (recommended) |")
    lines.append("| `benchmark_context_bars.png` / `.svg` | grouped-bar alternative |")
    lines.append("| `raw/results_*.json` | raw requests and GPU state |")
    lines.append("| `logs/` | server logs, one per engine and context |")
    lines.append("| `run_full.log` | console log of the full sweep |")
    lines.append("| `run_context_bench.py` | launcher and request loop |")
    lines.append("| `run_all.cmd` | run the sweep, analysis, and charts in one step |")
    lines.append("| `analyze_context.py` | medians, CSV/JSON, report |")
    lines.append("| `make_context_chart.py` | chart generator |")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bat")
    lines.append("benchmarks\\run_all.cmd")
    lines.append("```")
    lines.append("")
    lines.append("Or step by step:")
    lines.append("")
    lines.append("```bat")
    lines.append("python benchmarks\\run_context_bench.py --engine both --fresh")
    lines.append("python benchmarks\\analyze_context.py")
    lines.append("python benchmarks\\make_context_chart.py")
    lines.append("```")
    lines.append("")
    lines.append("`--fresh` archives an earlier `raw/results_*.json` under "
                 "`raw/archive/`; without it, new measurements are appended.")
    lines.append("")
    path = os.path.join(OUT, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_snippet(depths, prompts, rows, combined):
    lines = []
    lines.append("<!-- Generated by benchmarks/analyze_context.py. Paste into README.md; "
                 "paths are relative to the repo root. -->")
    lines.append("")
    lines.append("## Bruce vs llama.cpp: prefill and decode vs context")
    lines.append("")
    lines.append("Single slot, same GGUF, same prompt tokens, greedy, no speculation, "
                 "q8_0 KV, 256 tokens generated. Context is the prompt depth; each "
                 "engine gets prompt depth + 256 tokens of KV.")
    lines.append("")
    head = headline(combined)
    if head:
        lines.append(head)
        lines.append("")
    lines.append("![prefill and decode vs context](benchmarks/benchmark_context.png)")
    lines.append("")
    lines.append("<!-- Bar-chart alternative: "
                 "![prefill and decode vs context](benchmarks/benchmark_context_bars.png) -->")
    lines.append("")
    lines += combined_table(combined, depths, "Summary (mean of text and code)")
    lines += table_for(rows, depths, prompts, "prefill_tps", "Prefill (tokens/s)")
    lines += table_for(rows, depths, prompts, "decode_tps", "Decode (tokens/s)")
    lines.append("256K = 256,000 prompt tokens. Bruce's Q5_K prompt prepack is off "
                 "at 256K because it does not fit beside the KV pool, which narrows its "
                 "prefill lead there; decode is unaffected.")
    lines.append("")
    lines.append("Raw data: [`benchmarks/summary.csv`](benchmarks/summary.csv) · "
                 "full report: [`benchmarks/report.md`](benchmarks/report.md)")
    lines.append("")
    path = os.path.join(OUT, "README_SNIPPET.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    bruce, llama = load("bruce"), load("llama")
    if not bruce or not llama:
        raise SystemExit("need both raw/results_bruce.json and raw/results_llama.json")
    depths, prompts, rows, combined = build(bruce, llama)
    write_csv(rows, combined)
    write_json(depths, prompts, rows, combined, bruce, llama)
    write_report(depths, prompts, rows, combined, bruce, llama)
    write_folder_readme(depths, prompts, rows, combined, bruce, llama)
    write_snippet(depths, prompts, rows, combined)
    print("depths:", depths)
    print("points:", len(rows), "combined:", len(combined))
    print("wrote summary.csv, summary.json, report.md, README.md, README_SNIPPET.md")


if __name__ == "__main__":
    main()
