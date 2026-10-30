#!/usr/bin/env bash
# One-step long-context benchmark: run, analyze, chart.
set -euo pipefail
cd "$(dirname "$0")"

python run_context_bench.py --engine both --fresh
python analyze_context.py
python make_context_chart.py
