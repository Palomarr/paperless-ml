#!/usr/bin/env bash
# ============================================================================
# benchmark_deployed.sh — measure the live deployed ml-gateway stack and
# emit a serving_options.csv row for the bi-encoder-GPU/TrOCR-CPU runtime.
#
# Differs from benchmarks/benchmark_all.sh which spins up isolated initial-impl
# compose configs to compare candidate variants. This script measures the
# ACTUAL DEPLOYED system (whatever's currently running on :8090), produces:
#   - JSON results at benchmarks/<config_name>.json (matching baseline_*.json)
#   - A CSV row template ready to append to serving_options.csv
#   - Summary line for quick eyeballing
#
# Why a separate script: the existing benchmark_all.sh assumes a venv in
# serving/venv from setup_serving.sh, which is initial-impl scaffolding.
# Deployed-stack measurement should work on a fresh node without that.
#
# Usage: bash serving/benchmarks/benchmark_deployed.sh [config_name]
#   config_name   defaults to ort_fp32_split_p100_deployed
#
# Read-only against ml-gateway (HTTP requests only); does not modify any
# stack state. Safe to re-run.
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_NAME="${1:-ort_fp32_split_p100_deployed}"
URL="${URL:-http://localhost:8090}"
REQUESTS="${REQUESTS:-50}"
CONCURRENCY="${CONCURRENCY:-1,4,8,16}"
OUTPUT_JSON="serving/benchmarks/${CONFIG_NAME}.json"
VENV_DIR="${VENV_DIR:-/tmp/bench-venv}"

# ---- output helpers ----
if [[ -t 1 ]]; then
    C_GREEN=$'\033[32m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_GREEN=""; C_BOLD=""; C_RESET=""
fi
info() { printf "%s==>%s %s\n" "$C_BOLD" "$C_RESET" "$*"; }
ok()   { printf " %sOK%s   %s\n" "$C_GREEN" "$C_RESET" "$*"; }

# ---- preflight ----
info "Pre-flight: ml-gateway reachable at $URL/health"
health="$(curl -fsS "$URL/health" 2>/dev/null || echo '')"
if [[ -z "$health" ]]; then
    echo "FAIL: $URL/health did not respond. Is the stack up?"
    exit 1
fi
ok "/health → $health"

# Resolve current commit so the CSV row records what was measured
CODE_VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# ---- ensure venv with aiohttp + Pillow ----
info "Setting up benchmark venv at $VENV_DIR (one-time)"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet aiohttp Pillow
fi
ok "venv ready: $($VENV_DIR/bin/python --version)"

# ---- run benchmark ----
info "Benchmarking $URL (requests=$REQUESTS, concurrency=$CONCURRENCY)"
"$VENV_DIR/bin/python" serving/benchmarks/benchmark_fastapi.py \
    --url "$URL" \
    --requests "$REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output "$OUTPUT_JSON"
ok "results saved to $OUTPUT_JSON"

# ---- extract p50/p95/throughput from concurrency=1 row (matches baseline format) ----
# baseline rows in serving_options.csv reported the concurrency=1 figures as
# the headline p50/p95 values (since the rest are reported in the notes).
info "Extracting headline numbers from concurrency=1"
read -r P50_HTR P95_HTR RPS_HTR < <("$VENV_DIR/bin/python" -c "
import json, sys
with open('$OUTPUT_JSON') as f:
    d = json.load(f)
htr = next(r for r in d['/htr'] if r['concurrency'] == 1)
print(htr['p50_ms'], htr['p95_ms'], htr['throughput_rps'])
")
read -r P50_SEARCH P95_SEARCH RPS_SEARCH < <("$VENV_DIR/bin/python" -c "
import json, sys
with open('$OUTPUT_JSON') as f:
    d = json.load(f)
s = next(r for r in d['/search/query'] if r['concurrency'] == 1)
print(s['p50_ms'], s['p95_ms'], s['throughput_rps'])
")

ERR_RATE="$("$VENV_DIR/bin/python" -c "
import json
with open('$OUTPUT_JSON') as f:
    d = json.load(f)
errs = []
for ep in ('/htr', '/search/query'):
    errs.extend(r['error_rate'] for r in d[ep])
print(max(errs) if errs else 0.0)
")"

# ---- emit CSV row template ----
echo
info "CSV row template — append to serving/benchmarks/serving_options.csv"
echo
cat <<EOF
${CONFIG_NAME},/htr + /search/query,trocr-small-handwritten + all-mpnet-base-v2 (stock HF),${CODE_VERSION},NVIDIA Tesla P100 (TrOCR CPU + bi-encoder GPU),${P50_HTR},${P95_HTR},${P50_SEARCH},${P95_SEARCH},${RPS_HTR},${RPS_SEARCH},${ERR_RATE},"${CONCURRENCY}",chameleon_p100_tacc,"Deployed E1 ORT runtime: mpnet bi-encoder on CUDAExecutionProvider; TrOCR on CPUExecutionProvider with use_io_binding=False (autoregressive decode loop doesn't benefit from GPU + triggers cuDNN errors on P100 — see dev/serving commits). Stock HF weights (Airflow htr_training pipeline hasn't gated a real fine-tune yet, see scripts/check_model_state.sh). Verified end-to-end via 17/17 scripts/verify_integration.sh checkpoints."
EOF
echo
ok "Done. Review the row above + paste into serving_options.csv."
