#!/usr/bin/env bash
# Automated benchmark suite for all serving configurations.
# Launches each compose config, runs the appropriate benchmark,
# saves results, and tears down before the next config.
#
# Usage: cd serving && bash benchmarks/benchmark_all.sh [OPTIONS]
#   --configs       Comma-separated list of configs to run (default: all)
#   --requests      Number of requests per concurrency level (default: 100)
#   --concurrency   Comma-separated concurrency levels (default: 1,4,8,16)
#   --output-dir    Directory for results (default: benchmarks/results)
#   --skip-build    Skip docker compose build step
#
# Example:
#   bash benchmarks/benchmark_all.sh --configs fastapi,ort --requests 50

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ── Defaults ──────────────────────────────────────────────────────
REQUESTS=100
CONCURRENCY="1,4,8,16"
OUTPUT_DIR="benchmarks/results"
SKIP_BUILD=false
TRITON_SDK_IMAGE="nvcr.io/nvidia/tritonserver:24.01-py3-sdk"
TRITON_MEASUREMENT_INTERVAL=5000  # ms

# All available configs: name -> compose file
declare -A COMPOSE_FILES=(
  [fastapi]="docker-compose-fastapi.yaml"
  [ort]="docker-compose-ort.yaml"
  [ort-quant]="docker-compose-ort-quant.yaml"
  [ort-quant-cpu]="docker-compose-ort-quant-cpu.yaml"
  [triton]="docker-compose-triton.yaml"
  [ray]="docker-compose-ray.yaml"
)

# Order matters for the table
ALL_CONFIGS="fastapi ort ort-quant ort-quant-cpu triton ray"
SELECTED_CONFIGS="$ALL_CONFIGS"

# ── Parse args ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --configs)     SELECTED_CONFIGS="${2//,/ }"; shift 2 ;;
    --requests)    REQUESTS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --skip-build)  SKIP_BUILD=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

# ── Helpers ───────────────────────────────────────────────────────
log()     { echo -e "\033[0;32m$*\033[0m"; }
section() { echo -e "\n\033[0;36m\033[1m=== $* ===\033[0m\n"; }
err()     { echo -e "\033[0;31mERROR: $*\033[0m" >&2; }

# Ensure external network exists (referenced by compose files)
docker network create paperless-net 2>/dev/null || true

# Use existing venv from setup_serving.sh; install benchmark deps if missing
VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/venv"
if [[ ! -d "$VENV_DIR" ]]; then
  err "No venv found at $VENV_DIR — run setup_serving.sh first"
  exit 1
fi
"$VENV_DIR/bin/pip" install --quiet aiohttp Pillow requests 2>/dev/null
PYTHON="$VENV_DIR/bin/python"

wait_for_http() {
  local url="$1" timeout="${2:-120}"
  for i in $(seq 1 "$timeout"); do
    if curl -sf "$url" &>/dev/null; then return 0; fi
    sleep 1
  done
  err "Timed out waiting for $url"
  return 1
}

teardown() {
  local compose_file="$1"
  log "Tearing down $compose_file..."
  docker compose -f "$compose_file" down -v --remove-orphans 2>/dev/null || true
  sleep 2
}

# ── FastAPI benchmark (Python async) ─────────────────────────────
benchmark_fastapi() {
  local config_name="$1" compose_file="$2"
  local outfile="$OUTPUT_DIR/${config_name}.json"

  section "Benchmarking: $config_name"
  log "Compose: $compose_file"
  log "Output:  $outfile"

  # Tear down any leftover
  teardown "$compose_file"

  # Build and launch
  if [[ "$SKIP_BUILD" == false ]]; then
    log "Building $config_name..."
    docker compose -f "$compose_file" build --progress=plain 2>&1
  fi
  docker compose -f "$compose_file" up -d

  log "Waiting for FastAPI to be ready..."
  if ! wait_for_http "http://localhost:8000/health" 120; then
    err "FastAPI failed to start for $config_name"
    docker compose -f "$compose_file" logs --tail=30
    teardown "$compose_file"
    return 1
  fi
  log "Server ready."

  # Run benchmark
  "$PYTHON" benchmarks/benchmark_fastapi.py \
    --url "http://localhost:8000" \
    --requests "$REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --output "$outfile"

  teardown "$compose_file"
  log "Saved: $outfile"
}

# ── Triton benchmark (perf_analyzer) ─────────────────────────────
benchmark_triton() {
  local config_name="$1" compose_file="$2"
  local outdir="$OUTPUT_DIR/triton"

  section "Benchmarking: $config_name (Triton)"
  log "Compose: $compose_file"
  log "Output:  $outdir/"

  mkdir -p "$outdir"

  # Tear down any leftover
  teardown "$compose_file"

  # Launch Triton
  docker compose -f "$compose_file" up -d

  log "Waiting for Triton models to load..."
  if ! wait_for_http "http://localhost:8000/v2/health/ready" 120; then
    err "Triton failed to start"
    docker compose -f "$compose_file" logs --tail=30
    teardown "$compose_file"
    return 1
  fi
  log "Triton ready."

  # Verify models
  log "Models loaded:"
  curl -s localhost:8000/v2/models/htr_model | python3 -m json.tool 2>/dev/null || true
  curl -s localhost:8000/v2/models/search_model | python3 -m json.tool 2>/dev/null || true

  # Convert comma-separated concurrency to array
  IFS=',' read -ra CONC_LEVELS <<< "$CONCURRENCY"

  # Benchmark search_model
  log ""
  log ">>> search_model (bi-encoder)"
  echo "Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency" \
    > "$outdir/search_model.csv"
  for c in "${CONC_LEVELS[@]}"; do
    log "  concurrency=$c"
    docker run --rm --net=host "$TRITON_SDK_IMAGE" \
      perf_analyzer -u localhost:8000 -m search_model \
      -b 1 --shape input_ids:128 --shape attention_mask:128 \
      --concurrency-range "$c" \
      --measurement-interval "$TRITON_MEASUREMENT_INTERVAL" \
      -f /dev/stdout 2>&1 | tee -a "$outdir/search_model_full.log" | \
      tail -1 >> "$outdir/search_model.csv"
  done

  # Benchmark htr_model
  log ""
  log ">>> htr_model (TrOCR encoder)"
  echo "Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency" \
    > "$outdir/htr_model.csv"
  for c in "${CONC_LEVELS[@]}"; do
    log "  concurrency=$c"
    docker run --rm --net=host "$TRITON_SDK_IMAGE" \
      perf_analyzer -u localhost:8000 -m htr_model \
      -b 1 --shape pixel_values:3,384,384 \
      --concurrency-range "$c" \
      --measurement-interval "$TRITON_MEASUREMENT_INTERVAL" \
      -f /dev/stdout 2>&1 | tee -a "$outdir/htr_model_full.log" | \
      tail -1 >> "$outdir/htr_model.csv"
  done

  teardown "$compose_file"
  log "Saved: $outdir/search_model.csv, $outdir/htr_model.csv"
}

# ── Main loop ─────────────────────────────────────────────────────
section "Benchmark Suite"
log "Configs:     $SELECTED_CONFIGS"
log "Requests:    $REQUESTS per concurrency level"
log "Concurrency: $CONCURRENCY"
log "Output:      $OUTPUT_DIR/"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "$TIMESTAMP | configs=$SELECTED_CONFIGS requests=$REQUESTS concurrency=$CONCURRENCY" \
  >> "$OUTPUT_DIR/run_log.txt"

for config in $SELECTED_CONFIGS; do
  compose_file="${COMPOSE_FILES[$config]:-}"
  if [[ -z "$compose_file" ]]; then
    err "Unknown config: $config (available: ${!COMPOSE_FILES[*]})"
    continue
  fi

  if [[ "$config" == "triton" ]]; then
    benchmark_triton "$config" "$compose_file"
  else
    benchmark_fastapi "$config" "$compose_file"
  fi
done

section "All benchmarks complete"
log "Results in $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"
