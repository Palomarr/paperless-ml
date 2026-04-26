#!/usr/bin/env bash
# ============================================================================
# check_model_state.sh — read-only audit of HTR + bi-encoder model state.
#
# Answers: "Is ml-gateway serving REAL fine-tuned models from Elnath's
# Airflow training pipeline, or STOCK HuggingFace fallbacks?"
#
# Five layers of evidence:
#   1. MinIO production prefix    — destination for promoted models
#   2. MLflow registry            — what's been quality-gated and registered
#   3. ml-gateway /models/        — what's actually loaded in the container
#   4. ml-gateway boot logs       — fallback path vs real fetch
#   5. Pipeline-scheduler counters — has the retraining loop ever triggered?
#
# Usage: bash scripts/check_model_state.sh
# Read-only, safe to run any time, exits 0 always (diagnostic, not pass/fail).
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---- output helpers ----
if [[ -t 1 ]]; then
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_GREEN=""; C_RED=""; C_YELLOW=""; C_BOLD=""; C_RESET=""
fi
info()  { printf "%s==>%s %s\n" "$C_BOLD" "$C_RESET" "$*"; }
real()  { printf " %sREAL%s  %s\n" "$C_GREEN" "$C_RESET" "$*"; }
stock() { printf " %sSTOCK%s %s\n" "$C_YELLOW" "$C_RESET" "$*"; }
miss()  { printf " %sMISS%s  %s\n" "$C_RED" "$C_RESET" "$*"; }

# ---- preflight ----
command -v docker >/dev/null || { echo "docker not found"; exit 2; }
docker compose ps -q ml-gateway >/dev/null 2>&1 \
    || { echo "ml-gateway not running — bring stack up first (bash scripts/chameleon_setup.sh)"; exit 2; }

# Track per-layer verdict (real|stock|unknown). L3 records present|missing.
declare -A VERDICT

# ---- Layer 1: MinIO production prefix ----
info "Layer 1: MinIO production prefix (s3://paperless-datalake/warehouse/models/htr/production/)"
docker compose exec -T minio mc alias set local http://minio:9000 minioadmin minioadmin >/dev/null 2>&1 || true
prod_listing="$(docker compose exec -T minio mc ls --recursive \
    local/paperless-datalake/warehouse/models/htr/production/ 2>/dev/null || echo '')"
prod_count=$(echo "$prod_listing" | grep -cE "encoder_model\.onnx|decoder_model\.onnx" || true)
if (( prod_count >= 2 )); then
    real "production prefix has $prod_count ONNX file(s) — real promoted model"
    VERDICT[L1]="real"
else
    stock "production prefix has no encoder/decoder ONNX (fresh-node fallback expected)"
    VERDICT[L1]="stock"
fi
echo "$prod_listing" | head -10 | sed 's/^/    /'

# ---- Layer 2: MLflow registry ----
info "Layer 2: MLflow registered model 'htr'"
mlflow_versions="$(curl -fsS "http://localhost:5050/api/2.0/mlflow/registered-models/get?name=htr" 2>/dev/null \
    | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    versions = d.get("registered_model", {}).get("latest_versions", [])
    if versions:
        for v in versions:
            print(f"  v{v[\"version\"]} aliases={v.get(\"aliases\", [])} run_id={v.get(\"run_id\", \"\")[:12]}")
    else:
        print("NONE")
except Exception:
    print("NONE")
' 2>/dev/null || echo "NONE")"
if [[ "$mlflow_versions" == "NONE" ]] || [[ -z "$mlflow_versions" ]]; then
    stock "no 'htr' model registered (Airflow htr_training hasn't completed + gated)"
    VERDICT[L2]="stock"
else
    real "'htr' registered with version(s):"
    echo "$mlflow_versions" | sed 's/^/    /'
    VERDICT[L2]="real"
fi

# ---- Layer 3: ml-gateway /models/ contents (presence check, not real-vs-stock) ----
info "Layer 3: ml-gateway container /models/ contents"
models_listing="$(docker compose exec -T ml-gateway ls -la /models/ 2>/dev/null || echo '')"
if echo "$models_listing" | grep -q "encoder_model.onnx"; then
    real "ml-gateway has loaded model files (real or stock-exported)"
    echo "$models_listing" | head -15 | sed 's/^/    /'
    VERDICT[L3]="present"
else
    miss "/models/ missing expected ONNX files — ml-gateway not serving"
    VERDICT[L3]="missing"
fi

# ---- Layer 4: boot logs ----
info "Layer 4: ml-gateway boot path (fallback vs real fetch)"
boot_logs="$(docker compose logs ml-gateway 2>&1 \
    | grep -iE "HTR_MODEL_URI|fallback|Bi-encoder ONNX (exported|missing|downloaded)|HTR ONNX downloaded|microsoft/trocr|all-mpnet|production sync" \
    | head -10 || echo '')"
echo "$boot_logs" | sed 's/^/    /'
if echo "$boot_logs" | grep -q "falling back to stock"; then
    stock "boot took fresh-node fallback path (HF stock model exported at runtime)"
    VERDICT[L4]="stock"
elif echo "$boot_logs" | grep -q "HTR ONNX downloaded"; then
    real "boot fetched HTR ONNX from MinIO production prefix"
    VERDICT[L4]="real"
else
    info "boot path unclear from logs (no diagnostic line found)"
    VERDICT[L4]="unknown"
fi

# ---- Layer 5: pipeline-scheduler counters ----
info "Layer 5: pipeline-scheduler decision history"
sched_metrics="$(docker compose exec -T pipeline-scheduler python3 -c \
    "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/metrics').read().decode())" 2>/dev/null \
    | grep -E "^(training_quality_gate|pipeline_scheduler_decisions|pipeline_scheduler_last)" \
    | head -15 || echo '')"
echo "$sched_metrics" | sed 's/^/    /'
gate_passes_int=$(echo "$sched_metrics" | grep "^training_quality_gate_passes_total" | head -1 | awk '{print int($NF)}' || echo 0)
last_trigger_int=$(echo "$sched_metrics" | grep "^pipeline_scheduler_last_trigger_timestamp_seconds" | awk '{print int($NF)}' || echo 0)
if (( ${gate_passes_int:-0} >= 1 )); then
    real "pipeline-scheduler has gated $gate_passes_int successful run(s)"
    VERDICT[L5]="real"
elif (( ${last_trigger_int:-0} > 0 )); then
    stock "scheduler triggered but no gate-passing runs yet"
    VERDICT[L5]="stock"
else
    stock "scheduler never triggered (corrections + time thresholds not yet met)"
    VERDICT[L5]="stock"
fi

# ---- Aggregate verdict ----
echo
real_count=0; stock_count=0
for layer in L1 L2 L4 L5; do  # L3 is presence-only, doesn't distinguish real/stock
    case "${VERDICT[$layer]:-unknown}" in
        real)  real_count=$((real_count + 1)) ;;
        stock) stock_count=$((stock_count + 1)) ;;
    esac
done

if (( real_count >= 3 )); then
    real "Aggregate: REAL fine-tuned model deployed ($real_count/4 layers confirm)"
elif (( stock_count >= 3 )); then
    stock "Aggregate: STOCK HuggingFace fallback ($stock_count/4 layers confirm)"
    echo "    For real models: Elnath's Airflow htr_training DAG must complete a"
    echo "    gate-passing run, then pipeline-scheduler.promote_latest() syncs"
    echo "    artifacts to the production prefix on next ml-gateway restart."
else
    info "Aggregate: MIXED state ($real_count real / $stock_count stock) — investigate per-layer output above"
fi
