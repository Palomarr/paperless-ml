#!/usr/bin/env bash
# ===========================================================================
# demo_triton.sh — Triton demo for recording (Segment 2)
#
# Run this during screen recording. It:
#   1. Shows the Triton model repo structure and configs
#   2. Launches Triton and waits for READY
#   3. Verifies both models via REST API
#   4. Runs perf_analyzer for both models
#   5. Tears down Triton
#
# Usage:
#   cd ~/paperless-ml/serving
#   bash demo_triton.sh
# ===========================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

COMPOSE_FILE="docker-compose-triton.yaml"
SDK_IMAGE="nvcr.io/nvidia/tritonserver:24.01-py3-sdk"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

section() { echo -e "\n${CYAN}${BOLD}=== $* ===${NC}\n"; sleep 1; }
log()     { echo -e "${GREEN}$*${NC}"; }

# -------------------------------------------------------------------
# Clean slate
# -------------------------------------------------------------------
docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true

# -------------------------------------------------------------------
# 1. Show model repo
# -------------------------------------------------------------------
section "Triton model repository"

find triton_model_repo -type f
sleep 2

echo ""
log "--- htr_model config ---"
cat triton_model_repo/htr_model/config.pbtxt
sleep 2

echo ""
log "--- search_model config ---"
cat triton_model_repo/search_model/config.pbtxt
sleep 2

# -------------------------------------------------------------------
# 2. Launch Triton
# -------------------------------------------------------------------
section "Launching Triton Inference Server"

docker compose -f "$COMPOSE_FILE" up -d
echo ""

log "Waiting for models to load..."
for i in $(seq 1 90); do
    if curl -sf http://localhost:8000/v2/models/htr_model &>/dev/null && \
       curl -sf http://localhost:8000/v2/models/search_model &>/dev/null; then
        log "Both models READY."
        break
    fi
    sleep 2
done
sleep 1

# -------------------------------------------------------------------
# 3. Verify models via REST API
# -------------------------------------------------------------------
section "Model verification"

echo "htr_model:"
curl -s localhost:8000/v2/models/htr_model | python3 -m json.tool
echo ""
echo "search_model:"
curl -s localhost:8000/v2/models/search_model | python3 -m json.tool
sleep 2

# -------------------------------------------------------------------
# 4. Benchmark with perf_analyzer
# -------------------------------------------------------------------
section "Benchmark: HTR model (perf_analyzer)"

docker run --rm --net=host "$SDK_IMAGE" \
    perf_analyzer -u localhost:8000 -m htr_model \
    -b 1 --shape pixel_values:3,384,384 \
    --concurrency-range 1 \
    --measurement-interval 5000

sleep 3

section "Benchmark: Search model (perf_analyzer)"

docker run --rm --net=host "$SDK_IMAGE" \
    perf_analyzer -u localhost:8000 -m search_model \
    -b 1 --shape input_ids:128 --shape attention_mask:128 \
    --concurrency-range 1 \
    --measurement-interval 5000

sleep 3

# -------------------------------------------------------------------
# 5. Tear down
# -------------------------------------------------------------------
section "Tearing down Triton"

docker compose -f "$COMPOSE_FILE" down

echo ""
log "Demo complete."
