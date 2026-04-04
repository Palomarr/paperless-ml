#!/usr/bin/env bash
# ===========================================================================
# demo_fastapi.sh — FastAPI contract validation demo for recording (Segment 1)
#
# Run this during screen recording. It:
#   1. Shows the agreed JSON contract files
#   2. Launches FastAPI + ONNX Runtime
#   3. Runs the contract test (expects PASS/PASS)
#   4. Tears down
#
# Usage:
#   cd ~/paperless-ml/serving
#   bash demo_fastapi.sh
# ===========================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

COMPOSE_FILE="docker-compose-ort.yaml"

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
# 1. Show JSON contracts
# -------------------------------------------------------------------
section "Agreed JSON contracts"

log "--- HTR input ---"
cat ../contracts/htr_input.json
echo ""
sleep 2

log "--- Search input ---"
cat ../contracts/search_input.json
echo ""
sleep 2

# -------------------------------------------------------------------
# 2. Launch FastAPI + ONNX Runtime
# -------------------------------------------------------------------
section "Launching FastAPI + ONNX Runtime"

docker compose -f "$COMPOSE_FILE" up -d
echo ""

log "Waiting for FastAPI server to start..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health &>/dev/null; then
        log "FastAPI server is up."
        break
    fi
    sleep 2
done
sleep 1

# -------------------------------------------------------------------
# 3. Contract test
# -------------------------------------------------------------------
section "Contract validation"

cd ..
python3 serving/scripts/test_contract.py
cd serving

sleep 3

# -------------------------------------------------------------------
# 4. Tear down
# -------------------------------------------------------------------
section "Tearing down FastAPI"

docker compose -f "$COMPOSE_FILE" down

echo ""
log "Segment 1 complete."
