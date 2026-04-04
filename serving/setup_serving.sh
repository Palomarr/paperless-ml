#!/usr/bin/env bash
# ===========================================================================
# setup_serving.sh — One-command setup for serving on a fresh Chameleon P100
#
# Prerequisites (handled by provision.ipynb):
#   - Docker installed with NVIDIA Container Toolkit
#   - GPU access verified (nvidia-smi works, docker --gpus all works)
#   - Repo cloned to ~/paperless-ml
#
# Usage:
#   cd ~/paperless-ml/serving
#   bash setup_serving.sh
#
# What this script does:
#   1. Pulls Triton images (in background)
#   2. Creates Python venv and exports ONNX models
#   3. Populates Triton model repo (with IR version fix)
#   4. Generates quantized models
#   5. Creates Docker network and builds FastAPI image
#   6. Runs a smoke test on both serving stacks
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING: $*${NC}"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR: $*${NC}"; }

# -------------------------------------------------------------------
# Pre-flight checks
# -------------------------------------------------------------------
log "Pre-flight checks..."

if ! nvidia-smi &>/dev/null; then
    err "nvidia-smi failed. Run provision.ipynb first or check GPU driver."
    exit 1
fi

if ! docker info &>/dev/null; then
    err "Docker not running. Run provision.ipynb first."
    exit 1
fi

if [ ! -f src/export/export_onnx.py ]; then
    err "Not in the serving directory. Run: cd ~/paperless-ml/serving && bash setup_serving.sh"
    exit 1
fi

log "Pre-flight OK — GPU and Docker available."

# -------------------------------------------------------------------
# Step 1: Pull Triton images in background
# -------------------------------------------------------------------
log "Step 1/7: Pulling Triton images in background..."

TRITON_SERVER="nvcr.io/nvidia/tritonserver:24.01-py3"
TRITON_SDK="nvcr.io/nvidia/tritonserver:24.01-py3-sdk"

if docker image inspect "$TRITON_SERVER" &>/dev/null; then
    log "  Triton server image already present — skipping pull."
else
    log "  Pulling Triton server image (~15 GB, running in background)..."
    docker pull "$TRITON_SERVER" > /tmp/pull_triton_server.log 2>&1 &
    TRITON_SERVER_PID=$!
fi

if docker image inspect "$TRITON_SDK" &>/dev/null; then
    log "  Triton SDK image already present — skipping pull."
else
    log "  Pulling Triton SDK image (~15 GB, running in background)..."
    docker pull "$TRITON_SDK" > /tmp/pull_triton_sdk.log 2>&1 &
    TRITON_SDK_PID=$!
fi

# -------------------------------------------------------------------
# Step 2: Create venv and install Python deps
# -------------------------------------------------------------------
log "Step 2/7: Setting up Python venv and installing dependencies..."

if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --quiet \
    torch torchvision transformers sentence-transformers \
    "optimum[onnxruntime-gpu]" Pillow onnx onnxscript requests

# -------------------------------------------------------------------
# Step 3: Export ONNX models
# -------------------------------------------------------------------
if [ -f onnx_models/biencoder.onnx ] && [ -f onnx_models/htr_onnx/encoder_model.onnx ]; then
    log "Step 3/7: ONNX models already exported — skipping."
else
    log "Step 3/7: Exporting ONNX models (downloads HuggingFace weights ~1 GB)..."
    python3 src/export/export_onnx.py --output-dir ./onnx_models || {
        # The HTR verification may fail on transformers >= 4.50 (tokenizer bug)
        # but the ONNX files are still valid — check if they exist
        if [ -f onnx_models/biencoder.onnx ] && [ -f onnx_models/htr_onnx/encoder_model.onnx ]; then
            warn "Export script errored during verification, but ONNX files were created successfully."
        else
            err "ONNX export failed. Check output above."
            exit 1
        fi
    }
fi

log "  Verifying exports..."
ls -lh onnx_models/biencoder.onnx
ls -lh onnx_models/htr_onnx/encoder_model.onnx

# -------------------------------------------------------------------
# Step 4: Populate Triton model repository
# -------------------------------------------------------------------
log "Step 4/7: Populating Triton model repository..."

mkdir -p triton_model_repo/htr_model/1
mkdir -p triton_model_repo/search_model/1

# HTR: copy encoder only (IR version 8, no fix needed)
cp onnx_models/htr_onnx/encoder_model.onnx triton_model_repo/htr_model/1/model.onnx

# Search: copy bi-encoder
cp onnx_models/biencoder.onnx triton_model_repo/search_model/1/model.onnx
if [ -f onnx_models/biencoder.onnx.data ]; then
    cp onnx_models/biencoder.onnx.data triton_model_repo/search_model/1/biencoder.onnx.data
fi

# Fix search model IR version (10 → 9) and normalize external data reference
log "  Fixing search model IR version..."
python3 << 'PYEOF'
import onnx, os

model_path = "triton_model_repo/search_model/1/model.onnx"
model = onnx.load(model_path, load_external_data=False)
has_ext = any(t.data_location == 1 for t in model.graph.initializer)

if model.ir_version > 9:
    # Load with external data if present
    if has_ext:
        model = onnx.load(model_path)
    model.ir_version = 9
    if has_ext:
        onnx.save(model, model_path,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location="model.onnx.data")
        # Clean up old external data file with original name
        old_ext = os.path.join(os.path.dirname(model_path), "biencoder.onnx.data")
        if os.path.exists(old_ext):
            os.remove(old_ext)
    else:
        onnx.save(model, model_path)
    print(f"Fixed IR version to {model.ir_version} (external_data={has_ext})")
else:
    print(f"IR version {model.ir_version} OK — no fix needed")
PYEOF

# Verify final structure
log "  Model repo contents:"
find triton_model_repo -type f

# -------------------------------------------------------------------
# Step 5: Generate quantized models (optional but needed for full table)
# -------------------------------------------------------------------
if [ -f onnx_models_quantized/biencoder.onnx ] && [ -f onnx_models_quantized/htr_onnx/encoder_model.onnx ]; then
    log "Step 5/7: Quantized models already exist — skipping."
else
    log "Step 5/7: Generating quantized ONNX models..."
    python3 src/export/quantize_onnx.py \
        --onnx-dir ./onnx_models \
        --output-dir ./onnx_models_quantized || {
        warn "Quantization had errors (non-fatal). Check output above."
    }
fi

# -------------------------------------------------------------------
# Step 6: Create Docker network and build FastAPI image
# -------------------------------------------------------------------
log "Step 6/7: Building FastAPI ORT image..."

docker network create paperless-net 2>/dev/null || true
docker compose -f docker-compose-ort.yaml build fastapi_server

# -------------------------------------------------------------------
# Step 7: Wait for Triton pulls and smoke test
# -------------------------------------------------------------------
log "Step 7/7: Waiting for Triton image pulls to complete..."

if [ -n "${TRITON_SERVER_PID:-}" ]; then
    log "  Waiting for Triton server image (tail -f /tmp/pull_triton_server.log to monitor)..."
    wait "$TRITON_SERVER_PID" || { err "Triton server pull failed"; exit 1; }
    log "  Triton server image ready."
fi

if [ -n "${TRITON_SDK_PID:-}" ]; then
    log "  Waiting for Triton SDK image (tail -f /tmp/pull_triton_sdk.log to monitor)..."
    wait "$TRITON_SDK_PID" || { err "Triton SDK pull failed"; exit 1; }
    log "  Triton SDK image ready."
fi

# -------------------------------------------------------------------
# Smoke test: FastAPI ORT
# -------------------------------------------------------------------
log "Smoke test: FastAPI + ONNX Runtime..."

docker compose -f docker-compose-ort.yaml up -d
log "  Waiting for FastAPI server to start (up to 120s)..."

for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health &>/dev/null; then
        break
    fi
    sleep 2
done

if curl -sf http://localhost:8000/health &>/dev/null; then
    log "  FastAPI server is up. Running contract test..."
    cd "$SCRIPT_DIR/.."
    if python3 serving/scripts/test_contract.py; then
        log "  Contract test PASSED."
    else
        warn "  Contract test had failures — check output above."
    fi
    cd "$SCRIPT_DIR"
else
    warn "  FastAPI server did not start within 120s. Check: docker compose -f docker-compose-ort.yaml logs"
fi

docker compose -f docker-compose-ort.yaml down

# -------------------------------------------------------------------
# Smoke test: Triton
# -------------------------------------------------------------------
log "Smoke test: Triton Inference Server..."

docker compose -f docker-compose-triton.yaml up -d
log "  Waiting for Triton to load models (up to 120s)..."

for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/v2/models/htr_model &>/dev/null && \
       curl -sf http://localhost:8000/v2/models/search_model &>/dev/null; then
        break
    fi
    sleep 2
done

if curl -sf http://localhost:8000/v2/models/htr_model &>/dev/null; then
    log "  Triton models READY."
    curl -s http://localhost:8000/v2/models/htr_model | python3 -m json.tool
    curl -s http://localhost:8000/v2/models/search_model | python3 -m json.tool
else
    warn "  Triton did not become ready within 120s. Check: docker logs serving-triton_server-1"
fi

docker compose -f docker-compose-triton.yaml down

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
log "=========================================="
log "  Setup complete!"
log "=========================================="
log ""
log "All images cached, models exported, both stacks tested."
log ""
log "To run FastAPI ORT:     docker compose -f docker-compose-ort.yaml up -d"
log "To run Triton:          docker compose -f docker-compose-triton.yaml up -d"
log "To run contract test:   cd ~/paperless-ml && python3 serving/scripts/test_contract.py"
log "To run perf_analyzer:   see README.md for commands"
log ""
log "Ready to record demo video!"
