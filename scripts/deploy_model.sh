#!/usr/bin/env bash
# ============================================================================
# deploy_model.sh — promote a registered MLflow model into the serving path.
#
# Pipeline:
#   (1) query MLflow for the latest registered version of MODEL_NAME
#   (2) stage the model artifacts from MLflow's artifact store (MinIO under
#       s3://paperless-datalake/mlflow-artifacts/) into a local dir
#   (3) copy the staged artifacts into the serving model directory on MinIO:
#       s3://paperless-datalake/warehouse/models/v<N>/
#   (4) signal ml-gateway to reload (restart for now; /admin/reload-model
#       endpoint is a future enhancement — see NOTES below)
#
# Usage:
#   bash scripts/deploy_model.sh                  # deploy latest registered version
#   bash scripts/deploy_model.sh --dry-run        # print plan without making changes
#   MODEL_NAME=paperless-retrieval bash scripts/deploy_model.sh
#
# Requires:
#   - docker compose stack up (mlflow service healthy)
#   - at least one model registered in MLflow under MODEL_NAME (see below)
#
# Registering a model from training code (for reference):
#   import mlflow
#   with mlflow.start_run() as run:
#       ...train...
#       mlflow.pytorch.log_model(model, "model", registered_model_name="paperless-htr")
#       # OR, gated by quality check:
#       if cer < 0.15:
#           mlflow.register_model(f"runs:/{run.info.run_id}/model", "paperless-htr")
#
# NOTES / deferred work:
#   * ml-gateway does not yet expose POST /admin/reload-model. For the Phase 2
#     demo we use `docker compose restart ml-gateway` to force weight reload,
#     which takes ~90 seconds (TrOCR + mpnet warm-up). Adding a dynamic reload
#     endpoint that atomically swaps ONNX weights without restart is tracked
#     as a follow-up — blocks on the actual on-disk model layout ml-gateway
#     uses. For Phase 2 the mechanism is demonstrable end-to-end.
#   * serving/src/export/export_onnx.py exports the pre-trained base models
#     (microsoft/trocr-small-handwritten, sentence-transformers/all-mpnet-base-v2)
#     from HuggingFace. If training registers a fine-tuned .pth state_dict,
#     the export step below needs a task-specific converter. For the current
#     demo we assume the registered artifact is already ONNX (or MLflow's
#     `mlflow.onnx.log_model` was used), and passthrough-copy.
# ============================================================================

set -euo pipefail

# ---------- config (all env-overridable) ----------
MODEL_NAME="${MODEL_NAME:-paperless-htr}"
WAREHOUSE_BUCKET="${WAREHOUSE_BUCKET:-paperless-datalake}"
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            grep '^#' "$0" | head -40
            exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

# ---------- output helpers ----------
C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
log()   { printf "%s==>%s %s\n"   "$C_BOLD"   "$C_RESET" "$*"; }
ok()    { printf "%s ✓ %s %s\n"   "$C_GREEN"  "$C_RESET" "$*"; }
warn()  { printf "%s !  %s %s\n"  "$C_YELLOW" "$C_RESET" "$*"; }
fail()  { printf "%s ✗ %s %s\n"   "$C_RED"    "$C_RESET" "$*"; exit 1; }

# ---------- preflight ----------
log "Preflight: checking stack is up and mlflow is healthy"
if ! docker compose ps --status=running --services 2>/dev/null | grep -q '^mlflow$'; then
    fail "mlflow service not running. Start the stack first: docker compose up -d"
fi
ok "mlflow service is up"

# ---------- step 1: find latest registered version ----------
log "Step 1/4: querying MLflow for latest registered version of '${MODEL_NAME}'"

# Use the mlflow container's Python env (already has mlflow + boto3 installed).
# `-T` disables TTY so we can capture stdout into a shell variable.
VERSION=$(docker compose exec -T mlflow python - <<PYEOF 2>/dev/null
import sys
from mlflow import MlflowClient
client = MlflowClient(tracking_uri="http://localhost:5000")
versions = client.search_model_versions(f"name='${MODEL_NAME}'")
if not versions:
    sys.stderr.write("no registered versions\n")
    sys.exit(2)
latest = max(versions, key=lambda v: int(v.version))
print(latest.version, end='')
PYEOF
) || {
    fail "No registered versions of '${MODEL_NAME}' in MLflow. Train + register a model first.
    Example (from training code):
      mlflow.pytorch.log_model(model, 'model', registered_model_name='${MODEL_NAME}')"
}
ok "latest version: v${VERSION}"

if (( DRY_RUN )); then
    log "DRY RUN — remaining steps would:"
    echo "    (2) stage artifacts from MLflow to /tmp/model-v${VERSION} inside mlflow container"
    echo "    (3) copy to s3://${WAREHOUSE_BUCKET}/warehouse/models/v${VERSION}/"
    echo "    (4) restart ml-gateway to pick up the new model"
    exit 0
fi

# ---------- step 2: stage artifacts from MLflow ----------
log "Step 2/4: staging artifacts from MLflow to local /tmp/model-v${VERSION}"
docker compose exec -T mlflow python - <<PYEOF
import os, shutil, sys
import mlflow.artifacts
from mlflow import MlflowClient
client = MlflowClient(tracking_uri="http://localhost:5000")
vinfo = client.get_model_version("${MODEL_NAME}", "${VERSION}")

# Clean + re-stage so repeated runs don't mix older files in.
dst = f"/tmp/model-v${VERSION}"
if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst, exist_ok=True)

# vinfo.source can be 'runs:/<run_id>/<path>' (MLflow 2.x) OR
# 'models:/m-<UUID>' (MLflow 3.x first-class logged models). download_artifacts
# resolves both schemes; using client.download_artifacts(run_id, "", ...)
# only walks the run's artifact path which is empty in the 3.x scheme.
local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=vinfo.source,
    dst_path=dst,
)
print(f"staged run_id={vinfo.run_id} source={vinfo.source} -> {local_path}")

staged_count = 0
for root, _, files in os.walk(dst):
    for f in files:
        full = os.path.join(root, f)
        rel = os.path.relpath(full, dst)
        size = os.path.getsize(full)
        print(f"  {rel} ({size} bytes)")
        staged_count += 1

if staged_count == 0:
    sys.stderr.write(f"ERROR: 0 files staged from {vinfo.source}\n")
    sys.exit(1)
print(f"staged {staged_count} files")
PYEOF
ok "artifacts staged"

# ---------- step 3: copy to MinIO serving path ----------
log "Step 3/4: copying to s3://${WAREHOUSE_BUCKET}/warehouse/models/v${VERSION}/"

# Upload in-place from the mlflow container using boto3 (already pip-installed
# by our compose command). Avoids cross-container plumbing — minio/mc is a
# scratch image without tar, and adding tar to it would mean a custom image.
docker compose exec -T mlflow python - <<PYEOF
import boto3, os, sys
s3 = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)
src = f"/tmp/model-v${VERSION}"
prefix = f"warehouse/models/v${VERSION}/"
bucket = "${WAREHOUSE_BUCKET}"
uploaded = 0
for root, _, files in os.walk(src):
    for f in files:
        fp = os.path.join(root, f)
        key = prefix + os.path.relpath(fp, src).replace(os.sep, "/")
        s3.upload_file(fp, bucket, key)
        print(f"  uploaded {key}")
        uploaded += 1
if uploaded == 0:
    sys.stderr.write(f"ERROR: 0 objects uploaded from {src} — staging dir empty\n")
    sys.exit(1)
print(f"total: {uploaded} objects under s3://{bucket}/{prefix}")
PYEOF
ok "uploaded to s3://${WAREHOUSE_BUCKET}/warehouse/models/v${VERSION}/"

# ---------- step 4: signal ml-gateway to reload ----------
log "Step 4/4: signalling ml-gateway to reload"
warn "ml-gateway /admin/reload-model is not implemented yet — using restart fallback"
docker compose restart ml-gateway
log "    waiting for ml-gateway /health to come back (cold-start ~90s)"
ATTEMPTS=0
until docker compose exec -T ml-gateway curl -fsS http://localhost:8000/health >/dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if (( ATTEMPTS > 30 )); then
        fail "ml-gateway did not become healthy after restart. Check: docker compose logs --tail=100 ml-gateway"
    fi
    sleep 5
done
ok "ml-gateway healthy"

echo
ok "Deployed ${MODEL_NAME} v${VERSION} to s3://${WAREHOUSE_BUCKET}/warehouse/models/v${VERSION}/"
echo
echo "Verify in MinIO console: http://<floating-ip>:9001/browser/${WAREHOUSE_BUCKET}/warehouse%2Fmodels%2Fv${VERSION}/"
echo "Verify in MLflow UI:    http://<floating-ip>:5050/#/models/${MODEL_NAME}/versions/${VERSION}"
