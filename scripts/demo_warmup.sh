#!/usr/bin/env bash
# ============================================================================
# demo_warmup.sh — bring a fresh Chameleon node from chameleon_setup.sh-green
# to demo-ready. Idempotent; re-run safe.
#
# Runs AFTER scripts/chameleon_setup.sh has succeeded (our compose stack up,
# peer repos cloned at ~/paperless_data, ~/paperless_data_integration,
# ~/paperless_training_integration).
#
# Wraps every workaround documented in docs/KNOWN_GAPS.md:
#   - sed patch to Elnath's build_drift_reference.py (upstream bug #2)
#   - inline pip install Pillow for batch_pipeline (upstream bug #1)
#   - inline pip install pyarrow for drift_monitor image (expected; not a bug)
#   - synthetic manifest.json upload for bullet 4 (scope disclosure)
#
# Usage:
#   bash scripts/demo_warmup.sh                  # full warm-up (~20 min cold)
#   bash scripts/demo_warmup.sh --skip-ingest    # skip IAM ingest (re-runs)
#   bash scripts/demo_warmup.sh --only-seed      # re-seed MLflow + audit only
#
# Expected runtime on a fresh node:
#   IAM ingest         ~5 min
#   drift_monitor build ~2 min
#   drift reference     ~2 min
#   batch_pipeline build ~2 min
#   everything else    <2 min
# ============================================================================

set -u  # unset vars are errors; set -e handled via trap for better messages

# ── Config ──────────────────────────────────────
PAPERLESS_ML=${PAPERLESS_ML:-"$HOME/paperless-ml"}
DATA_REPO=${DATA_REPO:-"$HOME/paperless_data"}
DATA_INT_REPO=${DATA_INT_REPO:-"$HOME/paperless_data_integration"}
TRAIN_REPO=${TRAIN_REPO:-"$HOME/paperless_training_integration"}
NETWORK=${NETWORK:-"paperless_ml_net"}

SKIP_INGEST=0
ONLY_SEED=0
for arg in "$@"; do
    case "$arg" in
        --skip-ingest) SKIP_INGEST=1 ;;
        --only-seed)   ONLY_SEED=1 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Output helpers ──────────────────────────────
C_GREEN=$'\033[32m'
C_YELLOW=$'\033[33m'
C_RED=$'\033[31m'
C_BOLD=$'\033[1m'
C_DIM=$'\033[2m'
C_RESET=$'\033[0m'

log()  { printf "\n%s==>%s %s%s%s\n" "$C_BOLD" "$C_RESET" "$C_BOLD" "$*" "$C_RESET"; }
ok()   { printf "%s  ✓%s %s\n" "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf "%s  !%s %s\n" "$C_YELLOW" "$C_RESET" "$*"; }
step() { printf "%s   ∙%s %s\n" "$C_DIM" "$C_RESET" "$*"; }
fail() { printf "%s  ✗%s %s\n" "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

CURRENT_STEP=""
# Capture $? FIRST before running any test — otherwise [[ ]] clobbers it and
# the error message reports "(exit 0)" regardless of actual failure code.
trap 'rc=$?; [[ -n "$CURRENT_STEP" ]] && fail "FAILED at step: $CURRENT_STEP (exit $rc)"' ERR
set -e

# ── Preflight ───────────────────────────────────
log "Preflight"
CURRENT_STEP="preflight"

[[ -d "$PAPERLESS_ML" ]] || fail "$PAPERLESS_ML missing — run chameleon_setup.sh first"
[[ -d "$DATA_REPO" ]] || fail "$DATA_REPO missing"
[[ -d "$DATA_INT_REPO" ]] || fail "$DATA_INT_REPO missing"
[[ -d "$TRAIN_REPO" ]] || fail "$TRAIN_REPO missing"

cd "$PAPERLESS_ML"
if ! sg docker -c 'docker compose ps --services --status=running' | grep -q ml-gateway; then
    fail "our stack isn't running — run chameleon_setup.sh first"
fi

if ! sg docker -c "docker network inspect $NETWORK" >/dev/null 2>&1; then
    fail "network $NETWORK missing — chameleon_setup.sh should have created it"
fi
ok "paperless-ml stack up, $NETWORK present"

# --only-seed shortcut skips the heavy Elnath chain
if (( ONLY_SEED == 1 )); then
    log "Only-seed mode — jumping to MLflow seed + fixture upload"
    goto_seed=1
else
    goto_seed=0
fi

# ── 0. Pre-build source patches for upstream bugs ───
if (( goto_seed == 0 )); then
log "[0/12] Pre-build source patches for upstream bugs"
CURRENT_STEP="pre-build patches"

# Dongting's training repo hardcodes http://127.0.0.1:5000 for MLflow in
# trainer/mlflow_helper.py, eval.py, quality_gate.py. Our pipeline-scheduler
# passes MLFLOW_TRACKING_URI=http://mlflow:5000 but the code ignores the
# env. Patch all three files in place so the image we build picks up the
# compose-network hostname. Upstream fix is in Yikai's PR #1 on
# gdtmax/paperless_training_integration — remove this block once merged.
TRAIN_HARDCODED=$(grep -rl 'http://127.0.0.1:5000' "$TRAIN_REPO" --include='*.py' 2>/dev/null || true)
if [[ -n "$TRAIN_HARDCODED" ]]; then
    # shellcheck disable=SC2086
    sed -i 's|http://127.0.0.1:5000|http://mlflow:5000|g' $TRAIN_HARDCODED
    ok "patched $(echo "$TRAIN_HARDCODED" | wc -l) training-repo files (upstream PR gdtmax#1)"
else
    ok "training repo already uses http://mlflow:5000 (Dongting merged PR #1)"
fi

# ── 1. Build training + data-generator images ───
log "[1/12] Build training + data-generator images (parallel)"
CURRENT_STEP="build training/data-generator images"

if sg docker -c 'docker image inspect paperless-training' >/dev/null 2>&1; then
    ok "paperless-training already built"
else
    step "building paperless-training"
    (cd "$TRAIN_REPO" && sg docker -c 'docker build -q -t paperless-training .' > /dev/null) &
    TRAIN_PID=$!
fi

if sg docker -c 'docker image inspect paperless-data-generator' >/dev/null 2>&1; then
    ok "paperless-data-generator already built"
else
    step "building paperless-data-generator"
    (cd "$DATA_REPO/data_generator" && sg docker -c 'docker build -q -t paperless-data-generator .' > /dev/null) &
    DG_PID=$!
fi

[[ -n "${TRAIN_PID:-}" ]] && wait $TRAIN_PID && ok "paperless-training built"
[[ -n "${DG_PID:-}" ]]    && wait $DG_PID    && ok "paperless-data-generator built"

# ── 2. Patch Elnath's build_drift_reference.py (upstream bug #2) ──
log "[2/12] Patch build_drift_reference.py — image_bytes → image_png"
CURRENT_STEP="sed patch drift reference builder"

BUILDER="$DATA_REPO/scripts/build_drift_reference.py"
if grep -q 'table.column("image_bytes")' "$BUILDER"; then
    sed -i 's/table.column("image_bytes")/table.column("image_png")/' "$BUILDER"
    ok "patched — KNOWN_GAPS bug #2 worked around locally"
elif grep -q 'table.column("image_png")' "$BUILDER"; then
    ok "already patched (either we ran earlier or Elnath fixed upstream)"
else
    warn "unexpected column reference — $BUILDER may have diverged; inspect manually"
fi

# ── 3. Ingest IAM (direct docker run, not `make`) ──
if (( SKIP_INGEST == 1 )); then
    log "[3/12] IAM ingest — SKIPPED per --skip-ingest"
else
    log "[3/12] Ingest IAM to MinIO warehouse (~5 min)"
    CURRENT_STEP="ingest IAM"

    # Check if already ingested
    if sg docker -c "docker compose exec -T minio mc ls local/paperless-datalake/warehouse/iam_dataset/train/" 2>/dev/null | grep -q parquet; then
        ok "IAM already ingested"
    else
        step "building paperless-ingest image"
        (cd "$DATA_REPO" && sg docker -c 'docker build -q -t paperless-ingest ./ingestion' > /dev/null)

        step "running ingest_iam.py on $NETWORK"
        sg docker -c "docker run --rm --network $NETWORK \
          -e MINIO_ENDPOINT=minio:9000 \
          -e MINIO_ACCESS_KEY=minioadmin \
          -e MINIO_SECRET_KEY=minioadmin \
          paperless-ingest python ingest_iam.py"
        ok "IAM ingested"
    fi
fi

# ── 4. Build drift_monitor image ────────────────
log "[4/12] Build drift_monitor image (~2 min)"
CURRENT_STEP="build drift_monitor image"

if sg docker -c 'docker image inspect drift_monitor:latest' >/dev/null 2>&1; then
    ok "drift_monitor:latest already built"
else
    step "building drift_monitor:latest"
    (cd "$DATA_INT_REPO" && sg docker -c 'docker build -q -f drift_monitor/Dockerfile -t drift_monitor:latest .' > /dev/null)
    ok "drift_monitor:latest built"
fi

# ── 5. Build drift reference + upload to MinIO ──
log "[5/12] Build drift reference detector (~2 min)"
CURRENT_STEP="build drift reference"

# Check if detector already in MinIO
if sg docker -c "docker compose exec -T minio mc ls local/paperless-datalake/warehouse/drift_reference/htr_v1/cd/" 2>/dev/null | grep -q .; then
    ok "drift reference already uploaded"
else
    step "running build_drift_reference.py inside drift_monitor container"
    cd "$DATA_REPO"
    sg docker -c "docker run --rm --network $NETWORK \
      -v \"$PWD/scripts:/scripts:ro\" \
      -e MINIO_ENDPOINT=minio:9000 \
      -e MINIO_ACCESS_KEY=minioadmin \
      -e MINIO_SECRET_KEY=minioadmin \
      --entrypoint sh drift_monitor:latest \
      -c 'pip install --quiet pyarrow==18.1.0 && python /scripts/build_drift_reference.py'"
    ok "drift reference uploaded"
    cd "$PAPERLESS_ML"
fi

# ── 6. Build batch_pipeline image ───────────────
log "[6/12] Build paperless-batch image"
CURRENT_STEP="build paperless-batch image"

if sg docker -c 'docker image inspect paperless-batch' >/dev/null 2>&1; then
    ok "paperless-batch already built"
else
    (cd "$DATA_REPO" && sg docker -c 'docker build -q -t paperless-batch ./batch_pipeline' > /dev/null)
    ok "paperless-batch built"
fi

# ── 7. Run validate_ingestion (inline Pillow, upstream bug #1) ──
log "[7/12] Run validate_ingestion.py (expected I2 FAIL — narratable)"
CURRENT_STEP="validate ingestion"

sg docker -c "docker run --rm --network $NETWORK \
  -e MINIO_ENDPOINT=minio:9000 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin \
  -w /app --entrypoint sh paperless-batch \
  -c 'pip install --quiet Pillow==11.0.0 && python validate_ingestion.py'" || \
    warn "validate_ingestion returned non-zero (expected for I2 schema FAIL)"

ok "validation reports written to MinIO _validation/ prefixes"

# ── 8. Synthesize bullet-4 manifest.json ────────
log "[8/12] Upload synthetic manifest.json (KNOWN_GAPS bug #5 scope)"
CURRENT_STEP="upload synthetic manifest"

TS=$(date -u +%Y%m%d_%H%M%S)
cat > /tmp/manifest_${TS}.json <<EOF
{
  "version": "v_${TS}",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "htr_corrections",
  "shards": [],
  "data_quality": {
    "total_candidates": 184,
    "accepted": 142,
    "rejected": 42,
    "rejection_counts": {
      "R1_no_op_correction":      18,
      "R2_empty_after_strip":     7,
      "R3_crop_url_invalid":      2,
      "R4_crop_missing_in_minio": 4,
      "R5_correction_too_long":   11
    },
    "rejection_samples": {
      "R1_no_op_correction": [
        {"correction_id": "c-0042", "region_id": "r-0118", "crop_s3_url": "s3://paperless-images/documents/24/regions/p1_r0.png", "original_text": "invoice #2041", "corrected_text": "invoice #2041"},
        {"correction_id": "c-0061", "region_id": "r-0133", "crop_s3_url": "s3://paperless-images/documents/31/regions/p1_r1.png", "original_text": "paid", "corrected_text": "paid"}
      ],
      "R2_empty_after_strip": [
        {"correction_id": "c-0074", "region_id": "r-0155", "crop_s3_url": "s3://paperless-images/documents/37/regions/p2_r0.png", "original_text": "March 2026", "corrected_text": "   "}
      ],
      "R5_correction_too_long": [
        {"correction_id": "c-0091", "region_id": "r-0188", "crop_s3_url": "s3://paperless-images/documents/44/regions/p1_r2.png", "original_text": "Signed,", "corrected_text": "$(python3 -c 'print("x" * 80)')"}
      ]
    }
  }
}
EOF

sg docker -c "docker compose exec -T minio sh -c 'mc alias set local http://minio:9000 minioadmin minioadmin >/dev/null && mc pipe local/paperless-datalake/warehouse/htr_training/v_${TS}/manifest.json'" < /tmp/manifest_${TS}.json
rm -f /tmp/manifest_${TS}.json
ok "synthetic manifest uploaded: warehouse/htr_training/v_${TS}/manifest.json"

# ── 9. Bring up drift_monitor stack ─────────────
log "[9/12] Bring up drift_monitor service"
CURRENT_STEP="bring up drift_monitor"

if sg docker -c 'docker ps --format "{{.Names}}"' | grep -q "^drift_monitor$"; then
    ok "drift_monitor already running"
else
    (cd "$DATA_INT_REPO" && sg docker -c 'docker compose -f drift_monitor/compose.yml up -d')
    step "waiting 15s for detector load + first scrape"
    sleep 15
    ok "drift_monitor up — metrics at :8100/metrics"
fi

# ── 10. Import Grafana drift dashboard ──────────
log "[10/12] Import drift dashboard into Grafana"
CURRENT_STEP="import Grafana dashboard"

DASH_JSON="$DATA_INT_REPO/drift_monitor/grafana_dashboard.json"
[[ -f "$DASH_JSON" ]] || fail "Grafana dashboard JSON missing at $DASH_JSON"

RESP=$(curl -sS -u admin:admin -X POST http://localhost:3000/api/dashboards/db \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json
with open('$DASH_JSON') as f:
    d = json.load(f)
d.pop('id', None)
print(json.dumps({'dashboard': d, 'overwrite': True, 'folderId': 0}))
")")

if echo "$RESP" | grep -q '"status":"success"'; then
    DASH_URL=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('url','?'))")
    ok "dashboard imported: http://<floating-ip>:3000${DASH_URL}"
else
    warn "import response: $RESP"
fi
fi  # end of the big "if goto_seed == 0" block

# ── 11. Seed MLflow with 2 stub versions + set @production ──
log "[11/12] Seed MLflow paperless-htr (≥2 versions for bullets 5+6)"
CURRENT_STEP="seed MLflow"

cd "$PAPERLESS_ML"
sg docker -c "docker compose exec -T mlflow python - <<'PY'
import mlflow, mlflow.pyfunc
from mlflow import MlflowClient

class Stub(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None): return model_input

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('seed-for-demo')
c = MlflowClient('http://localhost:5000')

current = len(c.search_model_versions(\"name='paperless-htr'\"))
for _ in range(max(0, 2 - current)):
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path='model', python_model=Stub(),
            registered_model_name='paperless-htr',
        )

versions = sorted(c.search_model_versions(\"name='paperless-htr'\"), key=lambda v: int(v.version))
print(f'versions: {[v.version for v in versions]}')
c.set_registered_model_alias('paperless-htr', 'production', versions[-1].version)
print(f'@production: v{versions[-1].version}')
PY
"
ok "MLflow seeded"

# ── 11a. Extract an IAM crop as a handwritten fixture for bullet 2 ──
# Elnath's data_generator/fixtures/ directory doesn't exist anymore (his
# refactor pulls fixtures from MinIO at runtime). Pull one real IAM line
# image out of the parquet we just ingested and stage it at
# /tmp/handwritten_sample.png so step 12 can pre-upload it.
log "[11a/12] Stage IAM handwritten fixture for bullet 2 pre-upload"
CURRENT_STEP="extract IAM fixture"

FIXTURE_PATH="/tmp/handwritten_sample.png"
if [[ -f "$FIXTURE_PATH" && -s "$FIXTURE_PATH" ]]; then
    ok "fixture already staged at $FIXTURE_PATH"
else
    sg docker -c "docker run --rm --network $NETWORK \
      -v /tmp:/out \
      -e MINIO_ENDPOINT=minio:9000 \
      -e MINIO_ACCESS_KEY=minioadmin \
      -e MINIO_SECRET_KEY=minioadmin \
      -w /app --entrypoint sh paperless-batch -c '
pip install --quiet Pillow==11.0.0 > /dev/null 2>&1
python - <<PY
import io, pyarrow.parquet as pq
from minio import Minio
mc = Minio(\"minio:9000\", access_key=\"minioadmin\", secret_key=\"minioadmin\", secure=False)
shards = [o.object_name for o in mc.list_objects(\"paperless-datalake\",
    prefix=\"warehouse/iam_dataset/train/\", recursive=True)
    if o.object_name.endswith(\".parquet\")]
if not shards:
    raise SystemExit(\"no IAM shards under warehouse/iam_dataset/train/\")
resp = mc.get_object(\"paperless-datalake\", shards[0])
try:
    data = resp.read()
finally:
    resp.close(); resp.release_conn()
first_png = pq.read_table(io.BytesIO(data)).column(\"image_png\").to_pylist()[0]
with open(\"/out/handwritten_sample.png\", \"wb\") as f:
    f.write(first_png)
print(f\"saved /out/handwritten_sample.png ({len(first_png)} bytes) from {shards[0]}\")
PY'" > /dev/null 2>&1
    [[ -f "$FIXTURE_PATH" && -s "$FIXTURE_PATH" ]] \
        || fail "IAM fixture extraction produced no output at $FIXTURE_PATH"
    ok "IAM fixture staged at $FIXTURE_PATH ($(stat -c%s "$FIXTURE_PATH") bytes)"
fi

# ── 11b. Ensure OOD samples exist in MinIO for bullet 7c drift demo ──
# Previous demo uploads are gone on fresh nodes. Generate 5 typed-text
# PNGs (64×512 matches the drift_monitor crop size) and upload to
# s3://paperless-images/ood/ so bullet 7c's injection loop has payloads.
log "[11b/12] Ensure OOD samples staged in MinIO"
CURRENT_STEP="OOD samples synthesis"

OOD_COUNT=$(sg docker -c "docker compose exec -T minio mc ls local/paperless-images/ood/" 2>/dev/null \
    | grep -c '\.png' || true)
if (( OOD_COUNT >= 3 )); then
    ok "OOD samples present: $OOD_COUNT objects"
else
    sg docker -c "docker run --rm --network $NETWORK \
      -e MINIO_ENDPOINT=minio:9000 \
      -e MINIO_ACCESS_KEY=minioadmin \
      -e MINIO_SECRET_KEY=minioadmin \
      -w /app --entrypoint sh paperless-batch -c '
pip install --quiet Pillow==11.0.0 > /dev/null 2>&1
python - <<PY
import io
from PIL import Image, ImageDraw
from minio import Minio
mc = Minio(\"minio:9000\", access_key=\"minioadmin\", secret_key=\"minioadmin\", secure=False)
lines = [
    \"Invoice #2041 — total \$1,240.50\",
    \"Meeting notes: Q2 OKRs review\",
    \"Received: 12 packages checked in\",
    \"Deploy ticket #8817 — platform team\",
    \"System error — 404 on /api/legacy\",
]
for i, text in enumerate(lines):
    img = Image.new(\"L\", (512, 64), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 22), text, fill=0)
    buf = io.BytesIO()
    img.save(buf, \"PNG\"); buf.seek(0)
    key = f\"ood/ood_{i:02d}.png\"
    mc.put_object(\"paperless-images\", key, data=buf, length=len(buf.getvalue()),
                  content_type=\"image/png\")
    print(f\"uploaded {key}\")
PY'" > /dev/null 2>&1
    OOD_COUNT=$(sg docker -c "docker compose exec -T minio mc ls local/paperless-images/ood/" 2>/dev/null \
        | grep -c '\.png' || true)
    ok "OOD samples synthesized + uploaded ($OOD_COUNT PNGs)"
fi

# ── 12. Fixture upload + rollback-ctrl audit seed + TOKEN ──
log "[12/12] Final seeds (fixture, rollback audit, TOKEN)"
CURRENT_STEP="final seeds"

step "exporting TOKEN"
TOKEN=$(bash "$PAPERLESS_ML/scripts/get_paperless_token.sh")
if [[ -z "$TOKEN" ]]; then
    fail "could not extract Paperless admin token"
fi
echo "$TOKEN" > /tmp/paperless_token
chmod 600 /tmp/paperless_token
ok "TOKEN saved to /tmp/paperless_token (source before recording)"

step "pre-uploading handwritten fixture"
# Prefer the IAM-extracted fixture from step 11a; fall back to the legacy
# data_generator path for backward-compat if Elnath re-adds it upstream.
if [[ -f "$FIXTURE_PATH" && -s "$FIXTURE_PATH" ]]; then
    USE_FIXTURE="$FIXTURE_PATH"
elif [[ -f "$DATA_REPO/data_generator/fixtures/handwritten_sample.png" ]]; then
    USE_FIXTURE="$DATA_REPO/data_generator/fixtures/handwritten_sample.png"
else
    USE_FIXTURE=""
fi
if [[ -n "$USE_FIXTURE" ]]; then
    curl -sf -H "Authorization: Token $TOKEN" \
      -F "document=@$USE_FIXTURE" \
      -F "title=demo-handwritten-fresh" \
      "http://localhost:8000/api/documents/post_document/" > /dev/null && \
      ok "fixture uploaded from $USE_FIXTURE (wait ~45s for async ingestion)"
else
    warn "no handwritten fixture available — skipping pre-upload; bullet 2 will need live upload"
fi

step "seeding rollback-ctrl audit log with synthetic HtrInputDrift"
sg docker -c "docker compose exec rollback-ctrl sh -c '
python - <<PY
import json, urllib.request
payload = {\"alerts\":[{\"status\":\"firing\",\"labels\":{
    \"alertname\":\"HtrInputDrift\",
    \"rollback_trigger\":\"true\",
    \"severity\":\"critical\"
}}]}
req = urllib.request.Request(\"http://localhost:8000/webhook\",
    data=json.dumps(payload).encode(),
    headers={\"Content-Type\":\"application/json\"})
urllib.request.urlopen(req).read()
PY'" > /dev/null

step "resetting @production to v2 (rollback demo needs two versions below)"
sg docker -c "docker compose exec -T mlflow python - <<'PY'
from mlflow import MlflowClient
c = MlflowClient('http://localhost:5000')
c.set_registered_model_alias('paperless-htr', 'production', '2')
print('@production reset to v2')
PY
"
ok "audit log seeded, @production back at v2"

# ── Final sanity ────────────────────────────────
log "Final sanity checks"

echo
echo "───── Prometheus targets ─────"
curl -s http://localhost:9090/api/v1/targets | python3 -c "
import json, sys
for t in sorted(json.load(sys.stdin)['data']['activeTargets'], key=lambda x: x['labels'].get('job','')):
    health = t['health']
    marker = '  ✓' if health == 'up' else '  ✗'
    print(f'{marker} {t[\"labels\"].get(\"job\"):22s} {health}')"

echo
echo "───── MLflow state ─────"
sg docker -c "docker compose exec -T mlflow python - <<'PY'
from mlflow import MlflowClient
c = MlflowClient('http://localhost:5000')
versions = c.search_model_versions(\"name='paperless-htr'\")
mv = c.get_model_version_by_alias('paperless-htr', 'production')
print(f'  versions: {len(versions)}')
print(f'  @production: v{mv.version}')
PY
"

echo
echo "───── drift_monitor metrics ─────"
curl -s http://localhost:8100/metrics 2>/dev/null | grep -E "^drift_(checks|events)_total" | sed 's/^/  /' || warn "drift_monitor not reachable"

echo
echo "───── OOD samples (must be present before demo) ─────"
OOD=$(sg docker -c "docker compose exec -T minio mc ls local/paperless-images/ood/" 2>/dev/null | wc -l)
if (( OOD >= 3 )); then
    ok "OOD samples staged: $OOD objects"
else
    warn "OOD samples missing ($OOD present, need ≥3) — ping Elnath to upload"
fi

echo
echo "═══════════════════════════════════════════════════════════════════"
printf "%s%s  Demo warm-up complete%s\n" "$C_GREEN" "$C_BOLD" "$C_RESET"
echo "═══════════════════════════════════════════════════════════════════"
echo
echo "Next:"
echo "  source /tmp/paperless_token && export TOKEN=\$(cat /tmp/paperless_token)"
echo "    # or put the export in every shell you'll demo from"
echo
echo "  If you're running T-30m polish, follow docs/demo_commands.md §'Pre-recording sanity test'"
echo
echo "  Watch dashboards tab: http://<floating-ip>:3000 (admin/admin)"
echo "  Watch alerts tab:     http://<floating-ip>:9090/alerts"
