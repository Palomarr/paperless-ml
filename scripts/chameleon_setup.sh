#!/usr/bin/env bash
# ===========================================================================
# chameleon_setup.sh — one-command bring-up of the Paperless-ML stack on a
# fresh Chameleon CHI@UC bare-metal node.
#
# Assumes gpu_p100 node with NVIDIA drivers pre-installed
# by the CC-Ubuntu24.04-CUDA image. Running on any other hardware will fail
# loudly at the nvidia-container-toolkit install step — that's intentional.
#
# Prerequisites:
#   - Node reserved via paperless_data/provision_chameleon.ipynb (or equivalent)
#   - SSH as cc@<floating-ip>
#   - This repo cloned to ~/paperless-ml (or wherever; script uses its own
#     location as the anchor)
#
# Usage:
#   git clone https://github.com/Palomarr/paperless-ml.git ~/paperless-ml
#   cd ~/paperless-ml
#   bash scripts/chameleon_setup.sh
#
# Flags:
#   --skip-verify              don't run verify_integration.sh at the end
#   --skip-peers               don't clone REDES01/paperless_data and REDES01/paperless_data_integration
#                              (peer repos needed for full Path A integration; see DEPLOYMENT.md)
#   --skip-peer-components     don't bring up Elnath's peer components (htr_consumer,
#                              drift_monitor, behavior_emulator, airflow + paperless-training).
#                              Implied by --skip-peers since those services need the peer repos.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"

SKIP_VERIFY=0
SKIP_PEERS=0
SKIP_PEER_COMPONENTS=0
for arg in "$@"; do
    case "$arg" in
        --skip-verify)         SKIP_VERIFY=1 ;;
        --skip-peers)          SKIP_PEERS=1; SKIP_PEER_COMPONENTS=1 ;;
        --skip-peer-components) SKIP_PEER_COMPONENTS=1 ;;
        -h|--help)
            grep '^#' "$0" | head -30
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

# ===========================================================================
# Step 1 — system packages + Docker Engine
# ===========================================================================
log "Step 1/7: Installing system packages and Docker Engine"

if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq curl ca-certificates git make jq
    curl -fsSL https://get.docker.com | sudo sh
    ok "Docker Engine installed"
else
    ok "Docker already present ($(docker --version))"
fi

# Add current user to docker group so we can run without sudo. `sg docker -c`
# below activates the new group membership in this shell without needing a
# fresh login.
sudo groupadd -f docker
sudo usermod -aG docker "$USER"

# ===========================================================================
# Step 2 — NVIDIA Container Toolkit (GPU support for Docker)
# ===========================================================================
log "Step 2/7: Installing NVIDIA Container Toolkit"

if ! dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q '^ii'; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ok "NVIDIA Container Toolkit installed and Docker restarted"
else
    ok "NVIDIA Container Toolkit already present"
fi

# Smoke-test GPU visibility inside a container.
if sg docker -c "docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi" \
        >/tmp/nvidia-smi.log 2>&1; then
    ok "Docker can see the GPU ($(grep -m1 'GPU 0' /tmp/nvidia-smi.log | tr -s ' ' || echo 'device present'))"
else
    warn "Docker GPU smoke-test failed — check /tmp/nvidia-smi.log. Continuing; ml-gateway will fall back to CPU."
fi

# ===========================================================================
# Step 3 — Clone peer repos as siblings
# ===========================================================================
if (( SKIP_PEERS )); then
    log "Step 3/7: Skipping peer-repo clone (--skip-peers)"
else
    log "Step 3/7: Cloning peer repos into ${PROJECT_PARENT}/"

    clone_if_missing() {
        local url="$1" dst="$2"
        if [[ -d "$dst/.git" ]]; then
            # Pull latest main so re-runs pick up teammates' pushes. Uses
            # --ff-only to refuse if the local clone has diverged (dirty
            # working tree, detached HEAD, etc.) — in that case we warn
            # and continue with the existing clone rather than kill the
            # whole setup.
            if (cd "$dst" && git pull --ff-only) >/dev/null 2>&1; then
                ok "$(basename "$dst") updated to latest main"
            else
                warn "$(basename "$dst") pull failed (local changes or network?); using existing clone"
            fi
        else
            git clone --depth=50 "$url" "$dst"
            ok "Cloned $(basename "$dst")"
        fi
    }

    clone_if_missing "https://github.com/REDES01/paperless_data.git" \
                     "${PROJECT_PARENT}/paperless_data"
    clone_if_missing "https://github.com/REDES01/paperless_data_integration.git" \
                     "${PROJECT_PARENT}/paperless_data_integration"
    clone_if_missing "https://github.com/gdtmax/paperless_training_integration.git" \
                     "${PROJECT_PARENT}/paperless_training_integration"

    ok "Peer repos cloned: paperless_data + paperless_data_integration (Elnath), paperless_training_integration (Dongting)"
fi

# ===========================================================================
# Step 4 — docker compose up (our stack with shared-network overlay).
# The paperless_ml_net bridge is declared in docker-compose.yml and
# self-creates on first boot — no explicit network-create step needed.
# ===========================================================================
log "Step 4/7: Bringing up our stack (services + Path A overlay)"

cd "$REPO_ROOT"
sg docker -c "docker compose -f docker-compose.yml -f docker-compose.shared.yml up -d"

# Wait for ml-gateway (longest cold-start — TrOCR + mpnet first-time load).
log "    Waiting for ml-gateway /health (up to 240s; model download on first boot)"
ATTEMPTS=0
until sg docker -c "docker compose exec -T ml-gateway curl -fsS http://localhost:8000/health" \
        >/dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if (( ATTEMPTS > 40 )); then
        warn "ml-gateway did not become healthy in 240s. Check:"
        warn "  docker compose logs --tail=100 ml-gateway"
        break
    fi
    sleep 6
done
(( ATTEMPTS <= 40 )) && ok "ml-gateway healthy"

# ===========================================================================
# Step 6 — Extract Paperless API token for the admin user.
# Needed by Elnath's data_generator (token auth). Idempotent. 
# See scripts/get_paperless_token.sh.
# ===========================================================================
log "Step 5/7: Extracting Paperless API token for admin"
PAPERLESS_TOKEN="$(sg docker -c "bash ${REPO_ROOT}/scripts/get_paperless_token.sh" 2>/dev/null | tr -d '\r' | tail -n 1)"
if [[ -z "$PAPERLESS_TOKEN" || "${#PAPERLESS_TOKEN}" -lt 20 ]]; then
    warn "Token extraction returned unexpected output. Retry manually with:"
    warn "  bash scripts/get_paperless_token.sh"
    PAPERLESS_TOKEN=""
else
    ok "Paperless API token extracted (${#PAPERLESS_TOKEN}-char token)"
fi

# ===========================================================================
# Step 6 — Bring up peer components (Elnath's data-pipeline + retraining stack)
# Folds htr_consumer, drift_monitor, behavior_emulator, airflow, and the
# paperless-training image into the same bootstrap so the integrated system
# comes up in one command (per system-impl rubric: "no SSH-based intervention
# beyond a single bootstrap"). Skipped if --skip-peer-components or peer repos
# weren't cloned.
# ===========================================================================
if (( SKIP_PEER_COMPONENTS )); then
    log "Step 6/7: Skipping peer components (--skip-peer-components)"
elif (( SKIP_PEERS )); then
    log "Step 6/7: Skipping peer components (peer repos not cloned)"
else
    log "Step 6/7: Bringing up peer components (htr_consumer, drift_monitor, behavior_emulator, airflow)"
    sg docker -c "bash ${REPO_ROOT}/scripts/up_peer_components.sh tier1"
    # tier2 needs PAPERLESS_TOKEN to wire htr_consumer's auth header.
    if [[ -n "$PAPERLESS_TOKEN" ]]; then
        sg docker -c "PAPERLESS_TOKEN='${PAPERLESS_TOKEN}' bash ${REPO_ROOT}/scripts/up_peer_components.sh tier2"
        ok "Peer components running (tier1 + tier2)"
    else
        warn "Skipping tier2 peer-component bring-up — no PAPERLESS_TOKEN."
        warn "Run manually after token extraction:"
        warn "  PAPERLESS_TOKEN=\$(bash scripts/get_paperless_token.sh) bash scripts/up_peer_components.sh tier2"
    fi
fi

# ===========================================================================
# Step 7 — Run verify_integration.sh
# ===========================================================================
if (( SKIP_VERIFY )); then
    log "Step 7/7: Skipping verify_integration.sh (--skip-verify)"
else
    log "Step 7/7: Running verify_integration.sh (17 checkpoints)"
    sg docker -c "bash ${REPO_ROOT}/scripts/verify_integration.sh"
fi

# ===========================================================================
# Summary + access URLs
# ===========================================================================
FLOATING_IP="$(curl -fsS4 https://checkip.amazonaws.com 2>/dev/null || echo '<floating-ip>')"

echo
ok "Bring-up complete."
echo
echo "Services (replace <floating-ip> with your actual Chameleon IP if checkip is blocked):"
echo "  Paperless web + feedback UI : http://${FLOATING_IP}:8000   (admin / admin)"
echo "  Feedback UI                 : http://${FLOATING_IP}:8000/ml-ui/"
echo "  Grafana                     : http://${FLOATING_IP}:3000   (admin / admin)"
echo "  Prometheus alerts           : http://${FLOATING_IP}:9090/alerts"
echo "  Alertmanager                : http://${FLOATING_IP}:9093"
echo "  Qdrant dashboard            : http://${FLOATING_IP}:6333/dashboard"
echo "  MinIO console               : http://${FLOATING_IP}:9001   (minioadmin / minioadmin)"
echo "  MLflow UI                   : http://${FLOATING_IP}:5050"
if (( ! SKIP_PEER_COMPONENTS )) && (( ! SKIP_PEERS )); then
    echo "  Airflow (peer retraining)   : http://${FLOATING_IP}:8080   (airflow / airflow)"
fi
echo
if [[ -n "$PAPERLESS_TOKEN" ]]; then
    echo "Paperless API token (for data_generator, demos, curl scripts):"
    echo "  ${PAPERLESS_TOKEN}"
    echo "  (re-extract anytime with: bash scripts/get_paperless_token.sh)"
    echo
    if (( ! SKIP_PEERS )); then
        echo "Run production-traffic generator:"
        echo "  bash scripts/run_data_generator.sh --rate 2.0 --duration 300"
        echo "  (wraps token extraction + image build + docker run)"
        echo
    fi
fi
echo "Next steps:"
echo "  bash scripts/seed_demo.sh --trigger-alert    # populate counters + fire/resolve alert"
echo "  bash scripts/run_data_generator.sh           # real synthetic traffic via Elnath's generator"
echo "  bash scripts/verify_integration.sh           # 13-checkpoint integration test"
echo "  bash scripts/deploy_model.sh                 # MLflow → MinIO → ml-gateway reload"
echo
echo "Teardown:"
echo "  docker compose -f docker-compose.yml -f docker-compose.shared.yml down"
