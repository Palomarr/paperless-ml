#!/usr/bin/env bash
# ===========================================================================
# chameleon_setup.sh — one-command bring-up of the Paperless-ML stack on a
# fresh Chameleon CHI@UC bare-metal node.
#
# Assumes gpu_p100 or gpu_rtx_6000 node with NVIDIA drivers pre-installed
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
#   --skip-verify   don't run verify_integration.sh at the end
#   --skip-peers    don't clone REDES01/paperless_data and REDES01/paperless_data_integration
#                   (peer repos needed for full Path A integration; see DEPLOYMENT.md)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"

SKIP_VERIFY=0
SKIP_PEERS=0
for arg in "$@"; do
    case "$arg" in
        --skip-verify) SKIP_VERIFY=1 ;;
        --skip-peers)  SKIP_PEERS=1 ;;
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
log "Step 1/6: Installing system packages and Docker Engine"

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
log "Step 2/6: Installing NVIDIA Container Toolkit"

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
# Step 3 — Clone peer repos (Elnath's data stack + consumer) as siblings
# ===========================================================================
if (( SKIP_PEERS )); then
    log "Step 3/6: Skipping peer-repo clone (--skip-peers)"
else
    log "Step 3/6: Cloning peer repos into ${PROJECT_PARENT}/"

    clone_if_missing() {
        local url="$1" dst="$2"
        if [[ -d "$dst/.git" ]]; then
            ok "$(basename "$dst") already cloned"
        else
            git clone --depth=50 "$url" "$dst"
            ok "Cloned $(basename "$dst")"
        fi
    }

    clone_if_missing "https://github.com/REDES01/paperless_data.git" \
                     "${PROJECT_PARENT}/paperless_data"
    clone_if_missing "https://github.com/REDES01/paperless_data_integration.git" \
                     "${PROJECT_PARENT}/paperless_data_integration"

    ok "Peer repos on plain main (D1 + D3 merged upstream). Path A works end-to-end."
fi

# ===========================================================================
# Step 4 — Create shared bridge network (Path A)
# ===========================================================================
log "Step 4/6: Creating paperless_ml_net shared bridge"
sg docker -c "bash ${REPO_ROOT}/scripts/create_network.sh"
ok "paperless_ml_net ready"

# ===========================================================================
# Step 5 — docker compose up (our stack with shared-network overlay)
# ===========================================================================
log "Step 5/6: Bringing up our stack (13 services + Path A overlay)"

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
# Step 6 — Run verify_integration.sh
# ===========================================================================
if (( SKIP_VERIFY )); then
    log "Step 6/6: Skipping verify_integration.sh (--skip-verify)"
else
    log "Step 6/6: Running verify_integration.sh (13 checkpoints)"
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
echo
echo "Path A (optional end-to-end HTR round-trip with Elnath's stacks):"
echo "  See docs/DEPLOYMENT.md §5 for bring-up of paperless_data + paperless_data_integration."
echo
echo "Teardown:"
echo "  docker compose -f docker-compose.yml -f docker-compose.shared.yml down"
