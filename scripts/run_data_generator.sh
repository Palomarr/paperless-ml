#!/usr/bin/env bash
# ===========================================================================
# run_data_generator.sh — one-command wrapper around data_generator
# (REDES01/paperless_data/data_generator).
#
# Handles token extraction, image build (if missing), and the docker run
# invocation with the right network + hostname. Pass CLI args straight
# through to generator.py.
#
# Usage:
#   bash scripts/run_data_generator.sh                      # default: --rate 2.0 --duration 300
#   bash scripts/run_data_generator.sh --rate 1.0 --duration 60
#   bash scripts/run_data_generator.sh --help
#
# Prerequisites:
#   - docker compose stack up (chameleon_setup.sh has been run)
#   - paperless_data cloned as a sibling directory (handled by setup script)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"
GEN_DIR="${PROJECT_PARENT}/paperless_data/data_generator"

# ---------- output helpers ----------
C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
log()  { printf "%s==>%s %s\n" "$C_BOLD"   "$C_RESET" "$*"; }
ok()   { printf "%s ✓ %s %s\n" "$C_GREEN"  "$C_RESET" "$*"; }
fail() { printf "%s ✗ %s %s\n" "$C_RED"    "$C_RESET" "$*"; exit 1; }

# ---------- preflight ----------
[[ -d "$GEN_DIR" ]] || fail "$GEN_DIR not found — run scripts/chameleon_setup.sh first to clone peer repos"

# Ensure the shared network exists (generator attaches to it).
docker network inspect paperless_ml_net >/dev/null 2>&1 \
    || fail "paperless_ml_net not found — run scripts/create_network.sh or scripts/chameleon_setup.sh"

# ---------- step 1: build image if missing ----------
if docker image inspect paperless-data-generator >/dev/null 2>&1; then
    ok "paperless-data-generator image present"
else
    log "Building paperless-data-generator image from ${GEN_DIR}"
    (cd "$GEN_DIR" && docker build -q -t paperless-data-generator .)
    ok "Image built"
fi

# ---------- step 2: extract Paperless API token ----------
log "Extracting Paperless API token"
TOKEN="$(bash "${SCRIPT_DIR}/get_paperless_token.sh")"
if [[ -z "$TOKEN" || "${#TOKEN}" -lt 20 ]]; then
    fail "Token extraction returned unexpected output (got '${TOKEN}')"
fi
ok "Token extracted (${#TOKEN} chars)"

# ---------- step 3: run the generator, passing through any extra CLI args ----------
log "Running generator (args: ${*:-'<defaults>'})"

# Defaults if caller passed nothing — readable on-camera rate.
if [[ $# -eq 0 ]]; then
    set -- --rate 2.0 --duration 300
fi

docker run --rm --network paperless_ml_net \
    -e PAPERLESS_TOKEN="$TOKEN" \
    paperless-data-generator \
    --paperless-url http://paperless-webserver-1:8000 \
    "$@"
