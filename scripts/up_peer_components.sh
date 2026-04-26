#!/usr/bin/env bash
# ============================================================================
# up_peer_components.sh — bring up Elnath's complementary peer components
# alongside our paperless_ml stack, without duplicating shared services.
#
# Why: our verify_integration.sh stack (13 services) is the canonical
# integration. Elnath's repo bundles 8 duplicates for solo-dev plus 6
# unique components (htr_consumer, region_slicer, drift_monitor,
# behavior_emulator, airflow, paperless-training image) we genuinely
# need for the ongoing-operation rubric (April 28 - May 4).
#
# This script brings up the unique components only, with the right
# env var overrides for service names + creds + route names.
#
# Usage:
#   bash scripts/up_peer_components.sh tier1
#       Builds htr_trainer + htr_batch images, brings up airflow + behavior_emulator.
#       Smoke-tests the retraining-loop preconditions without exotic deps.
#
#   bash scripts/up_peer_components.sh tier2
#       Adds htr_consumer + drift_monitor + region_slicer (requires data-stack
#       postgres + Phase 2 schema + drift reference baked).
#
#   bash scripts/up_peer_components.sh down
#       Tears down the peer components (leaves our paperless_ml stack alone).
#
# Pre-flight assumptions:
#   - paperless_ml stack is up (`docker compose ps` shows 13 healthy services)
#   - Peer repos cloned at ~/paperless_data_integration and ~/paperless_data
#   - paperless_ml_net network exists
#   - PAPERLESS_TOKEN exported (only required for tier2)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PEER_INT_DIR="${PEER_INT_DIR:-${HOME}/paperless_data_integration}"
PEER_DATA_DIR="${PEER_DATA_DIR:-${HOME}/paperless_data}"

cd "${ROOT_DIR}"

# ---- output helpers ----
if [[ -t 1 ]]; then
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_GREEN=""; C_RED=""; C_YELLOW=""; C_BOLD=""; C_RESET=""
fi
info() { printf "%s==>%s %s\n" "$C_BOLD" "$C_RESET" "$*"; }
ok()   { printf " %sOK%s   %s\n" "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf " %sWARN%s %s\n" "$C_YELLOW" "$C_RESET" "$*"; }
fail() { printf " %sFAIL%s %s\n" "$C_RED" "$C_RESET" "$*"; exit 1; }

# ---- preflight ----
preflight() {
    info "Pre-flight checks"
    [[ -d "$PEER_INT_DIR" ]] || fail "peer repo not found: $PEER_INT_DIR (set PEER_INT_DIR if non-default)"
    [[ -d "$PEER_DATA_DIR" ]] || fail "peer repo not found: $PEER_DATA_DIR"
    docker network inspect paperless_ml_net >/dev/null 2>&1 || fail "paperless_ml_net not present — bring up paperless_ml first"
    docker compose ps -q ml-gateway >/dev/null 2>&1 || fail "ml-gateway not running — bring up paperless_ml first"

    # Confirm ml_gateway alias is live (added in docker-compose.shared.yml)
    if ! docker network inspect paperless_ml_net --format '{{json .Containers}}' \
            | grep -q "ml_gateway\|fastapi_server"; then
        warn "ml_gateway alias may not be live; recreate ml-gateway with shared overlay:"
        warn "  docker compose -f docker-compose.yml -f docker-compose.shared.yml up -d --force-recreate ml-gateway"
    fi
    ok "preflight passed"
}

# ============================================================================
# TIER 1: training images + airflow + behavior_emulator
# ============================================================================

build_training_images() {
    info "Building htr_trainer image (paperless-training)"
    cd "$PEER_INT_DIR"
    docker compose -p training -f training/compose.yml build
    ok "htr_trainer:latest built"

    info "Building htr_batch image (used by airflow build_snapshot)"
    docker build -t htr_batch:latest "$PEER_DATA_DIR/batch_pipeline"
    ok "htr_batch:latest built"
    cd "$ROOT_DIR"
}

up_airflow() {
    info "Bringing up Airflow (webserver + scheduler + airflow-postgres)"
    cd "$PEER_INT_DIR"
    docker compose -p airflow -f airflow/compose.yml up -d --build
    cd "$ROOT_DIR"
    info "Waiting up to 180s for Airflow webserver health..."
    local elapsed=0
    while (( elapsed < 180 )); do
        if curl -fsS "http://localhost:8080/health" >/dev/null 2>&1; then
            ok "Airflow webserver healthy at http://localhost:8080 (admin/admin)"
            return 0
        fi
        sleep 5
        elapsed=$(( elapsed + 5 ))
    done
    warn "Airflow webserver didn't respond at :8080/health within 180s"
    warn "Check: docker compose -p airflow -f $PEER_INT_DIR/airflow/compose.yml logs --tail=50"
}

up_behavior_emulator() {
    info "Bringing up behavior_emulator (BE_MODE=${BE_MODE:-normal})"
    info "  Override: attaching to BOTH paperless_ml_net + paperless_data_default"
    info "            (peer compose only joins paperless_ml_net; postgres is on the other)"
    # Override compose: dual-network attach so behavior_emulator can reach
    # both ml-gateway (on paperless_ml_net) and data-stack postgres (on
    # paperless_data_default). Required because Elnath's compose only
    # attaches to one network.
    local override="/tmp/behavior_emulator.override.yml"
    cat > "$override" <<'EOF'
services:
  behavior_emulator:
    networks:
      - paperless_ml_net
      - paperless_data_default
networks:
  paperless_data_default:
    external: true
    name: paperless_data_default
EOF
    cd "$PEER_INT_DIR"
    BE_MODE="${BE_MODE:-normal}" docker compose -p behavior_emulator \
        -f behavior_emulator/compose.yml \
        -f "$override" \
        up -d --build --force-recreate
    cd "$ROOT_DIR"
    sleep 5
    if docker ps --filter "name=behavior_emulator" --filter "status=running" -q | grep -q .; then
        ok "behavior_emulator running (logs: docker logs -f behavior_emulator)"
    else
        warn "behavior_emulator not running — check logs"
    fi
}

# ============================================================================
# TIER 2: htr_consumer + drift_monitor + data-stack postgres
# ============================================================================

up_data_stack_postgres() {
    info "Bringing up data-stack postgres only (skip his minio/redpanda/qdrant duplicates)"
    cd "$PEER_DATA_DIR"
    docker compose -p paperless_data -f docker/docker-compose.yaml up -d postgres
    cd "$ROOT_DIR"
    info "Waiting for data-stack postgres to accept connections..."
    local elapsed=0
    while (( elapsed < 60 )); do
        if docker exec postgres pg_isready -U user -d paperless >/dev/null 2>&1; then
            ok "data-stack postgres ready (named: postgres, network: paperless_data_default)"
            return 0
        fi
        sleep 3
        elapsed=$(( elapsed + 3 ))
    done
    warn "data-stack postgres didn't become ready in 60s"
}

apply_phase2_migration() {
    info "Applying Phase 2 schema migration (paperless_doc_id bridge column)"
    if [[ -f "$PEER_INT_DIR/seed/phase2_add_paperless_doc_id.sql" ]]; then
        cat "$PEER_INT_DIR/seed/phase2_add_paperless_doc_id.sql" \
            | docker exec -i postgres psql -U user -d paperless
        ok "Phase 2 migration applied"
    else
        warn "Phase 2 migration SQL not found at expected path; skipping (may need manual apply)"
    fi
}

up_htr_consumer() {
    if [[ -z "${PAPERLESS_TOKEN:-}" ]]; then
        warn "PAPERLESS_TOKEN not set; htr_consumer will exit. Run:"
        warn "  TOKEN=\$(bash $ROOT_DIR/scripts/get_paperless_token.sh)"
        warn "  PAPERLESS_TOKEN=\$TOKEN bash $0 tier2"
        return 1
    fi
    info "Bringing up htr_consumer (with HTR_ENDPOINT + MinIO + Kafka tuning overrides)"
    # Override compose: addresses three peer-compose issues:
    #  1. HTR_ENDPOINT hardcoded to /predict/htr — override to /htr
    #  2. MinIO creds hardcoded admin/paperless_minio — override to minioadmin/minioadmin
    #  3. KafkaConsumer uses kafka-python defaults (max_poll_interval=300s,
    #     max_poll_records=500). Long documents (e.g. 241-page books) take
    #     longer than 5 min to process, exceed the poll interval, get the
    #     consumer kicked from group, infinite-loop on restart. The patched
    #     consumer.py at scripts/peer_patches/htr_consumer.py adds env-driven
    #     overrides; we bind-mount it over the upstream file.
    local override="/tmp/htr_consumer.override.yml"
    cat > "$override" <<EOF
services:
  htr_consumer:
    environment:
      HTR_ENDPOINT: /htr
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      KAFKA_MAX_POLL_INTERVAL_MS: "1800000"
      KAFKA_MAX_POLL_RECORDS: "1"
    volumes:
      - ${ROOT_DIR}/scripts/peer_patches/htr_consumer.py:/app/consumer.py:ro
EOF
    cd "$PEER_INT_DIR"
    docker compose -p htr_consumer \
        -f htr_consumer/compose.yml \
        -f "$override" \
        up -d --build --force-recreate
    cd "$ROOT_DIR"
    sleep 5
    ok "htr_consumer running with kafka tuning patch (logs: docker logs -f htr_consumer)"
}

up_drift_monitor() {
    info "Bringing up drift_monitor (with MinIO cred override)"
    info "  Note: requires drift reference baked at warehouse/drift_reference/htr_v1/cd"
    info "        If reference missing, service will crash-loop until built."
    local override="/tmp/drift_monitor.override.yml"
    cat > "$override" <<'EOF'
services:
  drift_monitor:
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
EOF
    cd "$PEER_INT_DIR"
    docker compose -p drift_monitor \
        -f drift_monitor/compose.yml \
        -f "$override" \
        up -d --build --force-recreate
    cd "$ROOT_DIR"
    sleep 10
    if curl -fsS "http://localhost:8200/health" >/dev/null 2>&1; then
        ok "drift_monitor healthy at http://localhost:8200"
    else
        warn "drift_monitor /health not responding — likely missing reference (build via paperless_data/scripts/build_drift_reference.py)"
    fi
}

# ============================================================================
# Tear-down
# ============================================================================

down_all() {
    info "Tearing down peer components (paperless_ml stays up)"
    for proj_pair in \
        "drift_monitor:drift_monitor/compose.yml:$PEER_INT_DIR" \
        "htr_consumer:htr_consumer/compose.yml:$PEER_INT_DIR" \
        "behavior_emulator:behavior_emulator/compose.yml:$PEER_INT_DIR" \
        "airflow:airflow/compose.yml:$PEER_INT_DIR" \
        "paperless_data:docker/docker-compose.yaml:$PEER_DATA_DIR"; do
        IFS=: read -r proj cf dir <<< "$proj_pair"
        if (cd "$dir" && docker compose -p "$proj" -f "$cf" ps -q 2>/dev/null | grep -q .); then
            (cd "$dir" && docker compose -p "$proj" -f "$cf" down) || true
            ok "stopped $proj"
        fi
    done
}

# ============================================================================
# Main
# ============================================================================

case "${1:-}" in
    tier1)
        preflight
        build_training_images
        up_airflow
        up_behavior_emulator
        echo
        info "TIER 1 brought up. Verify with:"
        echo "    docker images | grep -E 'htr_trainer|htr_batch'"
        echo "    docker ps --filter 'name=airflow|behavior_emulator'"
        echo "    open http://<floating-ip>:8080  (admin/admin)"
        echo "    docker logs -f behavior_emulator"
        echo
        info "Tier 2 (htr_consumer + drift_monitor) requires:"
        echo "    PAPERLESS_TOKEN=\$(bash scripts/get_paperless_token.sh)"
        echo "    PAPERLESS_TOKEN=\$PAPERLESS_TOKEN bash $0 tier2"
        ;;
    tier2)
        preflight
        up_data_stack_postgres
        apply_phase2_migration
        up_htr_consumer
        up_drift_monitor
        echo
        info "TIER 2 brought up (with caveats):"
        echo "    drift_monitor may crash-loop if reference not yet baked"
        echo "    htr_consumer requires PAPERLESS_TOKEN to be valid"
        echo "    Verify with: docker ps --filter 'name=htr_consumer|drift_monitor|postgres'"
        ;;
    down)
        down_all
        ;;
    fix)
        # Re-apply overrides to existing peer components — useful when fixing
        # network attach + cred mismatches without re-doing the long image builds.
        preflight
        info "Re-applying overrides to existing peer components"
        up_behavior_emulator
        if [[ -n "${PAPERLESS_TOKEN:-}" ]]; then
            up_htr_consumer
        else
            warn "PAPERLESS_TOKEN not set — skipping htr_consumer recreate"
        fi
        up_drift_monitor
        echo
        info "Done. Verify with:"
        echo "    docker logs --tail=20 behavior_emulator"
        echo "    docker logs --tail=20 htr_consumer"
        echo "    docker logs --tail=20 drift_monitor"
        ;;
    *)
        echo "Usage: $0 {tier1|tier2|fix|down}"
        echo "  tier1   Training images + airflow + behavior_emulator (low-friction)"
        echo "  tier2   Data-stack postgres + htr_consumer + drift_monitor (needs prereqs)"
        echo "  fix     Re-apply override compose files (for iterating after a fix)"
        echo "  down    Stop all peer components (leaves paperless_ml alone)"
        exit 1
        ;;
esac
