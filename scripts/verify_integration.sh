#!/usr/bin/env bash
# Verify the paperless-ngx + ml_hooks overlay integration end-to-end.
# Runs 12 checkpoints against a locally-brought-up docker compose stack.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------- output helpers ----------
if [[ -t 1 ]]; then
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
    C_GREEN=""; C_RED=""; C_YELLOW=""; C_BOLD=""; C_RESET=""
fi
info()  { printf "%s==>%s %s\n" "$C_BOLD"   "$C_RESET" "$*"; }
pass()  { printf "%s PASS %s %s\n" "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf "%s WARN %s %s\n" "$C_YELLOW" "$C_RESET" "$*"; }
fail()  { printf "%s FAIL %s %s\n" "$C_RED"   "$C_RESET" "$*"; }

die()   { fail "$*"; exit 1; }

# ---------- args ----------
CLEAN=0
KEEP_UP=0
for arg in "$@"; do
    case "$arg" in
        --clean)    CLEAN=1 ;;
        --keep-up)  KEEP_UP=1 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--clean] [--keep-up]
  --clean    docker compose down -v before starting (wipe volumes)
  --keep-up  leave the stack running after verification (default: leave up)
EOF
            exit 0 ;;
        *) die "Unknown arg: $arg" ;;
    esac
done

# ---------- preflight ----------
command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null 2>&1 || die "docker compose plugin not found"

[[ -d paperless_patches/ml_hooks ]] || die "paperless_patches/ml_hooks not found — run from repo root"
[[ -f docker-compose.yml ]]         || die "docker-compose.yml not found at repo root"

# ---------- step 0: clean if asked, bring up stack ----------
if (( CLEAN )); then
    info "Cleaning previous stack (volumes included)"
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
fi

info "Generating ml_hooks migrations if missing"
if [[ ! -f paperless_patches/ml_hooks/migrations/0001_initial.py ]]; then
    # Bring up postgres first so makemigrations (which checks nothing but runs Django setup) works
    docker compose up -d postgres redis fastapi-stub >/dev/null
    # Wait briefly for DB
    for _ in {1..20}; do
        docker compose exec -T postgres pg_isready -U paperless >/dev/null 2>&1 && break
        sleep 1
    done
    # Use `run` so we don't depend on web being up; the paperless-web image has Django installed.
    docker compose run --rm --no-deps \
        -e PAPERLESS_DBHOST=postgres \
        paperless-web python manage.py makemigrations ml_hooks --noinput \
        || warn "makemigrations returned non-zero (may be fine if already present)"
fi

# Detect whether the shared network overlay is already in use. If it is,
# keep it across the up -d call so we don't detach our services from
# paperless_ml_net while verifying.
COMPOSE_FILES=(-f docker-compose.yml)
if [[ -f docker-compose.shared.yml ]] \
    && docker network inspect paperless_ml_net >/dev/null 2>&1; then
    COMPOSE_FILES+=(-f docker-compose.shared.yml)
    info "Detected paperless_ml_net — verifying in shared-overlay mode"
fi

info "Bringing up full stack"
docker compose "${COMPOSE_FILES[@]}" up -d >/dev/null

START_TS="$(date +%s)"

# ---------- checkpoint utilities ----------
wait_for_log() {
    local service="$1" pattern="$2" timeout="${3:-60}"
    local elapsed=0
    while (( elapsed < timeout )); do
        if docker compose logs --since="${START_TS}" "$service" 2>/dev/null | grep -q -- "$pattern"; then
            return 0
        fi
        sleep 2
        elapsed=$(( elapsed + 2 ))
    done
    return 1
}

# ---------- checkpoint 1: URL routes mounted ----------
info "Checkpoint 1: /api/ml/ URL namespace reachable"
attempt=0
probe_code=0
while (( attempt < 60 )); do
    probe_code="$(curl -s -o /dev/null -w '%{http_code}' -u admin:admin http://localhost:8000/api/ml/feedback/ || echo 000)"
    if [[ "$probe_code" == "200" || "$probe_code" == "500" ]]; then
        # 200 = mounted + table exists; 500 = mounted but table missing (still proves the route)
        break
    fi
    sleep 2
    attempt=$(( attempt + 1 ))
done
if [[ "$probe_code" == "200" ]]; then
    pass "GET /api/ml/feedback/ -> 200 (URL mount + DB both working)"
elif [[ "$probe_code" == "500" ]]; then
    pass "URL mounted (got 500 — likely missing migration; checkpoint 2 will catch)"
else
    fail "GET /api/ml/feedback/ returned HTTP $probe_code after 120s — URL not mounted"
    docker compose logs --tail=80 paperless-web || true
    exit 1
fi

# ---------- checkpoint 2: migration applied ----------
info "Checkpoint 2: ml_hooks migration applied"
if wait_for_log paperless-web "Applying ml_hooks\." 120 \
   || docker compose exec -T paperless-web python manage.py showmigrations ml_hooks 2>/dev/null | grep -q "\[X\]"; then
    pass "ml_hooks.0001_initial is applied"
else
    fail "ml_hooks migration not applied — Feedback table likely missing"
    exit 1
fi

# ---------- checkpoint 3: feedback endpoint reachable ----------
info "Checkpoint 3: /api/ml/feedback/ responds 200"
# Give the paperless-web a moment to finish Django startup if migration just applied
sleep 3
http_code="$(curl -s -o /tmp/ml_feedback_body -w '%{http_code}' -u admin:admin http://localhost:8000/api/ml/feedback/ || echo 000)"
if [[ "$http_code" == "200" ]]; then
    pass "GET /api/ml/feedback/ → 200; body: $(head -c 200 /tmp/ml_feedback_body)"
else
    fail "GET /api/ml/feedback/ → HTTP $http_code"
    cat /tmp/ml_feedback_body 2>/dev/null || true
    exit 1
fi

# ---------- checkpoint 4: upload a document ----------
info "Checkpoint 4: upload a test document"
TMP_DOC="$(mktemp --suffix=.txt)"
printf "Integration test document.\nUploaded at %s.\n" "$(date -Iseconds)" > "$TMP_DOC"
upload_resp="$(curl -s -u admin:admin -X POST \
    -F "document=@${TMP_DOC}" \
    -F "title=ml_hooks integration test" \
    http://localhost:8000/api/documents/post_document/ || true)"
rm -f "$TMP_DOC"
if [[ -n "$upload_resp" ]]; then
    pass "Upload accepted; server response: $upload_resp"
else
    fail "Upload endpoint returned empty response"
    exit 1
fi

# ---------- checkpoint 5: signal handler fired ----------
info "Checkpoint 5: document_consumption_finished signal fired"
if wait_for_log paperless-web "ml_hooks: consumption finished for doc" 180; then
    pass "on_consumption_finished handler executed"
else
    fail "Signal handler never fired — consume pipeline may have stalled"
    docker compose logs --tail=120 paperless-web | tail -60 || true
    exit 1
fi

# ---------- checkpoint 6: upload event published + encode task ran ----------
info "Checkpoint 6: paperless.uploads event published + encode_document ran"
ok_upload=0; ok_embed=0
if wait_for_log paperless-web "published event to paperless.uploads" 120; then ok_upload=1; fi
if wait_for_log paperless-web "encode_document: doc" 120; then ok_embed=1; fi
if (( ok_upload && ok_embed )); then
    pass "Redpanda publish + Qdrant encode both ran"
else
    (( ok_upload )) || fail "paperless.uploads event was not published"
    (( ok_embed ))  || fail "encode_document never completed"
    docker compose logs --tail=150 paperless-web | tail -80 || true
    exit 1
fi

# ---------- checkpoint 7: vectors landed in Qdrant ----------
info "Checkpoint 7: document vectors upserted to Qdrant"
attempt=0
count=0
while (( attempt < 30 )); do
    count="$(curl -s -X POST http://localhost:6333/collections/document_chunks/points/count \
        -H 'Content-Type: application/json' \
        -d '{"exact":true}' 2>/dev/null \
        | python3 -c 'import sys,json;print(json.load(sys.stdin).get("result",{}).get("count",0))' 2>/dev/null || echo 0)"
    if [[ "$count" -gt 0 ]]; then
        break
    fi
    sleep 2
    attempt=$(( attempt + 1 ))
done
if [[ "$count" -gt 0 ]]; then
    pass "Qdrant document_chunks contains $count vector(s)"
else
    fail "No vectors in Qdrant after encode_document — upsert path may be broken"
    docker compose logs --tail=60 ml-gateway || true
    exit 1
fi

# ---------- checkpoint 8: modified search view returns merged results ----------
info "Checkpoint 8: /api/search/ returns documents (keyword + semantic merge)"
search_resp="$(curl -s -u admin:admin "http://localhost:8000/api/search/?query=integration" || echo '{}')"
docs_count="$(echo "$search_resp" | python3 -c "import sys,json;print(len(json.load(sys.stdin).get('documents',[])))" 2>/dev/null || echo 0)"
ml_added="$(echo "$search_resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('ml_semantic_added',0))" 2>/dev/null || echo 0)"
if [[ "$docs_count" -gt 0 ]]; then
    pass "/api/search/ returned $docs_count document(s); ml_semantic_added=$ml_added"
else
    fail "/api/search/?query=integration returned 0 documents (response: $(echo "$search_resp" | head -c 300))"
    exit 1
fi

# ---------- checkpoint 9: Prometheus is scraping ml-gateway /metrics ----------
info "Checkpoint 9: Prometheus scraping FastAPI metrics"
# Polls /api/v1/targets and checks health=="up" for job=="ml-gateway". Uses the
# plain /targets endpoint (no query params) to avoid URL-encoding issues.
attempt=0
target_health=""
while (( attempt < 30 )); do
    target_health="$(curl -s http://localhost:9090/api/v1/targets 2>/dev/null | python3 -c '
import sys, json
d = json.load(sys.stdin)
for t in d.get("data", {}).get("activeTargets", []):
    if t.get("labels", {}).get("job") == "ml-gateway":
        print(t.get("health", ""))
        break
' 2>/dev/null || echo "")"
    if [[ "$target_health" == "up" ]]; then
        break
    fi
    sleep 2
    attempt=$(( attempt + 1 ))
done
if [[ "$target_health" == "up" ]]; then
    pass "Prometheus reports ml-gateway target health=up"
else
    fail "Prometheus did not confirm ml-gateway target is up after 60s (got: ${target_health:-<empty>})"
    curl -s 'http://localhost:9090/api/v1/targets' | python3 -m json.tool | head -40 || true
    exit 1
fi

# ---------- checkpoint 10: paperless.uploads topic exists in Redpanda ----------
info "Checkpoint 10: paperless.uploads topic created by publisher"
attempt=0
topic_present=0
while (( attempt < 20 )); do
    if docker compose exec -T redpanda rpk topic list 2>/dev/null | awk '{print $1}' | grep -qx "paperless.uploads"; then
        topic_present=1
        break
    fi
    sleep 2
    attempt=$(( attempt + 1 ))
done
if (( topic_present )); then
    pass "redpanda topic paperless.uploads exists (producer fired at least once)"
else
    fail "paperless.uploads topic not found — publisher never succeeded"
    docker compose exec -T redpanda rpk topic list 2>/dev/null || true
    exit 1
fi

# ---------- checkpoint 11: corrections/queries/feedback events all publish ----------
info "Checkpoint 11: corrections + queries + feedback events emitted"
# Queries event already fired from checkpoint 8's /api/search/?query=integration.
# Trigger the remaining two by POSTing a correction and a search_click feedback.
LATEST_DOC_ID="$(curl -s -u admin:admin 'http://localhost:8000/api/documents/?ordering=-id&page_size=1' \
    | python3 -c 'import sys,json;r=json.load(sys.stdin)["results"];print(r[0]["id"] if r else 0)' 2>/dev/null || echo 0)"
if [[ "$LATEST_DOC_ID" -gt 0 ]]; then
    curl -s -u admin:admin -X POST http://localhost:8000/api/ml/feedback/ \
        -H 'Content-Type: application/json' \
        -d "{\"document\": $LATEST_DOC_ID, \"kind\": \"htr_correction\", \"correction_text\": \"verify harness test\"}" >/dev/null
    curl -s -u admin:admin -X POST http://localhost:8000/api/ml/feedback/ \
        -H 'Content-Type: application/json' \
        -d "{\"document\": $LATEST_DOC_ID, \"kind\": \"search_click\"}" >/dev/null
    sleep 2
fi
ok_q=0; ok_c=0; ok_f=0
wait_for_log paperless-web "published event to paperless.queries" 20 && ok_q=1
wait_for_log paperless-web "published event to paperless.corrections" 20 && ok_c=1
wait_for_log paperless-web "published event to paperless.feedback" 20 && ok_f=1
if (( ok_q && ok_c && ok_f )); then
    pass "All 3 downstream event types published (queries + corrections + feedback)"
else
    (( ok_q )) || fail "paperless.queries event missing"
    (( ok_c )) || fail "paperless.corrections event missing"
    (( ok_f )) || fail "paperless.feedback event missing"
    docker compose logs --tail=80 paperless-web | tail -40 || true
    exit 1
fi

# ---------- checkpoint 12: alerting pipeline wired ----------
info "Checkpoint 12: Prometheus rule group loaded + Alertmanager reachable"
# Three sub-checks: alerts.yml rule group is loaded, Prometheus knows about
# Alertmanager, and Alertmanager responds healthy.
ok_rules=0; ok_am_target=0; ok_am_health=0

rules_out="$(curl -s http://localhost:9090/api/v1/rules 2>/dev/null || echo '')"
if echo "$rules_out" | python3 -c '
import sys, json
d = json.load(sys.stdin)
groups = d.get("data", {}).get("groups", [])
print("1" if any(g.get("name") == "paperless-ml" for g in groups) else "0")
' 2>/dev/null | grep -qx 1; then
    ok_rules=1
fi

am_targets="$(curl -s http://localhost:9090/api/v1/alertmanagers 2>/dev/null || echo '')"
if echo "$am_targets" | python3 -c '
import sys, json
d = json.load(sys.stdin)
ams = d.get("data", {}).get("activeAlertmanagers", [])
print("1" if ams else "0")
' 2>/dev/null | grep -qx 1; then
    ok_am_target=1
fi

am_health="$(curl -s -o /dev/null -w '%{http_code}' http://localhost:9093/-/healthy 2>/dev/null || echo 000)"
if [[ "$am_health" == "200" ]]; then
    ok_am_health=1
fi

if (( ok_rules && ok_am_target && ok_am_health )); then
    pass "Rule group 'paperless-ml' loaded; Prometheus↔Alertmanager linked; Alertmanager healthy"
else
    (( ok_rules )) || fail "Prometheus did not load the 'paperless-ml' rule group (alerts.yml)"
    (( ok_am_target )) || fail "Prometheus has no active Alertmanager targets"
    (( ok_am_health )) || fail "Alertmanager /-/healthy not 200 (got: $am_health)"
    curl -s http://localhost:9090/api/v1/rules 2>/dev/null | head -c 500 || true
    exit 1
fi

# ---------- done ----------
echo
pass "All 12 checkpoints passed."
echo
echo "Stack is still running. Browse: http://localhost:8000 (admin / admin)"
echo "  Feedback API:  http://localhost:8000/api/ml/feedback/"
echo "  Prometheus:    http://localhost:9090/alerts"
echo "  Alertmanager:  http://localhost:9093"
echo "  Tear down:     docker compose down -v"
echo
echo "Smoke-test an alert firing:"
echo "  docker compose stop qdrant                 # wait ~90s"
echo "  curl -s http://localhost:9090/api/v1/alerts | python3 -m json.tool"
echo "  docker compose logs rollback-ctrl          # confirm webhook arrived"
echo "  docker compose start qdrant                # alert resolves"

if (( ! KEEP_UP )); then
    info "Leaving stack up (use --keep-up flag to suppress this notice)"
fi
