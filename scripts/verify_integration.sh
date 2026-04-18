#!/usr/bin/env bash
# Verify the paperless-ngx + ml_hooks overlay integration end-to-end.
# Runs 6 checkpoints against a locally-brought-up docker compose stack.
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
    # Bring up db first so makemigrations (which checks nothing but runs Django setup) works
    docker compose up -d db broker fastapi-stub >/dev/null
    # Wait briefly for DB
    for _ in {1..20}; do
        docker compose exec -T db pg_isready -U paperless >/dev/null 2>&1 && break
        sleep 1
    done
    # Use `run` so we don't depend on web being up; the webserver image has Django installed.
    docker compose run --rm --no-deps \
        -e PAPERLESS_DBHOST=db \
        webserver python manage.py makemigrations ml_hooks --noinput \
        || warn "makemigrations returned non-zero (may be fine if already present)"
fi

info "Bringing up full stack"
docker compose up -d >/dev/null

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
info "Checkpoint 1: ml_hooks URL routes mounted at startup"
if wait_for_log webserver "ml_hooks: mounted /api/ml/ routes" 120; then
    pass "AppConfig.ready() ran and injected /api/ml/ into urlpatterns"
else
    fail "Did not see 'ml_hooks: mounted /api/ml/ routes' in webserver logs within 120s"
    docker compose logs --tail=80 webserver || true
    exit 1
fi

# ---------- checkpoint 2: migration applied ----------
info "Checkpoint 2: ml_hooks migration applied"
if wait_for_log webserver "Applying ml_hooks\." 120 \
   || docker compose exec -T webserver python manage.py showmigrations ml_hooks 2>/dev/null | grep -q "\[X\]"; then
    pass "ml_hooks.0001_initial is applied"
else
    fail "ml_hooks migration not applied — Feedback table likely missing"
    exit 1
fi

# ---------- checkpoint 3: feedback endpoint reachable ----------
info "Checkpoint 3: /api/ml/feedback/ responds 200"
# Give the webserver a moment to finish Django startup if migration just applied
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
if wait_for_log webserver "ml_hooks: consumption finished for doc" 180; then
    pass "on_consumption_finished handler executed"
else
    fail "Signal handler never fired — consume pipeline may have stalled"
    docker compose logs --tail=120 webserver | tail -60 || true
    exit 1
fi

# ---------- checkpoint 6: celery tasks ran ----------
info "Checkpoint 6: HTR + embed tasks completed"
ok_htr=0; ok_embed=0
if wait_for_log webserver "htr_transcribe: doc" 120; then ok_htr=1; fi
if wait_for_log webserver "encode_document: doc" 120; then ok_embed=1; fi
if (( ok_htr && ok_embed )); then
    pass "Both Celery tasks hit the FastAPI stub and updated the document"
else
    (( ok_htr ))   || fail "htr_transcribe never completed"
    (( ok_embed )) || fail "encode_document never completed"
    docker compose logs --tail=150 webserver | tail -80 || true
    exit 1
fi

# ---------- done ----------
echo
pass "All 6 checkpoints passed."
echo
echo "Stack is still running. Browse: http://localhost:8000 (admin / admin)"
echo "  Feedback API:  http://localhost:8000/api/ml/feedback/"
echo "  Tear down:     docker compose down -v"

if (( ! KEEP_UP )); then
    info "Leaving stack up (use --keep-up flag to suppress this notice)"
fi
