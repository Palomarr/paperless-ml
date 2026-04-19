#!/usr/bin/env bash
# ===========================================================================
# seed_demo.sh — put the stack in a "ready to record" state.
#
# Uploads three realistic fixture documents, runs a few searches, submits
# corrections + ratings, optionally triggers and resolves an alert so the
# rollback-ctrl webhook fires on camera. After this runs, Grafana panels
# have real data and every demo-video bullet has something to point at.
#
# Usage:
#   bash scripts/seed_demo.sh                    # upload + search + feedback only
#   bash scripts/seed_demo.sh --trigger-alert    # also cycle qdrant to fire+resolve an alert
#   bash scripts/seed_demo.sh --host 192.5.86.1  # target a remote stack (default: localhost)
#
# Exits 0 if everything succeeded; 1 on first failure (fail-fast).
# ===========================================================================

set -euo pipefail

HOST="localhost"
TRIGGER_ALERT=0
AUTH="admin:admin"

while (( $# > 0 )); do
    case "$1" in
        --host)            HOST="$2"; shift 2 ;;
        --trigger-alert)   TRIGGER_ALERT=1; shift ;;
        -h|--help)
            grep '^#' "$0" | head -15
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

BASE="http://${HOST}:8000"
PROM="http://${HOST}:9090"

# ---------- output helpers ----------
C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
log()   { printf "%s==>%s %s\n" "$C_BOLD"   "$C_RESET" "$*"; }
ok()    { printf "%s ✓ %s %s\n" "$C_GREEN"  "$C_RESET" "$*"; }
warn()  { printf "%s !  %s %s\n" "$C_YELLOW" "$C_RESET" "$*"; }
fail()  { printf "%s ✗ %s %s\n" "$C_RED"    "$C_RESET" "$*"; exit 1; }

need() { command -v "$1" >/dev/null || fail "missing required command: $1"; }
need curl
need python3

# ---------- pre-flight ----------
log "Pre-flight: stack reachable at ${BASE}"
if ! curl -fsS -u "$AUTH" "${BASE}/api/ml/feedback/" >/dev/null; then
    fail "Paperless /api/ml/feedback/ not responding — is the stack up?"
fi
ok "Paperless + ml_hooks reachable"

# ---------- Step 1: write 3 fixture documents and upload ----------
log "Uploading 3 fixture documents via /api/documents/post_document/"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/invoice_acme_2387.txt" <<'DOC'
Invoice #2387
Acme Supplies Corporation
Date: March 15, 2026
Customer: Paperless Department

Office supplies line items:
  - 5 reams paper
  - 3 toner cartridges
  - 2 boxes staples

Subtotal: $4,200
Tax:      $336
Total:    $4,536

Payment terms: Net 30 days.
Contact billing@acme-supplies.com for questions.
DOC

cat > "$TMP/q2_planning_meeting.txt" <<'DOC'
Q2 2026 Planning Meeting Notes

Attendees: Alice, Bob, Carol, Dave

1. Budget review
   Acme Corporation contract renewal discussed. Total $12,500 quarterly.
   Office supplies budget increased to $5,000 after Q1 overrun.

2. Server room capacity
   Current utilization at 78%. Plans for expansion in Q3.
   New GPU nodes to be procured for ML workloads.

3. Hiring
   Approved: one senior engineer, two interns. Focus on
   infrastructure and data engineering roles.

Action items:
  - Bob: draft RFP for server expansion by April 25.
  - Carol: circulate revised Q2 budget.
DOC

cat > "$TMP/expense_policy_memo.txt" <<'DOC'
Memorandum
From: Operations
To:   All staff
Date: April 1, 2026

Re: Updated expense reporting policy

Effective immediately, all expense reports must be submitted within 14
days of incurring the expense. Receipts from vendors such as Acme
Corporation should be attached to the report.

Backup copies of receipts are automatically archived to the company
document management system; no manual filing is required.

Questions: contact Operations.
DOC

UPLOADED_IDS=()
for f in "$TMP"/*.txt; do
    title="$(basename "$f" .txt)"
    resp="$(curl -fsS -u "$AUTH" -X POST \
        -F "document=@${f}" \
        -F "title=${title}" \
        "${BASE}/api/documents/post_document/")"
    # Paperless returns a task UUID string (quoted)
    uuid="$(echo "$resp" | python3 -c 'import sys,json;print(json.load(sys.stdin))' 2>/dev/null || echo "$resp" | tr -d '"')"
    ok "Uploaded ${title} → task ${uuid:0:8}…"
done

# ---------- Step 2: wait for consumption ----------
log "Waiting up to 90s for all three documents to complete consumption"
DEADLINE=$((SECONDS + 90))
while (( SECONDS < DEADLINE )); do
    count="$(curl -fsS -u "$AUTH" "${BASE}/api/documents/?page_size=1" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin).get("count",0))' 2>/dev/null || echo 0)"
    if (( count >= 3 )); then
        ok "At least 3 documents now in the DB (total: $count)"
        break
    fi
    sleep 3
done

# ---------- Step 3: run 2 searches (populates paperless.queries + similarity) ----------
log "Running 2 semantic searches to populate query events + similarity histogram"
for q in "Acme Corporation supplies" "server room expansion"; do
    hit_count="$(curl -fsS -u "$AUTH" "${BASE}/api/search/?query=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$q")" \
        | python3 -c 'import sys,json;print(len(json.load(sys.stdin).get("documents",[])))' 2>/dev/null || echo 0)"
    ok "  query=\"$q\" → $hit_count document(s)"
done

# ---------- Step 4: submit feedback (corrections + ratings) ----------
log "Submitting 2 HTR corrections + 2 search ratings via /api/ml/feedback/"
LATEST_IDS="$(curl -fsS -u "$AUTH" "${BASE}/api/documents/?ordering=-id&page_size=3" \
    | python3 -c 'import sys,json;print(" ".join(str(r["id"]) for r in json.load(sys.stdin)["results"]))')"
read -r -a DOC_IDS <<< "$LATEST_IDS"

if (( ${#DOC_IDS[@]} < 3 )); then
    fail "Expected at least 3 documents for feedback seeding (got ${#DOC_IDS[@]})"
fi

for i in 0 1; do
    curl -fsS -u "$AUTH" -X POST \
        -H 'Content-Type: application/json' \
        -d "{\"document\": ${DOC_IDS[$i]}, \"kind\": \"htr_correction\", \"correction_text\": \"seeded correction for demo video ($i)\"}" \
        "${BASE}/api/ml/feedback/" >/dev/null
    ok "  correction on doc ${DOC_IDS[$i]}"
done

for i in 0 2; do
    rating=$((i == 0 ? 1 : 0))
    curl -fsS -u "$AUTH" -X POST \
        -H 'Content-Type: application/json' \
        -d "{\"document\": ${DOC_IDS[$i]}, \"kind\": \"search_rating\", \"rating\": $rating, \"query_text\": \"Acme Corporation supplies\"}" \
        "${BASE}/api/ml/feedback/" >/dev/null
    rating_label=$((rating == 1 ? 0 : 0)); [[ $rating -eq 1 ]] && rating_label="👍" || rating_label="👎"
    ok "  rating=$rating_label on doc ${DOC_IDS[$i]}"
done

# ---------- Step 5 (optional): fire + resolve an alert ----------
if (( TRIGGER_ALERT )); then
    log "Triggering QdrantDown alert (stopping qdrant for ~90s)"

    sg docker -c "docker compose stop qdrant" >/dev/null
    ok "qdrant stopped — alert should fire in ~60–90s (for: 1m)"

    # Poll /alerts until QdrantDown appears in firing state (max 120s)
    DEADLINE=$((SECONDS + 120))
    fired=0
    while (( SECONDS < DEADLINE )); do
        if curl -fsS "${PROM}/api/v1/alerts" 2>/dev/null \
            | python3 -c '
import sys, json
d = json.load(sys.stdin)
for a in d.get("data", {}).get("alerts", []):
    if a.get("labels", {}).get("alertname") == "QdrantDown" and a.get("state") == "firing":
        sys.exit(0)
sys.exit(1)
' 2>/dev/null; then
            fired=1
            break
        fi
        sleep 5
    done

    if (( fired )); then
        ok "QdrantDown is firing in Prometheus"
        log "rollback-ctrl webhook log (last 10 lines):"
        sg docker -c "docker compose logs --tail=10 rollback-ctrl" || true
    else
        warn "QdrantDown did not reach firing state in 120s"
    fi

    log "Restoring qdrant"
    sg docker -c "docker compose start qdrant" >/dev/null
    ok "qdrant restarted — alert should resolve within ~30s"
fi

# ---------- Summary ----------
echo
ok "Seed complete. Grafana panels should now show traffic within ~30s."
echo
echo "Rehearse the demo video against:"
echo "  Paperless UI (documents)  : ${BASE}/documents"
echo "  Feedback UI (/ml-ui/)     : ${BASE}/ml-ui/"
echo "  Grafana dashboard         : http://${HOST}:3000/d/paperless-ml-overview"
echo "  Prometheus alerts         : ${PROM}/alerts"
echo "  Alertmanager              : http://${HOST}:9093"
echo
if (( ! TRIGGER_ALERT )); then
    echo "Re-run with --trigger-alert to cycle qdrant and show the rollback webhook firing."
fi
