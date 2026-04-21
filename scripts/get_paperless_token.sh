#!/usr/bin/env bash
# ===========================================================================
# get_paperless_token.sh — extract (or create) the Paperless DRF API token
# for the `admin` user. Idempotent: reuses the existing token if one exists.
#
# Required by data_generator (REDES01/paperless_data/data_generator/),
# which authenticates with `Authorization: Token <key>`.
#
# Usage:
#   export PAPERLESS_TOKEN=$(bash scripts/get_paperless_token.sh)
#
# Prerequisites:
#   - docker compose stack is up
#   - paperless-web is healthy enough to run `manage.py shell`
#
# On success: writes the 40-char token to stdout, followed by a newline.
# On failure: writes an error message to stderr and exits non-zero.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Self-wrap with `sg docker` if we can't talk to the Docker daemon directly.
# On fresh Chameleon nodes the `cc` user is in the docker group per
# /etc/group, but the current shell's effective groups don't include it
# until a new login (or `newgrp docker`). Detect + re-exec rather than
# require callers to know the difference.
if ! docker version >/dev/null 2>&1; then
    if getent group docker >/dev/null 2>&1 && command -v sg >/dev/null 2>&1; then
        exec sg docker -c "bash $0 $*"
    else
        echo "ERROR: docker not accessible and can't sg-wrap (missing 'docker' group or 'sg' command)" >&2
        exit 1
    fi
fi

# Paperless-web's admin user is created by PAPERLESS_ADMIN_USER env on first
# boot, but the DB must be migrated first. Wait up to 120s.
ATTEMPTS=0
until docker compose exec -T paperless-web python -c "import django" >/dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if (( ATTEMPTS > 60 )); then
        echo "ERROR: paperless-web not ready after 120s" >&2
        exit 1
    fi
    sleep 2
done

# Run Django shell to create-or-get the admin's DRF token.
# `manage.py shell -c` prints the script's stdout; we strip any stray Django
# warnings to stderr and capture just the final line.
docker compose exec -T paperless-web python manage.py shell -c "
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
u = get_user_model().objects.get(username='admin')
t, _ = Token.objects.get_or_create(user=u)
print(t.key)
" 2>/dev/null | tr -d '\r' | tail -n 1
