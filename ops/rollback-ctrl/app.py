"""Rollback controller.

Receives Alertmanager webhook payloads. For alerts labeled
`rollback_trigger: "true"` that are firing (not resolved), executes a
rollback: moves the `@production` alias on MLflow's htr
registered model to the previous version, then restarts the ml-gateway
container so it picks up the reverted weights on its next boot.

For rubric context (Joint responsibilities §12/15):
  "Three-person teams should incorporate re-training and CI/CD
   including models with well-justified rules, but do not necessarily
   need separate environments. However, if there are no separate
   environments, there must be an automated model roll back process
   that kicks in if the production system is not doing well."

That's this service.

Rollback-trigger alerts (defined in ops/prometheus/alerts.yml):
  HTRConfidenceLow      rolling 1h avg confidence < 0.6
  HTRCorrectionRateHigh correction rate > 0.3
  SearchCTRLow          click-through rate < 0.15 for 2h
  HtrInputDrift         sustained MMD drift > 0.2/s for 2m

Non-rollback alerts (availability / latency — severity=critical but
rollback_trigger absent) are logged only, not acted on: rolling back
the model doesn't fix an infrastructure outage.

--- Idempotency ---
Alertmanager retries webhooks on its own cadence. Without dedup, a
single alert could trigger a rollback twice within seconds. We keep a
per-alertname last-rollback timestamp; subsequent webhooks for the
same alert within COOLDOWN_SECONDS are logged and skipped. Different
alertnames are handled independently (HTRConfidenceLow and
HtrInputDrift can both fire and both roll back within the cooldown
window).

--- Security note ---
This container mounts /var/run/docker.sock so it can restart
ml-gateway on rollback. That grants root-equivalent access to the
Docker daemon — the accepted course-project tradeoff for a short-lived
demo. Production alternatives:
  - A privileged sidecar with a scoped RBAC role
  - Kubernetes ServiceAccount + RoleBinding allowing restart of the
    ml-gateway Deployment only
  - A separate admin-API endpoint on ml-gateway that accepts an
    HMAC-signed reload request, keeping Docker out of the loop
"""
import logging
import os
import threading
import time

import docker
import mlflow
from fastapi import FastAPI, Request
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s rollback-ctrl: %(message)s",
)
log = logging.getLogger("rollback-ctrl")

# --- Config (env-overridable for test) ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("ROLLBACK_MODEL_NAME", "htr")
PRODUCTION_ALIAS = os.environ.get("ROLLBACK_ALIAS", "production")
ML_GATEWAY_CONTAINER = os.environ.get("ML_GATEWAY_CONTAINER", "paperless-ml-ml-gateway-1")
COOLDOWN_SECONDS = int(os.environ.get("ROLLBACK_COOLDOWN_SECONDS", "300"))  # 5 min

# Per-alertname "last rollback at" timestamps. Thread-safe because the
# uvicorn worker count is 1 in our compose deployment; a Lock guards
# against future multi-worker configs.
_LOCK = threading.Lock()
_LAST_ROLLBACK_AT: dict[str, float] = {}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="rollback-ctrl")


def _in_cooldown(alertname: str) -> tuple[bool, float]:
    """Return (in_cooldown, seconds_remaining)."""
    with _LOCK:
        last = _LAST_ROLLBACK_AT.get(alertname, 0.0)
    elapsed = time.time() - last
    if elapsed < COOLDOWN_SECONDS:
        return True, COOLDOWN_SECONDS - elapsed
    return False, 0.0


def _mark_rolled_back(alertname: str) -> None:
    with _LOCK:
        _LAST_ROLLBACK_AT[alertname] = time.time()


def _execute_rollback(alertname: str) -> dict:
    """Swap @production alias to previous version + restart ml-gateway.

    Returns a dict describing what happened; the webhook response
    surfaces this to the caller (useful for manual curl tests).
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Find the currently-promoted version.
    try:
        current_mv = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        current_ver = int(current_mv.version)
        log.info(
            "rollback[%s]: current @%s is %s v%d",
            alertname, PRODUCTION_ALIAS, MODEL_NAME, current_ver,
        )
    except Exception as e:
        # No alias set yet. Fall back to the highest registered version
        # as "current" so we can still revert to the previous numeric version.
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            log.warning("rollback[%s]: no registered versions of %s; skipping",
                        alertname, MODEL_NAME)
            return {"status": "no_registered_versions", "model": MODEL_NAME}
        current_ver = max(int(v.version) for v in versions)
        log.warning(
            "rollback[%s]: no @%s alias (%s); assuming current=v%d",
            alertname, PRODUCTION_ALIAS, e, current_ver,
        )

    if current_ver <= 1:
        log.warning(
            "rollback[%s]: %s is at v%d — no previous version to revert to",
            alertname, MODEL_NAME, current_ver,
        )
        return {
            "status": "at_version_floor",
            "model": MODEL_NAME,
            "current_version": current_ver,
        }

    previous_ver = current_ver - 1

    # Swap the alias. MLflow set_registered_model_alias is idempotent —
    # reassigning to the same version is a no-op, so a concurrent
    # duplicate call is safe.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=str(previous_ver),
    )
    log.warning(
        "ROLLBACK: %s @%s moved v%d -> v%d (trigger=%s)",
        MODEL_NAME, PRODUCTION_ALIAS, current_ver, previous_ver, alertname,
    )

    # Restart ml-gateway so its next boot picks up the reverted weights.
    # A dynamic-reload endpoint on ml-gateway is deferred; for Phase 2
    # a ~90s restart is the accepted mechanism.
    try:
        docker_client = docker.from_env()
        container = docker_client.containers.get(ML_GATEWAY_CONTAINER)
        container.restart(timeout=30)
        log.warning(
            "ROLLBACK: ml-gateway container %s restart triggered",
            ML_GATEWAY_CONTAINER,
        )
        restart_status = "restarted"
    except Exception as e:
        log.error(
            "ROLLBACK: alias swapped but ml-gateway restart failed: %s",
            e,
        )
        restart_status = f"restart_failed: {e}"

    return {
        "status": "rollback_complete",
        "model": MODEL_NAME,
        "from_version": current_ver,
        "to_version": previous_ver,
        "ml_gateway": restart_status,
        "trigger_alert": alertname,
    }


@app.post("/webhook")
async def webhook(r: Request):
    body = await r.json()
    alerts = body.get("alerts", [])
    actions: list[dict] = []

    for alert in alerts:
        labels = alert.get("labels", {}) or {}
        status = alert.get("status", "")
        alertname = labels.get("alertname", "<unknown>")

        # Always log — this is the audit trail.
        log.warning(
            "rollback-trigger alertname=%s status=%s severity=%s labels=%s",
            alertname, status, labels.get("severity"), labels,
        )

        # Only rollback on *firing* alerts that explicitly request it.
        # Availability alerts (QdrantDown, etc.) log only — model
        # rollback won't fix an infra outage.
        if status != "firing":
            actions.append({"alertname": alertname, "action": "skipped_not_firing"})
            continue

        if labels.get("rollback_trigger") != "true":
            actions.append({"alertname": alertname, "action": "log_only"})
            continue

        # Idempotency: dedupe same-alertname retries within the cooldown.
        in_cool, remaining = _in_cooldown(alertname)
        if in_cool:
            log.info(
                "rollback[%s]: in cooldown (%.0fs remaining); skipping duplicate",
                alertname, remaining,
            )
            actions.append({
                "alertname": alertname,
                "action": "skipped_cooldown",
                "cooldown_remaining_seconds": int(remaining),
            })
            continue

        # Mark BEFORE executing so a concurrent retry doesn't race in.
        _mark_rolled_back(alertname)

        try:
            result = _execute_rollback(alertname)
            actions.append({"alertname": alertname, "action": "rollback", **result})
        except Exception as e:
            log.error("rollback[%s]: execution failed: %s", alertname, e)
            actions.append({
                "alertname": alertname,
                "action": "rollback_failed",
                "error": str(e),
            })

    return {"received": len(alerts), "actions": actions}


@app.get("/health")
async def health():
    return {"status": "ok"}
