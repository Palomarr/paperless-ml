"""Rollback controller stub.

Receives Alertmanager webhook payloads, logs the alert name/status/labels,
and returns 200. Actual model-repo symlink swap + Triton reload will be
implemented once Triton is deployed (tracked as N9 in docs/HANDOFF.md).

Architecture reference: architecture.html:968-970 ("Rollback controller:
Script triggered by Prometheus alertmanager webhook. If correction_rate
> 0.3 or avg_confidence < 0.6, swaps Triton model repo symlink to
previous version and signals reload.").
"""
import logging

from fastapi import FastAPI, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s rollback-ctrl: %(message)s",
)

app = FastAPI(title="rollback-ctrl (stub)")


@app.post("/webhook")
async def webhook(r: Request):
    body = await r.json()
    alerts = body.get("alerts", [])
    for alert in alerts:
        labels = alert.get("labels", {})
        logging.warning(
            "rollback-trigger alertname=%s status=%s severity=%s labels=%s",
            labels.get("alertname"),
            alert.get("status"),
            labels.get("severity"),
            labels,
        )
    # TODO(N9/Triton): swap model repo symlink to previous version and
    # POST to Triton's /v2/repository/models/<name>/unload + /load endpoints
    # once Triton is the serving backend. Blocked until N9.
    return {"received": len(alerts)}


@app.get("/health")
async def health():
    return {"status": "ok"}
