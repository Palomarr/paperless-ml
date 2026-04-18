"""Redpanda event publishing via Pandaproxy REST API.

Pandaproxy is Redpanda's HTTP frontend that accepts produce/consume requests
without requiring a Kafka client library. We use it so the Paperless container
doesn't need kafka-python / confluent-kafka installed.

All publishes are non-fatal: failures log a warning but never raise, so a
temporarily-down Redpanda never breaks Paperless's consume pipeline.

Topics (per contracts/CONTRACTS.md — coordinated with data role):
  paperless.uploads.v1     Document consumed, triggers HTR consumer
  paperless.corrections.v1 User corrected an HTR transcription
  paperless.queries.v1     User ran a search
  paperless.feedback.v1    User clicked / rated a search result
"""
import logging
import os
import uuid
from datetime import datetime, timezone

import requests

log = logging.getLogger("paperless.ml_hooks.events")

REDPANDA_PROXY_URL = os.getenv("REDPANDA_PROXY_URL", "http://redpanda:8082")
_PUBLISH_TIMEOUT_S = 5
_PANDAPROXY_JSON = "application/vnd.kafka.json.v2+json"


def _iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def publish(topic: str, event: dict) -> None:
    """Best-effort produce. Logs on success, warns on failure, never raises."""
    url = f"{REDPANDA_PROXY_URL.rstrip('/')}/topics/{topic}"
    body = {"records": [{"value": event}]}
    try:
        resp = requests.post(
            url,
            json=body,
            headers={"Content-Type": _PANDAPROXY_JSON},
            timeout=_PUBLISH_TIMEOUT_S,
        )
        resp.raise_for_status()
        log.info(
            "published event to %s: event_id=%s",
            topic,
            event.get("event_id"),
        )
    except Exception as exc:
        log.warning("redpanda publish to %s failed: %s", topic, exc)


def publish_upload_event(document) -> None:
    """Emit paperless.uploads.v1 when a document finishes consumption."""
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "paperless.uploads.v1",
        "emitted_at": _iso_now(),
        "paperless_doc_id": document.pk,
        "title": document.title,
        "mime_type": document.mime_type,
        "page_count": getattr(document, "page_count", None),
        "uploaded_at": document.created.isoformat() if document.created else None,
        "source": "paperless_web",
    }
    publish("paperless.uploads", event)
