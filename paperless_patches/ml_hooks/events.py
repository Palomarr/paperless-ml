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
# Pandaproxy creates topics on first produce, which can take 5–10s for
# metadata propagation. Give enough headroom that cold-start doesn't fail.
_PUBLISH_TIMEOUT_S = int(os.getenv("REDPANDA_PUBLISH_TIMEOUT_S", "15"))
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


def _username_of(user) -> str | None:
    """Normalize Django users (including AnonymousUser) to a plain username or None."""
    if not user or not getattr(user, "is_authenticated", False):
        return None
    return getattr(user, "username", None) or None


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


def publish_correction_event(feedback) -> None:
    """Emit paperless.corrections.v1 when a user submits an HTR correction.

    Optional region_id / original_htr_output / original_confidence /
    model_version_at_correction travel in Feedback.metadata (JSONField).
    """
    metadata = feedback.metadata or {}
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "paperless.corrections.v1",
        "emitted_at": _iso_now(),
        "paperless_doc_id": feedback.document_id,
        "corrected_text": feedback.correction_text,
        "user_id": _username_of(feedback.user),
        "region_id": metadata.get("region_id"),
        "original_htr_output": metadata.get("original_htr_output"),
        "original_confidence": metadata.get("original_confidence"),
        "model_version_at_correction": metadata.get("model_version_at_correction"),
    }
    publish("paperless.corrections", event)


def publish_feedback_event(feedback) -> None:
    """Emit paperless.feedback.v1 on search click / rating.

    session_id and query_id travel in Feedback.metadata so downstream
    analytics can correlate click → originating query.
    """
    metadata = feedback.metadata or {}
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "paperless.feedback.v1",
        "emitted_at": _iso_now(),
        "session_id": metadata.get("session_id", "anonymous"),
        "query_id": metadata.get("query_id"),
        "paperless_doc_id": feedback.document_id,
        "kind": feedback.kind,
        "rating": feedback.rating,
        "user_id": _username_of(feedback.user),
    }
    publish("paperless.feedback", event)


def publish_query_event(
    *,
    query_text: str,
    user,
    session_id: str,
    keyword_result_count: int,
    semantic_result_count: int,
    merged_result_ids: list,
    top_similarity_score,
    fallback_to_keyword: bool,
    model_version: str,
) -> None:
    """Emit paperless.queries.v1 after a merged search completes.

    Fires on every valid query (len >= 3), whether semantic added docs or not,
    so downstream can compute CTR denominators correctly.
    """
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "paperless.queries.v1",
        "emitted_at": _iso_now(),
        "session_id": session_id or "anonymous",
        "query_text": query_text,
        "user_id": _username_of(user),
        "keyword_result_count": keyword_result_count,
        "semantic_result_count": semantic_result_count,
        "merged_result_ids": merged_result_ids,
        "top_similarity_score": top_similarity_score,
        "fallback_to_keyword": fallback_to_keyword,
        "model_version": model_version,
    }
    publish("paperless.queries", event)
