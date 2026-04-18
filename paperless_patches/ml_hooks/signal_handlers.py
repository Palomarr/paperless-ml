import logging

from ml_hooks import events
from ml_hooks import tasks

log = logging.getLogger("paperless.ml_hooks")


def on_consumption_finished(sender, document, **kwargs):
    log.info("ml_hooks: consumption finished for doc %s", document.pk)
    # Notify downstream data role (HTR consumer) via Redpanda.
    events.publish_upload_event(document)
    # Encode current document.content (Tesseract text) to Qdrant.
    # When HTR merged_text later lands via PATCH /api/documents/<id>/,
    # document_updated signal fires → on_document_updated re-encodes.
    tasks.encode_document.delay(document.pk)


def on_document_updated(sender, document, **kwargs):
    log.info("ml_hooks: document updated %s -> re-encode", document.pk)
    tasks.encode_document.delay(document.pk)
