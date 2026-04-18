"""Celery tasks owned by ml_hooks.

HTR is intentionally NOT here. Architecturally, the data role's htr_consumer
(REDES01/paperless_data_integration) owns the HTR pipeline: it consumes
paperless.uploads events, slices pages into handwritten regions via the
region_slicer, calls ml-gateway /htr per region, and writes merged text back
through PATCH /api/documents/<id>/.

Our side emits the paperless.uploads event (see signal_handlers.py and
events.py) and encodes the current document.content to Qdrant. When the data
role later PATCHes the doc with HTR-merged text, Paperless fires
document_updated, our handler re-encodes, and the new text becomes searchable.
"""
import logging

from celery import shared_task
from documents.models import Document

from ml_hooks import ml_client

log = logging.getLogger("paperless.ml_hooks.tasks")


@shared_task(
    name="ml_hooks.encode_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def encode_document(self, document_id: int) -> None:
    doc = Document.objects.filter(pk=document_id).first()
    if doc is None or not doc.content:
        return

    payload = {"document_id": str(doc.pk), "text": doc.content}
    result = ml_client.post("/search/encode", payload)
    log.info(
        "encode_document: doc %s indexed %s chunks in Qdrant",
        doc.pk,
        result.get("chunks_indexed", "?"),
    )
