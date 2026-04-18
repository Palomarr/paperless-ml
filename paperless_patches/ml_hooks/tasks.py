import logging

from celery import shared_task
from documents.models import Document

from ml_hooks import ml_client

log = logging.getLogger("ml_hooks.tasks")

HTR_CONFIDENCE_FLAG_THRESHOLD = 0.75


@shared_task(
    name="ml_hooks.htr_transcribe",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def htr_transcribe(self, document_id: int) -> None:
    doc = Document.objects.filter(pk=document_id).first()
    if doc is None:
        log.warning("htr_transcribe: document %s vanished", document_id)
        return

    # TODO(integration): replace stub with real page/region extraction.
    # Contract: paperless_patches/../contracts/htr_input.json
    payload = {"document_id": str(doc.pk)}
    result = ml_client.post("/htr", payload)

    text = result.get("htr_output", "").strip()
    if not text:
        return

    merged = f"{doc.content or ''}\n{text}".strip()
    flagged = result.get("htr_confidence", 1.0) < HTR_CONFIDENCE_FLAG_THRESHOLD
    Document.objects.filter(pk=doc.pk).update(content=merged)
    log.info("htr_transcribe: doc %s updated (flagged=%s)", doc.pk, flagged)


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
    ml_client.post("/search/encode", payload)
    log.info("encode_document: doc %s upserted to Qdrant", doc.pk)
