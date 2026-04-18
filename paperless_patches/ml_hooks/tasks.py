import logging

from celery import shared_task
from documents.models import Document

from ml_hooks import ml_client

log = logging.getLogger("paperless.ml_hooks.tasks")

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

    # Real page/region rasterization lands in Job 4. Until then we skip
    # cleanly so the signal chain stays green during integration tests.
    # Contract for the real payload: contracts/htr_input.json
    log.info(
        "htr_transcribe: doc %s skipped (awaiting image pipeline, Job 4)",
        doc.pk,
    )


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

    # Real Qdrant upsert lands in Job 3 (needs the `/encode` endpoint on
    # FastAPI and the Qdrant service in compose). Skip cleanly for now.
    log.info(
        "encode_document: doc %s skipped (awaiting Qdrant wiring, Job 3)",
        doc.pk,
    )
