import base64
import io
import logging
import os
from pathlib import Path

from celery import shared_task
from documents.models import Document

from ml_hooks import ml_client

log = logging.getLogger("paperless.ml_hooks.tasks")

HTR_CONFIDENCE_FLAG_THRESHOLD = 0.75
HTR_RASTER_DPI = int(os.getenv("ML_HTR_RASTER_DPI", "200"))
HTR_MAX_PAGES = int(os.getenv("ML_HTR_MAX_PAGES", "5"))


def _get_image_source_path(doc: Document) -> str | None:
    """Return filesystem path of the original file if it's a PDF or image."""
    path_str = getattr(doc, "source_path", None)
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    mime = (doc.mime_type or "").lower()
    if mime == "application/pdf" or mime.startswith("image/"):
        return str(path)
    return None


def _rasterize_pages(source_path: str, max_pages: int) -> list:
    """Return PIL images, one per page. Lazy-imports pdf2image so the module
    still loads cleanly when poppler is absent."""
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pdf2image import convert_from_path
        return convert_from_path(source_path, dpi=HTR_RASTER_DPI, last_page=max_pages)
    # Non-PDF image types: open directly as a single "page"
    from PIL import Image
    return [Image.open(source_path).convert("RGB")]


def _encode_png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _merge_htr_into_content(original: str, transcriptions: list[dict]) -> str:
    parts = [(original or "").strip()]
    for t in transcriptions:
        header = f"[HTR page {t['page']} — conf {t['confidence']:.2f}"
        if t.get("flagged"):
            header += " — low confidence"
        header += "]"
        parts.append(f"\n\n{header}\n{t['text'].strip()}")
    return "".join(parts).strip()


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

    source = _get_image_source_path(doc)
    if source is None:
        log.info(
            "htr_transcribe: doc %s skipped (mime=%s, no image source)",
            doc.pk,
            doc.mime_type,
        )
        return

    try:
        pages = _rasterize_pages(source, HTR_MAX_PAGES)
    except Exception as exc:
        log.warning("htr_transcribe: doc %s rasterization failed: %s", doc.pk, exc)
        return

    if not pages:
        log.info("htr_transcribe: doc %s produced 0 pages", doc.pk)
        return

    transcriptions: list[dict] = []
    for idx, page_image in enumerate(pages):
        payload = {
            "document_id": str(doc.pk),
            "page_id": f"page-{idx}",
            "region_id": f"page-{idx}-full",
            "image_base64": _encode_png_b64(page_image),
        }
        try:
            result = ml_client.post("/htr", payload)
        except Exception as exc:
            log.warning(
                "htr_transcribe: doc %s page %s request failed: %s",
                doc.pk,
                idx,
                exc,
            )
            continue

        text = (result.get("htr_output") or "").strip()
        if not text:
            continue
        confidence = float(result.get("htr_confidence", 0.0))
        transcriptions.append(
            {
                "page": idx,
                "text": text,
                "confidence": confidence,
                "flagged": confidence < HTR_CONFIDENCE_FLAG_THRESHOLD,
            }
        )

    if not transcriptions:
        log.info("htr_transcribe: doc %s produced no transcriptions", doc.pk)
        return

    merged = _merge_htr_into_content(doc.content, transcriptions)
    Document.objects.filter(pk=doc.pk).update(content=merged)
    log.info(
        "htr_transcribe: doc %s updated with %d transcription(s) across %d page(s)",
        doc.pk,
        len(transcriptions),
        len(pages),
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

    payload = {"document_id": str(doc.pk), "text": doc.content}
    result = ml_client.post("/search/encode", payload)
    log.info(
        "encode_document: doc %s indexed %s chunks in Qdrant",
        doc.pk,
        result.get("chunks_indexed", "?"),
    )
