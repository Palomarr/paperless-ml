"""
Patched copy of paperless_data_integration/htr_consumer/db.py.

Drop-in replacement bind-mounted at /app/db.py via docker-compose.peer.yml.

Patch (G5): make `delete_existing_pages_and_regions` correction-aware.

Upstream behavior: on Kafka redelivery (e.g. after a transient
commit-failure or a forced retry), this function unconditionally deletes
all handwritten_regions for a document. If any region has user
corrections referencing it via `htr_corrections.region_id_fkey`, the
DELETE fails with ForeignKeyViolation, the transaction rolls back, and
the consumer logs an alarming traceback before committing the offset
and moving on. The doc's data is preserved (rollback) but the log noise
is misleading and operators can't tell real errors from this expected
case.

Patched behavior: pre-check for corrections; if any exist, raise a
clear `ReprocessSkipped` exception (subclass of Exception). The
consumer's outer `except Exception` catches it and logs at WARN level
without a traceback. User corrections stay intact.

This ships as a bind-mount overlay until upstream PR is merged.
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg

log = logging.getLogger(__name__)


class ReprocessSkipped(Exception):
    """Raised when reprocessing would clobber user corrections."""


def _conn_info() -> dict:
    return {
        "host":     os.environ.get("ML_DB_HOST", "postgres"),
        "port":     int(os.environ.get("ML_DB_PORT", "5432")),
        "dbname":   os.environ.get("ML_DB_NAME", "paperless"),
        "user":     os.environ.get("ML_DB_USER", "user"),
        "password": os.environ.get("ML_DB_PASSWORD", "paperless_postgres"),
    }


@contextmanager
def conn():
    c = psycopg.connect(**_conn_info(), autocommit=False)
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


# ── Data structures ────────────────────────────

@dataclass
class PageRow:
    page_number: int
    image_s3_url: str
    tesseract_text: str = ""
    htr_text: str = ""
    htr_confidence: float | None = None
    htr_flagged: bool = False


@dataclass
class RegionRow:
    crop_s3_url: str
    htr_output: str = ""
    htr_confidence: float | None = None
    page_number: int = 1


# ── Write operations ───────────────────────────

def upsert_document(
    cur,
    paperless_doc_id: int,
    title: str,
    page_count: int,
    tesseract_text: str,
    htr_text: str,
    merged_text: str,
    source: str = "user_upload",
) -> str:
    cur.execute(
        """
        INSERT INTO documents (
            filename, source, page_count, tesseract_text, htr_text,
            merged_text, paperless_doc_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (paperless_doc_id) DO UPDATE SET
            page_count      = EXCLUDED.page_count,
            tesseract_text  = EXCLUDED.tesseract_text,
            htr_text        = EXCLUDED.htr_text,
            merged_text     = EXCLUDED.merged_text,
            source          = EXCLUDED.source
        RETURNING id
        """,
        (title, source, page_count, tesseract_text, htr_text,
         merged_text, paperless_doc_id),
    )
    (ml_id,) = cur.fetchone()
    return str(ml_id)


def delete_existing_pages_and_regions(cur, document_id: str) -> None:
    """Clean slate before re-inserting pages and regions.

    Pre-check: if any user corrections reference handwritten_regions for
    this document, refuse to delete and raise ReprocessSkipped. The
    consumer treats this as a recoverable skip (no traceback, advances
    Kafka offset, preserves user feedback intact).
    """
    cur.execute(
        """
        SELECT count(*)
        FROM htr_corrections c
        JOIN handwritten_regions r ON c.region_id = r.id
        JOIN document_pages p     ON r.page_id   = p.id
        WHERE p.document_id = %s
        """,
        (document_id,),
    )
    (correction_count,) = cur.fetchone()
    if correction_count > 0:
        raise ReprocessSkipped(
            f"document_id={document_id} has {correction_count} user "
            "correction(s); skipping reprocess to preserve them"
        )

    cur.execute(
        """
        DELETE FROM handwritten_regions
        WHERE page_id IN (SELECT id FROM document_pages WHERE document_id = %s)
        """,
        (document_id,),
    )
    cur.execute(
        "DELETE FROM document_pages WHERE document_id = %s",
        (document_id,),
    )


def insert_page(cur, document_id: str, page: PageRow) -> str:
    cur.execute(
        """
        INSERT INTO document_pages (
            document_id, page_number, image_s3_url, tesseract_text,
            htr_text, htr_confidence, htr_flagged
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            document_id,
            page.page_number,
            page.image_s3_url,
            page.tesseract_text,
            page.htr_text,
            page.htr_confidence,
            page.htr_flagged,
        ),
    )
    (page_id,) = cur.fetchone()
    return str(page_id)


def insert_region(cur, page_id: str, region: RegionRow) -> str:
    cur.execute(
        """
        INSERT INTO handwritten_regions (
            page_id, crop_s3_url, htr_output, htr_confidence
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (page_id, region.crop_s3_url, region.htr_output, region.htr_confidence),
    )
    (region_id,) = cur.fetchone()
    return str(region_id)
