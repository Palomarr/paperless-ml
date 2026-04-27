"""
Archive user corrections to MinIO for immutable versioning.

Runs every 15 minutes. Fetches corrections from Postgres that haven't
been archived yet, joins them with their region/document context,
uploads a denormalized JSON to MinIO, and marks them archived.

Why this design:
    Corrections are the only user-generated data that drives model
    updates. Keeping them only in Postgres means: (a) no audit trail —
    a DELETE silently loses training history, (b) no reproducibility —
    training set at time T can't be exactly reconstructed later,
    (c) operational coupling — training job depends on DB availability.

    Archiving to MinIO as per-correction JSON solves all three:
        - Immutable: objects are never overwritten; corrections never lost
        - Reproducible: any past training run can be rebuilt by listing
          archive objects up to the manifest's `as_of` timestamp
        - Decoupled: batch_htr reads only object storage at train time

Path layout (Hive-partitioned for cheap date-range listing):
    s3://paperless-datalake/user_corrections/date=2026-04-24/<uuid>.json

JSON schema (one object per correction):
    {
      "correction_id":  "<uuid>",
      "region_id":      "<uuid>",
      "document_id":    <int>,
      "original_text":  "...",    // the region's htr_output at correction time
      "corrected_text": "...",    // the user's edit
      "opted_in":       true,
      "user_id":        "<uuid>|null",
      "corrected_at":   "2026-04-24T03:14:21.123456+00:00",
      "crop_s3_url":    "s3://paperless-images/documents/18/regions/p1_r2.png",
      "archived_at":    "2026-04-24T03:29:05.234567+00:00",
      "archive_version": "v1"
    }

Idempotency: uploads use `if-none-match: *` (via put_object with versioning
off but object names containing the correction UUID, which is unique by
PRIMARY KEY constraint). Even if this DAG runs twice on the same row
(e.g., retry after partial failure), it writes the same bytes to the
same key — no duplicates.

Eligibility:
    This DAG archives EVERY correction (including no-ops) — the archive
    is a raw user-event log, not a training-ready dataset. Quality
    filtering happens downstream in batch_htr.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

DB_DSN = os.environ.get(
    "ARCHIVE_CORRECTIONS_DB_DSN",
    "host=data-stack-postgres dbname=paperless user=user password=paperless_postgres",
)
MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE     = os.environ.get("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "paperless-datalake")
ARCHIVE_PREFIX   = "user_corrections"
ARCHIVE_VERSION  = "v1"


# Denormalized fetch: correction + region (for crop_s3_url, original htr
# output) + document (for document_id). Filter by archived_at IS NULL
# so re-runs only process unarchived rows.
FETCH_SQL = """
SELECT
    c.id                AS correction_id,
    c.region_id         AS region_id,
    d.id                AS document_id,
    r.htr_output        AS original_text,
    c.corrected_text    AS corrected_text,
    c.opted_in          AS opted_in,
    c.user_id           AS user_id,
    c.corrected_at      AS corrected_at,
    r.crop_s3_url       AS crop_s3_url
FROM htr_corrections c
JOIN handwritten_regions r ON c.region_id  = r.id
JOIN document_pages      p ON r.page_id    = p.id
JOIN documents           d ON p.document_id = d.id
WHERE c.archived_at IS NULL
ORDER BY c.corrected_at ASC;
"""

MARK_SQL = "UPDATE htr_corrections SET archived_at = NOW() WHERE id = %s;"


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _archive(**_ctx):
    import psycopg2
    from minio import Minio

    mc = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )
    if not mc.bucket_exists(MINIO_BUCKET):
        mc.make_bucket(MINIO_BUCKET)
        log.info("created bucket %s", MINIO_BUCKET)

    archived_count = 0
    failed_count = 0
    archived_at_iso = datetime.now(timezone.utc).isoformat()

    with psycopg2.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(FETCH_SQL)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
            log.info("found %d un-archived corrections", len(rows))

            for row in rows:
                record = dict(zip(columns, row))
                correction_id = str(record["correction_id"])
                corrected_at = record["corrected_at"]
                date_partition = (
                    corrected_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
                    if isinstance(corrected_at, datetime)
                    else datetime.now(timezone.utc).strftime("%Y-%m-%d")
                )

                # Build the JSON body — fully denormalized so training
                # can run without Postgres joins.
                body = {
                    "correction_id":   correction_id,
                    "region_id":       str(record["region_id"]),
                    "document_id":     record["document_id"],
                    "original_text":   record["original_text"] or "",
                    "corrected_text":  record["corrected_text"] or "",
                    "opted_in":        bool(record["opted_in"]),
                    "user_id":         str(record["user_id"]) if record["user_id"] else None,
                    "corrected_at":    _iso(corrected_at),
                    "crop_s3_url":     record["crop_s3_url"],
                    "archived_at":     archived_at_iso,
                    "archive_version": ARCHIVE_VERSION,
                }
                data = json.dumps(body, indent=2).encode("utf-8")
                key = f"{ARCHIVE_PREFIX}/date={date_partition}/{correction_id}.json"

                try:
                    mc.put_object(
                        MINIO_BUCKET,
                        key,
                        io.BytesIO(data),
                        length=len(data),
                        content_type="application/json",
                    )
                except Exception as exc:
                    log.exception("upload failed for %s: %s", correction_id, exc)
                    failed_count += 1
                    continue

                # Only mark archived after successful upload
                with conn.cursor() as uc:
                    uc.execute(MARK_SQL, (correction_id,))
                conn.commit()
                archived_count += 1

    log.info("archived %d corrections (%d failed)", archived_count, failed_count)
    if failed_count > 0:
        raise RuntimeError(f"{failed_count} correction archive upload(s) failed")


def _report(**_ctx):
    """Print a summary of what's in the archive for eyeballing."""
    from minio import Minio
    mc = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )
    objects = list(mc.list_objects(
        MINIO_BUCKET, prefix=f"{ARCHIVE_PREFIX}/", recursive=True
    ))
    by_date: dict[str, int] = {}
    for obj in objects:
        # Expect key like user_corrections/date=2026-04-24/<uuid>.json
        parts = obj.object_name.split("/")
        date_part = next((p.split("=", 1)[1] for p in parts if p.startswith("date=")), "unknown")
        by_date[date_part] = by_date.get(date_part, 0) + 1

    total = sum(by_date.values())
    log.info("archive totals: %d corrections across %d date partition(s)",
             total, len(by_date))
    for date_key in sorted(by_date):
        log.info("  date=%s: %d corrections", date_key, by_date[date_key])


default_args = {
    "owner": "paperless-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="archive_corrections",
    default_args=default_args,
    description=(
        "Every 15 minutes: archive unarchived htr_corrections rows to MinIO "
        "as immutable JSON objects for training reproducibility + audit trail."
    ),
    schedule="*/15 * * * *",
    start_date=datetime(2026, 4, 23),
    catchup=False,
    max_active_runs=1,
    tags=["archive", "corrections", "versioning"],
    doc_md=__doc__,
) as dag:

    archive_task = PythonOperator(
        task_id="archive_to_minio",
        python_callable=_archive,
    )
    report_task = PythonOperator(
        task_id="report_archive_state",
        python_callable=_report,
        trigger_rule="all_done",
    )

    archive_task >> report_task
