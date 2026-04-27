"""
Search feedback reranker DAG.

Hourly aggregation of user thumbs-up/down feedback into per-document stats
that ml_gateway reads at query time to rerank search results.

Task graph:
    aggregate_feedback  ─▶  notify_result

Runs hourly so that feedback submitted in the UI is reflected in rankings
within an hour, not 24 hours. Uses PythonOperators (scheduler-local SQL)
rather than DockerOperator — the aggregation is a few SQL statements and
doesn't need an isolated image.

For ml_gateway to read the output, document_feedback_stats must exist in
the ML Postgres (schema migration 03_search_feedback_stats.sql).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

DB_DSN = os.environ.get(
    "SEARCH_RERANKER_DB_DSN",
    "host=data-stack-postgres dbname=paperless user=user password=paperless_postgres",
)

# Single statement: compute per-doc counts + rates and insert. Skips docs
# with zero impressions (their reranker-boost would be 0 anyway).
# Skips soft-deleted docs — their stats would never be used.
AGGREGATE_SQL = """
WITH impressions AS (
    SELECT DISTINCT
        unnest(result_doc_ids)::uuid AS document_id,
        id                           AS session_id
    FROM query_sessions
),
impression_counts AS (
    SELECT document_id, COUNT(*) AS total_impressions
    FROM impressions
    GROUP BY document_id
),
feedback_counts AS (
    SELECT
        document_id,
        SUM(CASE WHEN feedback_type = 'thumbs_up'   THEN 1 ELSE 0 END) AS thumbs_up,
        SUM(CASE WHEN feedback_type = 'thumbs_down' THEN 1 ELSE 0 END) AS thumbs_down,
        SUM(CASE WHEN feedback_type = 'click'       THEN 1 ELSE 0 END) AS clicks
    FROM search_feedback
    GROUP BY document_id
)
INSERT INTO document_feedback_stats
    (document_id, thumbs_up, thumbs_down, clicks, total_impressions, up_rate, down_rate, computed_at)
SELECT
    ic.document_id,
    COALESCE(fc.thumbs_up,   0),
    COALESCE(fc.thumbs_down, 0),
    COALESCE(fc.clicks,      0),
    ic.total_impressions,
    CASE WHEN ic.total_impressions > 0
         THEN COALESCE(fc.thumbs_up,   0)::float / ic.total_impressions
         ELSE 0.0 END,
    CASE WHEN ic.total_impressions > 0
         THEN COALESCE(fc.thumbs_down, 0)::float / ic.total_impressions
         ELSE 0.0 END,
    NOW()
FROM impression_counts ic
LEFT JOIN feedback_counts fc USING (document_id)
WHERE ic.document_id IS NOT NULL
  AND EXISTS (SELECT 1 FROM documents d WHERE d.id = ic.document_id AND d.deleted_at IS NULL);
"""


def _aggregate(**_ctx):
    import psycopg2
    log.info("aggregating search feedback into document_feedback_stats...")
    with psycopg2.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE document_feedback_stats")
        cur.execute(AGGREGATE_SQL)
        conn.commit()

        cur.execute("""
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE thumbs_up   > 0),
                   COUNT(*) FILTER (WHERE thumbs_down > 0)
            FROM document_feedback_stats
        """)
        total, has_up, has_down = cur.fetchone()
        log.info(
            "wrote %d doc stats rows (%d with thumbs_up, %d with thumbs_down)",
            total, has_up, has_down,
        )


def _log_snapshot(**_ctx):
    import psycopg2
    with psycopg2.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT document_id, thumbs_up, thumbs_down, clicks, total_impressions,
                   ROUND(up_rate::numeric,   3),
                   ROUND(down_rate::numeric, 3)
            FROM document_feedback_stats
            ORDER BY (thumbs_up + thumbs_down) DESC
            LIMIT 10
        """)
        rows = cur.fetchall()

    if not rows:
        log.info("no feedback stats yet — searches produce only pure-similarity rankings")
        return

    log.info("top documents by feedback volume:")
    log.info("%-40s %4s %4s %6s %8s %8s %8s",
             "document_id", "up", "down", "clicks", "impr", "up_r", "down_r")
    for doc_id, up, down, clk, imp, up_r, down_r in rows:
        log.info("%-40s %4d %4d %6d %8d %8s %8s",
                 str(doc_id)[:40], up, down, clk, imp, up_r, down_r)


default_args = {
    "owner": "paperless-ml",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="search_feedback_rerank",
    default_args=default_args,
    description=(
        "Hourly: aggregate user search feedback (thumbs-up/down/click) into "
        "document_feedback_stats for query-time reranking in ml_gateway."
    ),
    schedule="0 * * * *",  # hourly at :00
    start_date=datetime(2026, 4, 22),
    catchup=False,
    max_active_runs=1,
    tags=["search", "feedback", "rerank"],
    doc_md=__doc__,
) as dag:

    aggregate = PythonOperator(
        task_id="aggregate_feedback",
        python_callable=_aggregate,
    )

    snapshot = PythonOperator(
        task_id="notify_result",
        python_callable=_log_snapshot,
        trigger_rule="all_done",
    )

    aggregate >> snapshot
