"""
DocIdCompatMiddleware — runs early in the chain. Three transforms:

  1. URL path: /api/documents/<UUID>/...  →  /api/documents/<int>/...
     (so Paperless's REST router resolves it; covers any code path that
     does pass through a UUID).

  2. Request body: POST /api/ml/search/feedback/ {document_id: int}
                   → mutate body so document_id is the data-stack UUID
     (so paperless_ml's INSERT into search_feedback.document_id::uuid casts).

  3. Response body: GET /api/ml/htr/queue/ → swap `document_id` from
     data-stack UUID to Paperless integer ID, so the sidebar's
     `[routerLink]="...e.document_id..."` resolves to /documents/<int>/
     (which Paperless's route handler accepts). The frontend reads
     `document_id` as a routerLink target; coercing a UUID via parseInt
     yields NaN → /api/documents/NaN/root/ → 404. Replacing the value
     with the integer fixes navigation. We also keep the data-stack
     UUID under `document_id_uuid` and add `paperless_doc_id` so any
     consumer that needs the original UUID (e.g., paperless_ml's chat)
     can still find it. Same swap is applied at the per-region level.

Lookups go to data-stack-postgres (Elnath's schema), where the documents
table carries both id::uuid (PK) and paperless_doc_id::integer (UNIQUE).
We cache mappings in process memory; doc IDs are immutable post-creation.

Failures are silent (passthrough) — translation is a best-effort overlay.
If data-stack is unreachable or a doc isn't there, we leave the request
unchanged and let the downstream view return its native error.
"""
import json
import logging
import os
import re

import psycopg
from django.utils.deprecation import MiddlewareMixin


log = logging.getLogger("ml_compat")

UUID_PATH_RE = re.compile(r"^/api/documents/([0-9a-fA-F-]{36})(/.*)?$")
SEARCH_FEEDBACK_PATH = "/api/ml/search/feedback/"
HTR_QUEUE_PATH = "/api/ml/htr/queue/"


def _dsn() -> str:
    return (
        f"host={os.environ.get('PAPERLESS_ML_DBHOST', 'data-stack-postgres')} "
        f"port={os.environ.get('PAPERLESS_ML_DBPORT', '5432')} "
        f"dbname={os.environ.get('PAPERLESS_ML_DBNAME', 'paperless')} "
        f"user={os.environ.get('PAPERLESS_ML_DBUSER', 'user')} "
        f"password={os.environ.get('PAPERLESS_ML_DBPASSWORD', 'paperless_postgres')}"
    )


class DocIdCompatMiddleware(MiddlewareMixin):
    # Class-level cache: maps both directions. Fine to keep across requests
    # since (uuid, paperless_doc_id) is set at insert time and never updates.
    _cache: dict = {}

    def _uuid_to_int(self, uuid_str: str):
        if uuid_str in self._cache:
            return self._cache[uuid_str]
        try:
            with psycopg.connect(_dsn(), connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT paperless_doc_id FROM documents WHERE id = %s",
                        (uuid_str,),
                    )
                    row = cur.fetchone()
        except Exception as exc:
            log.debug("ml_compat: uuid->int lookup failed (%s): %s", uuid_str, exc)
            return None
        if not row or row[0] is None:
            return None
        int_id = row[0]
        self._cache[uuid_str] = int_id
        self._cache[int_id] = uuid_str
        return int_id

    def _int_to_uuid(self, int_id: int):
        if int_id in self._cache:
            return self._cache[int_id]
        try:
            with psycopg.connect(_dsn(), connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id::text FROM documents WHERE paperless_doc_id = %s",
                        (int_id,),
                    )
                    row = cur.fetchone()
        except Exception as exc:
            log.debug("ml_compat: int->uuid lookup failed (%s): %s", int_id, exc)
            return None
        if not row or not row[0]:
            return None
        uuid_str = row[0]
        self._cache[int_id] = uuid_str
        self._cache[uuid_str] = int_id
        return uuid_str

    def process_request(self, request):
        # G2: rewrite /api/documents/<UUID>/... → /api/documents/<int>/...
        m = UUID_PATH_RE.match(request.path)
        if m:
            uuid_str, rest = m.group(1).lower(), m.group(2) or ""
            int_id = self._uuid_to_int(uuid_str)
            if int_id is not None:
                new_path = f"/api/documents/{int_id}{rest}"
                log.info("ml_compat: rewrote %s → %s", request.path, new_path)
                request.path = new_path
                request.path_info = new_path

        # G1: rewrite POST body for search_feedback if document_id is integer
        if (
            request.method == "POST"
            and request.path == SEARCH_FEEDBACK_PATH
            and request.content_type == "application/json"
        ):
            try:
                raw = request.body
                payload = json.loads(raw)
            except (ValueError, AttributeError):
                return None

            doc_id = payload.get("document_id")
            int_id = None
            if isinstance(doc_id, int):
                int_id = doc_id
            elif isinstance(doc_id, str) and doc_id.isdigit():
                int_id = int(doc_id)

            if int_id is not None:
                uuid_str = self._int_to_uuid(int_id)
                if uuid_str:
                    payload["document_id"] = uuid_str
                    new_body = json.dumps(payload).encode("utf-8")
                    # Replace cached body + reset stream so downstream view
                    # sees the rewritten payload. CONTENT_LENGTH must match.
                    request._body = new_body
                    request.META["CONTENT_LENGTH"] = str(len(new_body))
                    log.info(
                        "ml_compat: rewrote search_feedback document_id %s → %s",
                        int_id, uuid_str,
                    )
        return None

    def process_response(self, request, response):
        # G2 (response side): inject `paperless_doc_id` into queue response
        # so the sidebar's "Open Document" link reads an integer instead of
        # parseInt(undefined) → NaN.
        if (
            request.method == "GET"
            and request.path == HTR_QUEUE_PATH
            and getattr(response, "status_code", 0) == 200
            and "application/json" in response.get("Content-Type", "")
        ):
            try:
                data = json.loads(response.content)
            except (ValueError, AttributeError):
                return response
            if not isinstance(data, list):
                return response

            mutated = False
            for group in data:
                if not isinstance(group, dict):
                    continue
                doc_uuid = group.get("document_id")
                if not isinstance(doc_uuid, str):
                    continue
                int_id = self._uuid_to_int(doc_uuid)
                if int_id is None:
                    continue
                # Swap `document_id` value to the integer so the sidebar's
                # routerLink resolves to /documents/<int>/. Preserve the
                # original UUID under `document_id_uuid` and `paperless_doc_id`
                # for any consumer that still wants the UUID form.
                group["document_id_uuid"] = doc_uuid
                group["paperless_doc_id"] = int_id
                group["document_id"] = int_id
                # Same swap at region level (sidebar may read either).
                for region in group.get("regions", []) or []:
                    if isinstance(region, dict):
                        region["paperless_doc_id"] = int_id
                        region["document_id_uuid"] = doc_uuid
                        region["document_id"] = int_id
                mutated = True

            if mutated:
                new_body = json.dumps(data).encode("utf-8")
                response.content = new_body
                response["Content-Length"] = str(len(new_body))
                log.info(
                    "ml_compat: injected paperless_doc_id into htr queue response (%d groups)",
                    len(data),
                )
        return response
