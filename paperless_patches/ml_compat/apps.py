"""
ml_compat — translation shim between Paperless's integer doc IDs and Elnath's
data-stack-postgres UUID doc IDs.

Why: Elnath's paperless_ml app and its sidebar frontend operate on data-stack
documents.id (uuid). Paperless-ngx's REST API and URL routing use integer
documents.id. The two never aligned upstream, so:

  - GET /api/documents/<UUID>/  → Paperless 404 (sidebar's "Open Document" 404)
  - POST /api/ml/search/feedback/ {document_id: <int>} → paperless_ml UUID cast
    error (sidebar's thumbs feedback silently dropped)

This middleware bridges the gap at the request boundary, by looking up the
mapping in data-stack documents (which carries both id::uuid and
paperless_doc_id::integer + UNIQUE).

Removable in one commit when REDES01/paperless-ngx-ml emits paperless_doc_id
in queue responses and accepts integers in feedback POSTs (or aligns the
schema). Until then, this lives here so the demo isn't broken.
"""
from django.apps import AppConfig
from django.conf import settings


MIDDLEWARE_PATH = "ml_compat.middleware.DocIdCompatMiddleware"


class MlCompatConfig(AppConfig):
    name = "ml_compat"
    verbose_name = "ml/Paperless doc-id translation"

    def ready(self) -> None:
        # Append our middleware. Same pattern as ml_counters: mutate
        # settings.MIDDLEWARE before BaseHandler.load_middleware() runs.
        if MIDDLEWARE_PATH not in settings.MIDDLEWARE:
            settings.MIDDLEWARE = list(settings.MIDDLEWARE) + [MIDDLEWARE_PATH]
