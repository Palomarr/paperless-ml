"""
CounterForwarderMiddleware — fires ml-gateway counter increments when
Elnath's paperless_ml POST endpoints succeed.

Why a middleware: paperless_ml's views write to the data-stack postgres via
raw psycopg cursors (no Django ORM, no post_save signals to hook). The only
seam that fires reliably for every successful write is the response object
itself.
"""
import logging
import os

import requests
from django.utils.deprecation import MiddlewareMixin


log = logging.getLogger("ml_counters")

ML_GATEWAY_BASE_URL = os.environ.get(
    "ML_COUNTERS_GATEWAY_URL", "http://ml-gateway:8000"
)
COUNTER_TIMEOUT_S = float(os.environ.get("ML_COUNTERS_TIMEOUT_S", "1.0"))

# Map paperless_ml POST paths → ml-gateway counter increment endpoints.
PATH_TO_COUNTER_ENDPOINT = {
    "/api/ml/htr/corrections/": "/metrics/correction-recorded",
    "/api/ml/search/feedback/": "/metrics/click-recorded",
}


class CounterForwarderMiddleware(MiddlewareMixin):
    """
    Runs after every response. If the request was a 2xx POST to one of the
    paperless_ml feedback endpoints, fires a best-effort POST to the
    matching ml-gateway counter endpoint. Failures are logged but never
    propagated — feedback persistence (paperless_ml's DB write) is the
    source of truth; counter increments are observability sugar.
    """

    def process_response(self, request, response):
        if request.method != "POST":
            return response
        if not (200 <= response.status_code < 300):
            return response

        counter_path = PATH_TO_COUNTER_ENDPOINT.get(request.path)
        if not counter_path:
            return response

        url = f"{ML_GATEWAY_BASE_URL}{counter_path}"
        try:
            requests.post(url, timeout=COUNTER_TIMEOUT_S)
        except requests.RequestException as exc:
            log.debug(
                "ml_counters: forward to %s failed (non-fatal): %s", url, exc
            )

        return response
