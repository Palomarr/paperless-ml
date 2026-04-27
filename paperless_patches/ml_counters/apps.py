"""
ml_counters — bridge between Elnath's paperless_ml fork and our ml-gateway counters.

Elnath's fork at REDES01/paperless-ngx@dev ships /api/ml/htr/corrections/ and
/api/ml/search/feedback/ as POST endpoints that write to the data-stack
postgres (htr_corrections, search_feedback tables). Our alert rules in
ops/prometheus/alerts.yml depend on counter ratios that are only emitted by
ml-gateway when /metrics/correction-recorded and /metrics/click-recorded are
hit (see serving/src/fastapi_app/app.py).

This app injects a single middleware that watches for successful POSTs on the
two paperless_ml endpoints and fires fire-and-forget HTTP calls to the
ml-gateway counter endpoints. It does NOT replace the paperless_ml writes —
it runs alongside them so the rate ratios (HtrCorrectionRateHigh,
SearchCTRLow) remain meaningful even when feedback flows through Elnath's
DB-backed views instead of our DRF FeedbackViewSet.
"""
from django.apps import AppConfig
from django.conf import settings


MIDDLEWARE_PATH = "ml_counters.middleware.CounterForwarderMiddleware"


class MlCountersConfig(AppConfig):
    name = "ml_counters"
    verbose_name = "ml-gateway counter forwarder"

    def ready(self) -> None:
        # Append our middleware to the runtime MIDDLEWARE list. Django's
        # BaseHandler.load_middleware() reads settings.MIDDLEWARE the first
        # time it's invoked — which is after apps.populate() finishes ready()
        # for every app. So mutating settings.MIDDLEWARE here lands before the
        # middleware chain is materialized.
        if MIDDLEWARE_PATH not in settings.MIDDLEWARE:
            settings.MIDDLEWARE = list(settings.MIDDLEWARE) + [MIDDLEWARE_PATH]
