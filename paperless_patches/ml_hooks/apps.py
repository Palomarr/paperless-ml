import logging

from django.apps import AppConfig

log = logging.getLogger("paperless.ml_hooks")


class MlHooksConfig(AppConfig):
    name = "ml_hooks"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        self._connect_signals()
        self._install_url_routes()
        # _install_middleware() (FAB on document pages) deprecated in favor of
        # the sidebar tabs added by REDES01/paperless-ngx@dev's paperless_ml
        # frontend (HTR Review + Semantic Search). Sidebar is a strict UX
        # superset (region-level corrections, confidence color-coding, crop
        # preview, opt-in toggle, search mode selector, model_version display).

    def _connect_signals(self):
        from documents.signals import document_consumption_finished
        from documents.signals import document_updated

        from ml_hooks.signal_handlers import on_consumption_finished
        from ml_hooks.signal_handlers import on_document_updated

        document_consumption_finished.connect(
            on_consumption_finished,
            dispatch_uid="ml_hooks.on_consumption_finished",
        )
        document_updated.connect(
            on_document_updated,
            dispatch_uid="ml_hooks.on_document_updated",
        )

    def _install_url_routes(self):
        from django.urls import clear_url_caches
        from django.urls import include
        from django.urls import re_path
        from paperless import urls as paperless_urls

        from ml_hooks.views import get_ml_global_search_view

        marker = "_ml_hooks_installed"
        if getattr(paperless_urls, marker, False):
            return

        # 1. /api/ml/* — feedback API and other ml_hooks routes
        paperless_urls.urlpatterns.insert(
            0,
            re_path(r"^api/ml/", include("ml_hooks.urls")),
        )
        # 2. /api/search/ override — replaces Paperless's GlobalSearchView with
        #    one that merges semantic results from FastAPI. Inserted BEFORE
        #    the existing ^api/ include so Django matches our pattern first.
        paperless_urls.urlpatterns.insert(
            0,
            re_path(
                r"^api/search/$",
                get_ml_global_search_view().as_view(),
                name="ml_global_search",
            ),
        )
        # 3. /ml-ui/* — DEPRECATED. Redirects to the Angular sidebar tabs
        #    added by REDES01/paperless-ngx@dev's paperless_ml frontend. The
        #    Django template UI here was a strict subset — every feature is
        #    covered (and richer) by the sidebar UI: region-level corrections,
        #    confidence color-coding, crop preview, opt-in toggle, search
        #    mode selector, model_version display.
        from django.views.generic import RedirectView
        paperless_urls.urlpatterns.insert(
            0,
            re_path(
                r"^ml-ui/search/?$",
                RedirectView.as_view(url="/ml/search", permanent=True),
                name="ml_ui_search_deprecated",
            ),
        )
        paperless_urls.urlpatterns.insert(
            0,
            re_path(
                r"^ml-ui/.*$",
                RedirectView.as_view(url="/ml/htr-review", permanent=True),
                name="ml_ui_deprecated_redirect",
            ),
        )

        setattr(paperless_urls, marker, True)
        clear_url_caches()
        log.info(
            "ml_hooks: mounted /api/ml/ routes + /api/search/ override; "
            "/ml-ui/* now redirects to /ml/htr-review (sidebar UI canonical)",
        )

    def _install_middleware(self):
        """Append MlFeedbackFabMiddleware to settings.MIDDLEWARE if absent.

        Django's BaseHandler.load_middleware() reads settings.MIDDLEWARE at
        request-dispatch-time (first request after startup), so appending
        here in ready() lands before the chain is built. Idempotent: checks
        for the entry before appending so reloads don't duplicate it.
        """
        from django.conf import settings

        target = "ml_hooks.middleware.MlFeedbackFabMiddleware"
        middleware = list(settings.MIDDLEWARE)
        if target in middleware:
            return
        middleware.append(target)
        settings.MIDDLEWARE = middleware
        log.info("ml_hooks: installed MlFeedbackFabMiddleware")
