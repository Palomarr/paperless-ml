import logging

from django.apps import AppConfig

log = logging.getLogger("paperless.ml_hooks")


class MlHooksConfig(AppConfig):
    name = "ml_hooks"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        self._connect_signals()
        self._install_url_routes()

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
        # 3. /ml-ui/* — Django template feedback UI (R4).
        #    HTR correction editor + search rating thumbs. Pure additive;
        #    Paperless's Angular SPA at / is untouched.
        paperless_urls.urlpatterns.insert(
            0,
            re_path(r"^ml-ui/", include("ml_hooks.urls_ui")),
        )

        setattr(paperless_urls, marker, True)
        clear_url_caches()
        log.info(
            "ml_hooks: mounted /api/ml/ routes, /api/search/ override, "
            "and /ml-ui/ feedback UI",
        )
