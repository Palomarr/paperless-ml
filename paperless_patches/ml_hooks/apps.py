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

        marker = "_ml_hooks_installed"
        if getattr(paperless_urls, marker, False):
            return

        paperless_urls.urlpatterns.insert(
            0,
            re_path(r"^api/ml/", include("ml_hooks.urls")),
        )
        setattr(paperless_urls, marker, True)
        clear_url_caches()
        log.info("ml_hooks: mounted /api/ml/ routes")
