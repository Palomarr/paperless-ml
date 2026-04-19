"""URL patterns for the ml_hooks feedback UI (served at /ml-ui/*).

Kept separate from urls.py (which holds the DRF API router) so the API
and UI namespaces stay visually distinct. Mounted at /ml-ui/ via
apps.py:_install_url_routes.
"""
from django.urls import path

from ml_hooks import views

urlpatterns = [
    path("", views.ui_index, name="ml_ui_index"),
    path("doc/<int:pk>/", views.ui_doc_feedback, name="ml_ui_doc_feedback"),
    path("search/", views.ui_search_feedback, name="ml_ui_search"),
]
