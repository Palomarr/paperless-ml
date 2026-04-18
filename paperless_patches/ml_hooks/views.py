import logging

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from ml_hooks import ml_client
from ml_hooks.models import Feedback
from ml_hooks.serializers import FeedbackSerializer

log = logging.getLogger("paperless.ml_hooks.views")


class FeedbackViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    queryset = Feedback.objects.all().order_by("-created_at")
    serializer_class = FeedbackSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        user = self.request.user if self.request.user.is_authenticated else None
        serializer.save(user=user)


# ---------------------------------------------------------------------------
# Modified global search view per architecture (line 824, 847-849, 1121).
# Wraps Paperless's GlobalSearchView; merges semantic results from
# FastAPI /search/query into the `documents` field. All other fields
# (tags, correspondents, etc.) untouched.
# ---------------------------------------------------------------------------

# Imports of Paperless internals are deferred to method bodies so this
# module loads cleanly even outside a running Paperless app context
# (e.g. during a standalone Python check).


def _ml_global_search_view():
    from documents.models import Document
    from documents.permissions import get_objects_for_user_owner_aware
    from documents.serialisers import DocumentSerializer
    from documents.views import GlobalSearchView

    SEMANTIC_TOP_K = 5

    class MlGlobalSearchView(GlobalSearchView):
        def get(self, request, *args, **kwargs):
            response = super().get(request, *args, **kwargs)
            if response.status_code != 200:
                return response

            query = request.query_params.get("query")
            if not query or len(query) < 3:
                return response

            try:
                semantic = ml_client.post(
                    "/search/query",
                    {"query_text": query, "top_k": SEMANTIC_TOP_K},
                )
            except Exception as exc:
                log.warning("ml semantic search unavailable: %s", exc)
                return response

            results = semantic.get("results") or []
            ids: list[int] = []
            for r in results:
                doc_id = str(r.get("document_id", ""))
                if doc_id.isdigit():
                    ids.append(int(doc_id))
            if not ids:
                return response

            visible_docs = get_objects_for_user_owner_aware(
                request.user, "view_document", Document,
            )
            semantic_docs = list(visible_docs.filter(pk__in=ids))
            if not semantic_docs:
                return response

            existing_ids = {d["id"] for d in response.data.get("documents", [])}
            new_docs = [d for d in semantic_docs if d.pk not in existing_ids]
            if not new_docs:
                return response

            serialized = DocumentSerializer(
                new_docs, many=True, context={"request": request},
            ).data
            response.data["documents"] = list(response.data.get("documents", [])) + list(
                serialized,
            )
            response.data["ml_semantic_added"] = len(new_docs)
            log.info(
                "ml_global_search: query=%r added %d semantic docs",
                query,
                len(new_docs),
            )
            return response

    return MlGlobalSearchView


def get_ml_global_search_view():
    """Return MlGlobalSearchView class, instantiated lazily."""
    return _ml_global_search_view()
