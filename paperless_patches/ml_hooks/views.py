import logging

from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from ml_hooks import events
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
        feedback = serializer.save(user=user)
        # Emit the appropriate Redpanda event for this feedback kind, and
        # fire-and-forget a metric hook to ml-gateway so the rollback-trigger
        # alerts (correction rate, CTR) have their numerator data.
        if feedback.kind == Feedback.Kind.HTR_CORRECTION:
            events.publish_correction_event(feedback)
            ml_client.post_fire_and_forget("/metrics/correction-recorded")
        elif feedback.kind == Feedback.Kind.SEARCH_CLICK:
            events.publish_feedback_event(feedback)
            ml_client.post_fire_and_forget("/metrics/click-recorded")
        elif feedback.kind == Feedback.Kind.SEARCH_RATING:
            events.publish_feedback_event(feedback)


# ---------------------------------------------------------------------------
# Modified global search view per architecture (line 824, 847-849, 1121).
# Wraps Paperless's GlobalSearchView; merges semantic results from
# FastAPI /search/query into the `documents` field. All other fields
# (tags, correspondents, etc.) untouched.
# Also publishes paperless.queries.v1 for every valid query.
# ---------------------------------------------------------------------------


def _ml_global_search_view():
    from documents.models import Document
    from documents.permissions import get_objects_for_user_owner_aware
    from documents.serialisers import DocumentSerializer
    from documents.views import GlobalSearchView

    SEMANTIC_TOP_K = 5

    class MlGlobalSearchView(GlobalSearchView):
        def get(self, request, *args, **kwargs):
            response = super().get(request, *args, **kwargs)
            query = request.query_params.get("query") or ""
            session_id = request.query_params.get("session_id") or ""

            keyword_ids: list[int] = []
            if response.status_code == 200:
                keyword_ids = [
                    d["id"] for d in response.data.get("documents", []) if "id" in d
                ]

            semantic_hit_count = 0
            top_similarity = None
            fallback = False
            model_version = "unknown"
            merged_ids = list(keyword_ids)

            if response.status_code == 200 and len(query) >= 3:
                try:
                    semantic = ml_client.post(
                        "/search/query",
                        {"query_text": query, "top_k": SEMANTIC_TOP_K},
                    )
                except Exception as exc:
                    log.warning("ml semantic search unavailable: %s", exc)
                    fallback = True
                    semantic = None

                if semantic is not None:
                    results = semantic.get("results") or []
                    semantic_hit_count = len(results)
                    fallback = bool(semantic.get("fallback_to_keyword"))
                    model_version = semantic.get("model_version", "unknown")
                    if results:
                        top_similarity = results[0].get("similarity_score")

                    ids: list[int] = []
                    for r in results:
                        doc_id = str(r.get("document_id", ""))
                        if doc_id.isdigit():
                            ids.append(int(doc_id))

                    if ids:
                        visible_docs = get_objects_for_user_owner_aware(
                            request.user, "view_document", Document,
                        )
                        semantic_docs = list(visible_docs.filter(pk__in=ids))
                        existing_ids = set(keyword_ids)
                        new_docs = [
                            d for d in semantic_docs if d.pk not in existing_ids
                        ]
                        if new_docs:
                            serialized = DocumentSerializer(
                                new_docs, many=True, context={"request": request},
                            ).data
                            response.data["documents"] = list(
                                response.data.get("documents", []),
                            ) + list(serialized)
                            response.data["ml_semantic_added"] = len(new_docs)
                            log.info(
                                "ml_global_search: query=%r added %d semantic docs",
                                query,
                                len(new_docs),
                            )
                            merged_ids = [
                                d["id"]
                                for d in response.data["documents"]
                                if "id" in d
                            ]

            # Best-effort query event; failures logged but non-fatal.
            if len(query) >= 3:
                events.publish_query_event(
                    query_text=query,
                    user=request.user,
                    session_id=session_id,
                    keyword_result_count=len(keyword_ids),
                    semantic_result_count=semantic_hit_count,
                    merged_result_ids=merged_ids,
                    top_similarity_score=top_similarity,
                    fallback_to_keyword=fallback,
                    model_version=model_version,
                )

            return response

    return MlGlobalSearchView


def get_ml_global_search_view():
    """Return MlGlobalSearchView class, instantiated lazily."""
    return _ml_global_search_view()
