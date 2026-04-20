import functools
import logging

from django.db.models import Count
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.authentication import BasicAuthentication
from rest_framework.permissions import IsAuthenticated

from ml_hooks import events
from ml_hooks import ml_client
from ml_hooks.models import Feedback
from ml_hooks.serializers import FeedbackSerializer

log = logging.getLogger("paperless.ml_hooks.views")


def _ui_login_required(view_func):
    """Like django.contrib.auth.decorators.login_required, but also accepts DRF Basic Auth.

    Plain Django views rely on session middleware to populate request.user
    from the session cookie. Scripts using `curl -u user:pass` send a Basic
    Auth header that Django's session middleware ignores, so the stock
    login_required decorator would 302-redirect to the login page. We mirror
    DRF's BasicAuthentication here so the UI works for both browser users
    (session cookie) and scripts (Basic Auth).
    """

    @functools.wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            try:
                result = BasicAuthentication().authenticate(request)
            except Exception:
                result = None
            if result:
                request.user = result[0]

        if not request.user.is_authenticated:
            from django.contrib.auth.views import redirect_to_login

            return redirect_to_login(request.get_full_path())
        return view_func(request, *args, **kwargs)

    return wrapper


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
                            # Bump `total` so the Paperless UI renders the
                            # merged count. Paperless's GlobalSearchView sets
                            # `total` from keyword matches only; without this
                            # update, a query that matched 0 keyword docs but
                            # added N semantic docs would show "0 results" in
                            # the UI even though `documents` has N items.
                            response.data["total"] = len(response.data["documents"])
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


# ---------------------------------------------------------------------------
# Feedback UI views (R4) — standalone Django template pages at /ml-ui/*.
# Reuse the same side-effect chain as FeedbackViewSet: row insert, Redpanda
# event, ml-gateway metric hook. See docs/superpowers/specs/2026-04-18-feedback-ui-design.md.
# ---------------------------------------------------------------------------


@_ui_login_required
def ui_index(request):
    from documents.models import Document

    docs = (
        Document.objects.all()
        .annotate(ml_feedback_count=Count("ml_feedback"))
        .order_by("-created")[:20]
    )
    return render(request, "ml_hooks/index.html", {"docs": docs})


@_ui_login_required
def ui_doc_feedback(request, pk: int):
    from documents.models import Document

    doc = get_object_or_404(Document, pk=pk)

    if request.method == "POST":
        corrected = (request.POST.get("corrected_text") or "").strip()
        if not corrected:
            return render(
                request,
                "ml_hooks/doc_feedback.html",
                {"doc": doc, "error": "Correction text cannot be empty."},
            )
        fb = Feedback.objects.create(
            document=doc,
            user=request.user if request.user.is_authenticated else None,
            kind=Feedback.Kind.HTR_CORRECTION,
            correction_text=corrected,
        )
        try:
            events.publish_correction_event(fb)
        except Exception as exc:
            log.warning("ml_hooks ui: correction event publish failed: %s", exc)
        ml_client.post_fire_and_forget("/metrics/correction-recorded")
        return HttpResponseRedirect(
            reverse("ml_ui_doc_feedback", args=[doc.id]) + "?saved=1",
        )

    recent_feedback = (
        Feedback.objects.filter(document=doc).order_by("-created_at")[:10]
    )
    char_count = len(doc.content or "")
    return render(
        request,
        "ml_hooks/doc_feedback.html",
        {
            "doc": doc,
            "recent_feedback": recent_feedback,
            "char_count": char_count,
            "saved": request.GET.get("saved"),
        },
    )


@_ui_login_required
def ui_search_feedback(request):
    from documents.models import Document

    if request.method == "POST":
        doc_id = request.POST.get("document") or ""
        try:
            rating = int(request.POST.get("rating", "0"))
        except ValueError:
            rating = 0
        query_text = request.POST.get("q") or ""
        doc = get_object_or_404(Document, pk=doc_id)
        fb = Feedback.objects.create(
            document=doc,
            user=request.user if request.user.is_authenticated else None,
            kind=Feedback.Kind.SEARCH_RATING,
            rating=1 if rating == 1 else 0,
            query_text=query_text,
        )
        try:
            events.publish_feedback_event(fb)
        except Exception as exc:
            log.warning("ml_hooks ui: feedback event publish failed: %s", exc)
        redirect = reverse("ml_ui_search")
        return HttpResponseRedirect(
            f"{redirect}?q={query_text}&rated={doc_id}",
        )

    query = (request.GET.get("q") or "").strip()
    results: list[dict] = []
    error = None
    if len(query) >= 3:
        try:
            payload = ml_client.post(
                "/search/query",
                {"query_text": query, "top_k": 5},
            )
            results = payload.get("results") or []
        except Exception as exc:
            log.warning("ml_hooks ui: search backend unavailable: %s", exc)
            error = str(exc)

    return render(
        request,
        "ml_hooks/search_feedback.html",
        {
            "query": query,
            "results": results,
            "error": error,
            "rated": request.GET.get("rated"),
            "saved": request.GET.get("rated"),
        },
    )
