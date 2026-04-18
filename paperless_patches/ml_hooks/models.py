from django.conf import settings
from django.db import models
from documents.models import Document


class Feedback(models.Model):
    class Kind(models.TextChoices):
        HTR_CORRECTION = "htr_correction", "HTR correction"
        SEARCH_CLICK = "search_click", "Search result click"
        SEARCH_RATING = "search_rating", "Search rating"

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="ml_feedback",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    kind = models.CharField(max_length=32, choices=Kind.choices)
    correction_text = models.TextField(blank=True, default="")
    rating = models.SmallIntegerField(null=True, blank=True)
    query_text = models.TextField(blank=True, default="")
    metadata = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["document", "kind"]),
            models.Index(fields=["created_at"]),
        ]
