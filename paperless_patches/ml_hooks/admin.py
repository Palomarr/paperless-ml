from django.contrib import admin

from ml_hooks.models import Feedback


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "kind", "user", "rating", "created_at")
    list_filter = ("kind", "created_at")
    search_fields = ("correction_text", "query_text")
