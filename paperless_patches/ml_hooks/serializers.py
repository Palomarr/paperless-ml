from rest_framework import serializers

from ml_hooks.models import Feedback


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feedback
        fields = [
            "id",
            "document",
            "user",
            "kind",
            "correction_text",
            "rating",
            "query_text",
            "metadata",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]
