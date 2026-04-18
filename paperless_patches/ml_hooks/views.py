from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from ml_hooks.models import Feedback
from ml_hooks.serializers import FeedbackSerializer


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
