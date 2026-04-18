from rest_framework.routers import DefaultRouter

from ml_hooks.views import FeedbackViewSet

router = DefaultRouter()
router.register(r"feedback", FeedbackViewSet, basename="ml-feedback")

urlpatterns = router.urls
