from django.urls import path
from .views import agent_endpoint

urlpatterns = [
    path('agent/', agent_endpoint)
]
