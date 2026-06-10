from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name="homepage"),
    path('worker/', worker_dashboard, name="worker-dashboard"),
    path('worker/status/', worker_status, name="worker-status"),
    path('worker/start/', start_worker, name="worker-start"),
    path('worker/rescore/', rescore_worker, name="worker-rescore"),
    path('worker/stop/', stop_worker, name="worker-stop"),
    path('article/<slug:slug>/', ArticleDetailView.as_view(), name='article-detail'),
]
