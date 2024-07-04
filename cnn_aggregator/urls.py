from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name="homepage"),
    path('sync/', sync_cnn, name="sync"),
    path('article/<slug:slug>/', ArticleDetailView.as_view(), name='article-detail'),
]