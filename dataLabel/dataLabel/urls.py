from django.contrib import admin
from django.urls import include, path, re_path
from django.views.static import serve 
from django.conf import settings

urlpatterns = [
    #path('label/', include('label.urls')),
    path('', include('label.urls')),
    path('admin/', admin.site.urls),
    re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}), 
    re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}), 
]
