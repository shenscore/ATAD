from django.urls import re_path

from label import consumers

websocket_urlpatterns = [
        re_path(r'tads',consumers.Consumer.as_asgi()),
]
