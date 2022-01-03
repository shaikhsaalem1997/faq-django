from django.conf.urls import url
from chatbot import views

urlpatterns = [
      url(r'^message$', views.messageApi),
      url(r'^message/([0-9]+)$', views.messageApi),
]

