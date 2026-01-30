"""
URL configuration for chatbot app
CREATE NEW FILE: chatbot/urls.py
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat/', views.chat_view, name='chat'),
    path('chat/new/', views.new_chat, name='new_chat'),
    path('chat/<int:conversation_id>/', views.chat_conversation, name='chat_conversation'),
    path('chat/send/', views.send_message, name='send_message'),
    path('chat/upload/', views.upload_file, name='upload_file'),
    path('chat/switch-model/', views.switch_model, name='switch_model'),
    path('chat/delete/<int:conversation_id>/', views.delete_conversation, name='delete_conversation'),
]