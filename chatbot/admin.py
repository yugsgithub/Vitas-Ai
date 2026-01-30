"""
Admin configuration for Vitas AI
COPY THIS ENTIRE FILE INTO: chatbot/admin.py (REPLACE ALL CONTENT)
"""
from django.contrib import admin
from .models import UserProfile, ChatConversation, ChatMessage, UploadedFile, ContactMessage


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'phone', 'preferred_model', 'created_at']
    list_filter = ['preferred_model', 'created_at']
    search_fields = ['user__username', 'user__email', 'phone']


@admin.register(ChatConversation)
class ChatConversationAdmin(admin.ModelAdmin):
    list_display = ['user', 'title', 'model_type', 'created_at', 'updated_at']
    list_filter = ['model_type', 'created_at']
    search_fields = ['user__username', 'title']
    date_hierarchy = 'created_at'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'role', 'content_preview', 'timestamp']
    list_filter = ['role', 'timestamp']
    search_fields = ['content', 'conversation__user__username']
    date_hierarchy = 'timestamp'
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = 'Content'


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ['original_name', 'conversation', 'file_type', 'uploaded_at']
    list_filter = ['file_type', 'uploaded_at']
    search_fields = ['original_name', 'conversation__user__username']
    date_hierarchy = 'uploaded_at'


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'subject', 'is_read', 'created_at']
    list_filter = ['is_read', 'created_at']
    search_fields = ['name', 'email', 'subject', 'message']
    date_hierarchy = 'created_at'
    actions = ['mark_as_read', 'mark_as_unread']
    
    def mark_as_read(self, request, queryset):
        queryset.update(is_read=True)
    mark_as_read.short_description = "Mark selected messages as read"
    
    def mark_as_unread(self, request, queryset):
        queryset.update(is_read=False)
    mark_as_unread.short_description = "Mark selected messages as unread"