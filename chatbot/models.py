"""
Database Models for Vitas AI
COPY THIS ENTIRE FILE INTO: chatbot/models.py (REPLACE ALL CONTENT)
"""
from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    """Extended user profile"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    phone = models.CharField(max_length=15, blank=True, null=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    preferred_model = models.CharField(
        max_length=20, 
        choices=[('medicinal', 'Medicinal'), ('ayurvedic', 'Ayurvedic')],
        default='medicinal'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"


class ChatConversation(models.Model):
    """Main conversation/chat session"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=200, default="New Chat")
    model_type = models.CharField(
        max_length=20,
        choices=[('medicinal', 'Medicinal'), ('ayurvedic', 'Ayurvedic')],
        default='medicinal'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.user.username} - {self.title}"


class ChatMessage(models.Model):
    """Individual messages in a conversation"""
    conversation = models.ForeignKey(ChatConversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(
        max_length=10,
        choices=[('user', 'User'), ('assistant', 'Assistant')]
    )
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.conversation.title} - {self.role}: {self.content[:50]}"


class UploadedFile(models.Model):
    conversation = models.ForeignKey(ChatConversation, on_delete=models.CASCADE, related_name='files')
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=100)
    original_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.original_name


class ContactMessage(models.Model):
    """Contact form submissions"""
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.subject}"
