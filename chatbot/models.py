"""
Database Models for Vitas AI
"""
from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    """Extended profile attached 1-to-1 to every Django user."""

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
        verbose_name='User account',
        help_text='The Django user this profile belongs to.',
    )
    phone = models.CharField(
        max_length=15,
        blank=True,
        null=True,
        verbose_name='Phone number',
        help_text='Optional contact number.',
    )
    avatar = models.ImageField(
        upload_to='avatars/',
        blank=True,
        null=True,
        verbose_name='Profile picture',
        help_text='Uploaded avatar image.',
    )
    preferred_model = models.CharField(
        max_length=20,
        choices=[('medicinal', 'Medicinal (Gemini)'), ('ayurvedic', 'Ayurvedic (GGUF)')],
        default='medicinal',
        verbose_name='Preferred AI model',
        help_text='Which AI backend the user prefers by default.',
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Created')
    updated_at = models.DateTimeField(auto_now=True,     verbose_name='Last updated')

    class Meta:
        verbose_name        = 'User Profile'
        verbose_name_plural = 'User Profiles'
        ordering            = ['-created_at']

    def __str__(self):
        return f"{self.user.username}'s Profile"


class ChatConversation(models.Model):
    """A single chat session belonging to one user."""

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='conversations',
        verbose_name='Owner',
        help_text='The user who owns this conversation.',
    )
    title = models.CharField(
        max_length=200,
        default='New Chat',
        verbose_name='Chat title',
        help_text='Auto-generated from the first message.',
    )
    model_type = models.CharField(
        max_length=20,
        choices=[('medicinal', 'Medicinal (Gemini)'), ('ayurvedic', 'Ayurvedic (GGUF)')],
        default='medicinal',
        verbose_name='AI model used',
        help_text='Which AI backend was active for this conversation.',
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Started at')
    updated_at = models.DateTimeField(auto_now=True,     verbose_name='Last message at')

    class Meta:
        verbose_name        = 'Chat Conversation'
        verbose_name_plural = 'Chat Conversations'
        ordering            = ['-updated_at']

    def __str__(self):
        return f"[{self.model_type.upper()}] {self.user.username} — {self.title}"

    def message_count(self):
        return self.messages.count()
    message_count.short_description = 'Messages'


class ChatMessage(models.Model):
    """A single message inside a conversation (user or assistant turn)."""

    ROLE_CHOICES = [('user', '👤 User'), ('assistant', '🤖 Assistant')]

    conversation = models.ForeignKey(
        ChatConversation,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name='Conversation',
        help_text='The conversation this message belongs to.',
    )
    role = models.CharField(
        max_length=10,
        choices=ROLE_CHOICES,
        verbose_name='Sent by',
        help_text='"user" = human input, "assistant" = AI response.',
    )
    content = models.TextField(
        verbose_name='Message text',
        help_text='Full text of the message.',
    )
    timestamp = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Sent at',
    )

    class Meta:
        verbose_name        = 'Chat Message'
        verbose_name_plural = 'Chat Messages'
        ordering            = ['timestamp']

    def __str__(self):
        preview = self.content[:60].replace('\n', ' ')
        return f"[{self.role.upper()}] {preview}"


class UploadedFile(models.Model):
    """A file uploaded by the user during a conversation."""

    conversation = models.ForeignKey(
        ChatConversation,
        on_delete=models.CASCADE,
        related_name='files',
        verbose_name='Conversation',
    )
    file = models.FileField(
        upload_to='uploads/',
        verbose_name='File',
    )
    file_type = models.CharField(
        max_length=100,
        verbose_name='MIME type',
        help_text='e.g. image/png, application/pdf',
    )
    original_name = models.CharField(
        max_length=255,
        verbose_name='Original filename',
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Uploaded at',
    )

    class Meta:
        verbose_name        = 'Uploaded File'
        verbose_name_plural = 'Uploaded Files'
        ordering            = ['-uploaded_at']

    def __str__(self):
        return f"{self.original_name} ({self.file_type})"


class ContactMessage(models.Model):
    """Message submitted via the homepage contact form."""

    name = models.CharField(
        max_length=100,
        verbose_name='Sender name',
    )
    email = models.EmailField(
        verbose_name='Sender email',
    )
    subject = models.CharField(
        max_length=200,
        verbose_name='Subject',
    )
    message = models.TextField(
        verbose_name='Message body',
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='Received at',
    )
    is_read = models.BooleanField(
        default=False,
        verbose_name='Marked as read',
        help_text='Tick once you have reviewed this message.',
    )

    class Meta:
        verbose_name        = 'Contact Message'
        verbose_name_plural = 'Contact Messages'
        ordering            = ['-created_at']

    def __str__(self):
        status = '✅' if self.is_read else '🔴'
        return f"{status} {self.name} — {self.subject}"
