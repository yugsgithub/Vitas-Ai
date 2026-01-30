"""
Forms for Vitas AI
CREATE NEW FILE: chatbot/forms.py
"""
from django import forms
from .models import ContactMessage, UploadedFile


class ContactForm(forms.ModelForm):
    """Contact form for the homepage"""
    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'subject', 'message']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your Name',
                'required': True
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Your Email',
                'required': True
            }),
            'subject': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Subject',
                'required': True
            }),
            'message': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': 'Your Message',
                'rows': 5,
                'required': True
            }),
        }


class FileUploadForm(forms.ModelForm):
    """Form for uploading files in chat"""
    class Meta:
        model = UploadedFile
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'file-input',
                'accept': '.pdf,.jpg,.jpeg,.png,.doc,.docx'
            })
        }