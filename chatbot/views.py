"""
Views for Vitas AI with OpenAI Integration - CORRECTED VERSION
FILE: chatbot/views.py (REPLACE COMPLETELY)
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import json
import os

from .models import ChatConversation, ChatMessage, UploadedFile, UserProfile, ContactMessage
from .forms import ContactForm, FileUploadForm

# Import AI models
from .ai_models import generate_ai_response


def home(request):
    """Homepage view"""
    if request.method == 'POST':
        contact_form = ContactForm(request.POST)
        if contact_form.is_valid():
            contact_form.save()
            messages.success(request, 'Thank you for contacting us! We will get back to you soon.')
            return redirect('home')
    else:
        contact_form = ContactForm()
    
    context = {
        'contact_form': contact_form
    }
    return render(request, 'chatbot/home.html', context)


@login_required
def chat_view(request):
    """Main chat interface"""
    user = request.user
    
    # Get or create user profile
    profile, created = UserProfile.objects.get_or_create(user=user)
    
    # Get user's conversations
    conversations = ChatConversation.objects.filter(user=user).order_by('-updated_at')
    
    # Get current conversation (latest or create new)
    current_conversation = conversations.first()
    if not current_conversation:
        current_conversation = ChatConversation.objects.create(
            user=user,
            title="New Chat",
            model_type=profile.preferred_model
        )
    
    # Get messages for current conversation - FIXED: Use 'timestamp' instead of 'created_at'
    messages_list = ChatMessage.objects.filter(conversation=current_conversation).order_by('timestamp')
    
    context = {
        'user': user,
        'profile': profile,
        'conversations': conversations,
        'current_conversation': current_conversation,
        'messages': messages_list,
    }
    return render(request, 'chatbot/chat.html', context)


@login_required
def new_chat(request):
    """Create a new chat conversation"""
    user = request.user
    profile, created = UserProfile.objects.get_or_create(user=user)
    
    conversation = ChatConversation.objects.create(
        user=user,
        title="New Chat",
        model_type=profile.preferred_model
    )
    
    return redirect('chat_conversation', conversation_id=conversation.id)


@login_required
def chat_conversation(request, conversation_id):
    """View specific conversation"""
    user = request.user
    conversation = get_object_or_404(ChatConversation, id=conversation_id, user=user)
    profile, created = UserProfile.objects.get_or_create(user=user)
    conversations = ChatConversation.objects.filter(user=user).order_by('-updated_at')
    
    # FIXED: Use 'timestamp' instead of 'created_at'
    messages_list = ChatMessage.objects.filter(conversation=conversation).order_by('timestamp')
    
    context = {
        'user': user,
        'profile': profile,
        'conversations': conversations,
        'current_conversation': conversation,
        'messages': messages_list,
    }
    return render(request, 'chatbot/chat.html', context)


@login_required
@csrf_exempt
def send_message(request):
    """Handle sending messages with AI integration"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            conversation_id = data.get('conversation_id')
            message_text = data.get('message')
            model_type = data.get('model_type', 'medicinal')
            
            # Validate message
            if not message_text or not message_text.strip():
                return JsonResponse({
                    'status': 'error',
                    'message': 'Message cannot be empty'
                }, status=400)
            
            # Get or create conversation
            conversation = None
            if conversation_id:
                try:
                    conversation = ChatConversation.objects.get(
                        id=conversation_id,
                        user=request.user
                    )
                except ChatConversation.DoesNotExist:
                    conversation = None
            
            # Create new conversation if needed
            if not conversation:
                title = message_text[:50]
                if len(message_text) > 50:
                    title += '...'
                
                conversation = ChatConversation.objects.create(
                    user=request.user,
                    title=title,
                    model_type=model_type
                )
            else:
                # Update existing conversation
                conversation.model_type = model_type
                if conversation.messages.count() == 0:
                    conversation.title = message_text[:50] + ('...' if len(message_text) > 50 else '')
                conversation.save()
            
            # Save user message
            user_message = ChatMessage.objects.create(
                conversation=conversation,
                role='user',
                content=message_text
            )
            
            # ============================================
            # AI RESPONSE GENERATION WITH GOOGLE GEMINI
            # ============================================
            
            # Check if there are any uploaded files in this conversation
            uploaded_files = UploadedFile.objects.filter(conversation=conversation).order_by('-uploaded_at')
            
            file_path = None
            file_type = None
            
            if uploaded_files.exists():
                latest_file = uploaded_files.first()
                file_path = latest_file.file.path
                file_type = latest_file.file_type
            
            # Generate AI response using Google Gemini
            response_text = generate_ai_response(
                message=message_text,
                model_type=model_type,
                file_path=file_path,
                file_type=file_type
            )
            
            # ============================================
            # END AI INTEGRATION
            # ============================================
            
            # Save assistant response
            assistant_message = ChatMessage.objects.create(
                conversation=conversation,
                role='assistant',
                content=response_text
            )
            
            # Update timestamp
            conversation.save()
            
            # Return response
            return JsonResponse({
                'status': 'success',
                'conversation_id': conversation.id,
                'conversation_title': conversation.title,
                'user_message': {
                    'id': user_message.id,
                    'content': user_message.content,
                    'timestamp': user_message.timestamp.strftime('%I:%M %p')
                },
                'assistant_message': {
                    'id': assistant_message.id,
                    'content': assistant_message.content,
                    'timestamp': assistant_message.timestamp.strftime('%I:%M %p')
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


@login_required
@csrf_exempt
def upload_file(request):
    """Handle file uploads with AI processing"""
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            uploaded_file = request.FILES['file']
            conversation_id = request.POST.get('conversation_id')
            
            # Get conversation if provided
            conversation = None
            if conversation_id:
                try:
                    conversation = ChatConversation.objects.get(
                        id=conversation_id,
                        user=request.user
                    )
                except ChatConversation.DoesNotExist:
                    pass
            
            # Create conversation if needed
            if not conversation:
                conversation = ChatConversation.objects.create(
                    user=request.user,
                    title=f"File: {uploaded_file.name[:40]}",
                    model_type='medicinal'
                )
            
            # Save file
            file_obj = UploadedFile.objects.create(
                conversation=conversation,
                file=uploaded_file,
                file_type=uploaded_file.content_type,
                original_name=uploaded_file.name
            )
            
            # File is now stored and will be processed when user asks questions
            
            return JsonResponse({
                'status': 'success',
                'message': f'File "{uploaded_file.name}" uploaded successfully! You can now ask questions about it.',
                'file_name': uploaded_file.name,
                'file_id': file_obj.id,
                'conversation_id': conversation.id
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)


@login_required
@csrf_exempt
def switch_model(request):
    """Switch between medicinal and ayurvedic models"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            conversation_id = data.get('conversation_id')
            model_type = data.get('model_type')
            
            # Validate model type
            if model_type not in ['medicinal', 'ayurvedic']:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid model type'
                }, status=400)
            
            # Update conversation if exists
            if conversation_id:
                try:
                    conversation = ChatConversation.objects.get(
                        id=conversation_id,
                        user=request.user
                    )
                    conversation.model_type = model_type
                    conversation.save()
                except ChatConversation.DoesNotExist:
                    pass
            
            # Update user's preferred model
            profile, created = UserProfile.objects.get_or_create(user=request.user)
            profile.preferred_model = model_type
            profile.save()
            
            return JsonResponse({
                'status': 'success',
                'model_type': model_type,
                'message': f'Switched to {model_type.capitalize()} model'
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


@login_required
def delete_conversation(request, conversation_id):
    """Delete a conversation"""
    conversation = get_object_or_404(ChatConversation, id=conversation_id, user=request.user)
    conversation.delete()
    messages.success(request, 'Chat deleted successfully')
    return redirect('chat')