"""
AI Models Integration for Vitas AI using Google Gemini
FILE: chatbot/ai_models.py (REPLACE YOUR CURRENT ONE)
This version automatically tries multiple model names until one works
"""

import google.generativeai as genai
import os
from typing import Optional
import PyPDF2
from PIL import Image
import pytesseract
import io

# Configure Google Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'your-api-key-here')
genai.configure(api_key=GEMINI_API_KEY)

# List of models to try (in order of preference)
MODEL_NAMES = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-pro',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-pro',
    'models/gemini-pro',
    'models/gemini-1.5-flash-latest',
    'models/gemini-1.5-pro-latest',
    'models/gemini-pro-latest',
]

# Cache the working model name
_working_model = None


def get_working_model():
    """Find and return a working Gemini model"""
    global _working_model
    
    # If we already found a working model, use it
    if _working_model:
        return _working_model
    
    # Try each model
    print("Finding working Gemini model...")
    for model_name in MODEL_NAMES:
        try:
            model = genai.GenerativeModel(model_name)
            # Test with a simple prompt
            response = model.generate_content("test")
            print(f"✅ Found working model: {model_name}")
            _working_model = model_name
            return model_name
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)[:50]}")
            continue
    
    # If no model works, return the first one and let it fail with proper error
    print("⚠️ No working model found, using default")
    _working_model = 'gemini-pro'
    return _working_model


# ============================================
# SYSTEM PROMPTS
# ============================================

MEDICINAL_SYSTEM_PROMPT = """You are a professional medical AI assistant specialized in modern medicine. 

STRICT RULES:
1. ONLY answer medical and health-related questions
2. If the question is NOT about medicine, health, diseases, treatments, symptoms, or medical advice, respond with: "I can only answer medical and health-related questions. Please ask me about symptoms, diseases, treatments, medications, or general health concerns."
3. Provide evidence-based medical information
4. Always include a disclaimer: "This is for informational purposes only. Please consult a healthcare professional for personalized medical advice."
5. Be professional, accurate, and helpful
6. Use modern medical terminology
7. Cite medical guidelines when relevant

TOPICS YOU CAN ANSWER:
- Symptoms and diseases
- Medical treatments and procedures
- Medications and their effects
- Health conditions and diagnoses
- Medical test results
- Preventive healthcare
- Nutrition and diet (health-related)
- Mental health
- First aid
- Medical emergencies

TOPICS YOU MUST REJECT:
- General knowledge questions
- Non-medical topics
- Casual conversation
- Programming, math, history (unless related to medicine)
- Entertainment, sports, politics
- Any non-health related queries

Answer in a professional, empathetic manner."""

AYURVEDIC_SYSTEM_PROMPT = """You are an expert Ayurvedic practitioner AI assistant specializing in holistic Ayurvedic medicine.

STRICT RULES:
1. ONLY answer questions related to Ayurveda, holistic health, and natural remedies
2. If the question is NOT about Ayurveda, natural health, herbs, doshas, or holistic wellness, respond with: "I can only answer questions related to Ayurveda and holistic health. Please ask me about Ayurvedic remedies, doshas, herbs, natural treatments, or holistic wellness."
3. Provide traditional Ayurvedic knowledge and remedies
4. Always include: "This is based on Ayurvedic principles. For serious conditions, please consult both an Ayurvedic practitioner and a medical doctor."
5. Be knowledgeable about doshas, herbs, and Ayurvedic treatments
6. Explain concepts in simple terms
7. Recommend natural and holistic approaches

TOPICS YOU CAN ANSWER:
- Ayurvedic remedies and treatments
- Dosha balance (Vata, Pitta, Kapha)
- Herbal medicines and their uses
- Ayurvedic diet and nutrition
- Panchakarma and detoxification
- Mind-body balance
- Natural wellness practices
- Traditional healing methods
- Yoga and meditation (health aspects)

TOPICS YOU MUST REJECT:
- General knowledge questions
- Non-health related topics
- Modern medicine (refer to medicinal model)
- Casual conversation
- Any non-wellness queries

Answer in a warm, holistic manner."""


# ============================================
# PDF TEXT EXTRACTION
# ============================================

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# ============================================
# IMAGE TEXT EXTRACTION (OCR)
# ============================================

def extract_text_from_image(file_path: str) -> str:
    """Extract text from image using OCR"""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error reading image: {str(e)}"


# ============================================
# FILE PROCESSING
# ============================================

def process_uploaded_file(file_path: str, file_type: str) -> Optional[str]:
    """Process uploaded file and extract text"""
    try:
        if 'pdf' in file_type.lower():
            return extract_text_from_pdf(file_path)
        elif any(img_type in file_type.lower() for img_type in ['image', 'jpeg', 'jpg', 'png']):
            return extract_text_from_image(file_path)
        elif 'text' in file_type.lower():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


# ============================================
# MEDICINAL AI MODEL (GEMINI)
# ============================================

def get_medicinal_response(message: str, file_context: Optional[str] = None) -> str:
    """Get response from Medicinal AI model using Google Gemini"""
    try:
        # Get working model name
        model_name = get_working_model()
        model = genai.GenerativeModel(model_name)
        
        # Build the prompt
        full_prompt = f"{MEDICINAL_SYSTEM_PROMPT}\n\n"
        
        if file_context:
            full_prompt += f"Context from uploaded file:\n{file_context}\n\n"
        
        full_prompt += f"User question: {message}"
        
        # Generate response
        response = model.generate_content(full_prompt)
        
        ai_response = response.text
        
        return f"💊 {ai_response}"
        
    except Exception as e:
        error_msg = str(e)
        print(f"Medicinal AI Error: {error_msg}")
        
        if "API" in error_msg.upper() or "KEY" in error_msg.upper():
            return f"💊 ⚠️ API Key Error: Please verify your Gemini API key in .env file.\n\nGet a free key at: https://aistudio.google.com/app/apikey"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return f"💊 ⚠️ Model Error: Could not find a working Gemini model.\n\nTry:\n1. Update package: pip install --upgrade google-generativeai\n2. Enable API: https://console.cloud.google.com/apis/library (search 'Generative Language API')\n3. Get new API key: https://aistudio.google.com/app/apikey"
        elif "QUOTA" in error_msg.upper() or "LIMIT" in error_msg.upper():
            return f"💊 ⚠️ Rate Limit: You've reached the API limit. Please wait a minute and try again."
        else:
            return f"💊 ⚠️ Error: {error_msg}"


# ============================================
# AYURVEDIC AI MODEL (GEMINI)
# ============================================

def get_ayurvedic_response(message: str, file_context: Optional[str] = None) -> str:
    """Get response from Ayurvedic AI model using Google Gemini"""
    try:
        # Get working model name
        model_name = get_working_model()
        model = genai.GenerativeModel(model_name)
        
        # Build the prompt
        full_prompt = f"{AYURVEDIC_SYSTEM_PROMPT}\n\n"
        
        if file_context:
            full_prompt += f"Context from uploaded file:\n{file_context}\n\n"
        
        full_prompt += f"User question: {message}"
        
        # Generate response
        response = model.generate_content(full_prompt)
        
        ai_response = response.text
        
        return f"🌿 {ai_response}"
        
    except Exception as e:
        error_msg = str(e)
        print(f"Ayurvedic AI Error: {error_msg}")
        
        if "API" in error_msg.upper() or "KEY" in error_msg.upper():
            return f"🌿 ⚠️ API Key Error: Please verify your Gemini API key in .env file.\n\nGet a free key at: https://aistudio.google.com/app/apikey"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return f"🌿 ⚠️ Model Error: Could not find a working Gemini model.\n\nTry:\n1. Update package: pip install --upgrade google-generativeai\n2. Enable API: https://console.cloud.google.com/apis/library (search 'Generative Language API')\n3. Get new API key: https://aistudio.google.com/app/apikey"
        elif "QUOTA" in error_msg.upper() or "LIMIT" in error_msg.upper():
            return f"🌿 ⚠️ Rate Limit: You've reached the API limit. Please wait a minute and try again."
        else:
            return f"🌿 ⚠️ Error: {error_msg}"


# ============================================
# MAIN RESPONSE GENERATOR
# ============================================

def generate_ai_response(message: str, model_type: str, file_path: Optional[str] = None, file_type: Optional[str] = None) -> str:
    """Main function to generate AI response based on model type"""
    # Process uploaded file if provided
    file_context = None
    if file_path and file_type:
        file_context = process_uploaded_file(file_path, file_type)
        if file_context:
            print(f"Extracted {len(file_context)} characters from uploaded file")
            if len(file_context) > 10000:
                file_context = file_context[:10000] + "\n...(content truncated)"
    
    # Generate response based on model type
    if model_type == 'medicinal':
        return get_medicinal_response(message, file_context)
    elif model_type == 'ayurvedic':
        return get_ayurvedic_response(message, file_context)
    else:
        return "Invalid model type. Please select either 'medicinal' or 'ayurvedic'."


# ============================================
# UTILITY FUNCTIONS
# ============================================

def test_gemini_connection() -> bool:
    """Test if Gemini API is working"""
    try:
        model_name = get_working_model()
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'API working' if you can read this.")
        print("✅ Gemini API test successful!")
        return True
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False


def list_available_models():
    """List all available Gemini models"""
    try:
        print("Available Gemini models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"  - {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")