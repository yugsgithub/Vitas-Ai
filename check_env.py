"""
Quick Test Script - Check if .env and API key are working
SAVE AS: check_env.py (in project root, same level as manage.py)
RUN: python check_env.py
"""

import os
from pathlib import Path

print("=" * 60)
print("CHECKING .ENV FILE AND API KEY")
print("=" * 60)

# Check if .env file exists
env_file = Path('.env')
print("\n1. Checking if .env file exists...")
if env_file.exists():
    print("   ✅ .env file found!")
else:
    print("   ❌ .env file NOT found!")
    print("   Create .env file in project root with:")
    print("   GEMINI_API_KEY=your-key-here")
    exit(1)

# Check if python-dotenv is installed
print("\n2. Checking python-dotenv package...")
try:
    from dotenv import load_dotenv
    print("   ✅ python-dotenv is installed")
except ImportError:
    print("   ❌ python-dotenv NOT installed!")
    print("   Run: pip install python-dotenv")
    exit(1)

# Load .env file
print("\n3. Loading .env file...")
load_dotenv()
print("   ✅ .env file loaded")

# Check if API key is set
print("\n4. Checking for GEMINI_API_KEY...")
api_key = os.environ.get('GEMINI_API_KEY')

if api_key:
    print(f"   ✅ API key found!")
    print(f"   Key starts with: {api_key[:10]}...")
    print(f"   Key ends with: ...{api_key[-4:]}")
    print(f"   Key length: {len(api_key)} characters")
    
    # Check if key looks valid
    if api_key.startswith('AIza'):
        print("   ✅ Key format looks correct!")
    else:
        print("   ⚠️  WARNING: Key should start with 'AIza'")
        print("   Your key starts with:", api_key[:10])
    
    if api_key == 'your-api-key-here':
        print("   ❌ ERROR: You didn't replace the placeholder!")
        print("   Get your key from: https://aistudio.google.com/app/apikey")
        exit(1)
        
else:
    print("   ❌ NO API KEY FOUND!")
    print("   Add this to your .env file:")
    print("   GEMINI_API_KEY=your-actual-key-here")
    exit(1)

# Check google-generativeai package
print("\n5. Checking google-generativeai package...")
try:
    import google.generativeai as genai
    print("   ✅ google-generativeai is installed")
except ImportError:
    print("   ❌ google-generativeai NOT installed!")
    print("   Run: pip install google-generativeai")
    exit(1)

# Test API key with Gemini
print("\n6. Testing API key with Google Gemini...")
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Reply with just the word 'success' if this works")
    print("   ✅ API KEY WORKS!")
    print(f"   Response: {response.text[:50]}")
except Exception as e:
    print("   ❌ API key test FAILED!")
    print(f"   Error: {str(e)[:200]}")
    print("\n   Possible issues:")
    print("   1. API key is invalid - get a new one")
    print("   2. Gemini not available in your region")
    print("   3. Need to enable API in Google Cloud Console")
    exit(1)

# Check if settings.py has load_dotenv
print("\n7. Checking settings.py...")
settings_files = [
    'vitas_ai/settings.py',
    'config/settings.py',
    'settings.py'
]

found_settings = False
for settings_path in settings_files:
    if Path(settings_path).exists():
        found_settings = True
        print(f"   ✅ Found settings.py at: {settings_path}")
        
        with open(settings_path, 'r') as f:
            content = f.read()
            if 'load_dotenv' in content:
                print("   ✅ load_dotenv() found in settings.py")
            else:
                print("   ⚠️  WARNING: load_dotenv() NOT found in settings.py")
                print("   Add these lines at the TOP of settings.py:")
                print("   from dotenv import load_dotenv")
                print("   load_dotenv()")
        break

if not found_settings:
    print("   ⚠️  Could not find settings.py")

# All checks passed
print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED!")
print("=" * 60)
print("\nYour setup is correct!")
print("Restart Django and try again:")
print("  python manage.py runserver")
print("\nThen go to: http://127.0.0.1:8000/chat/")
print("=" * 60)