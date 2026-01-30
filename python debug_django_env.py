"""
Debug Django Settings and Environment Variables
SAVE AS: debug_django_env.py (in project root)
RUN: python debug_django_env.py
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("DEBUGGING DJANGO ENVIRONMENT VARIABLES")
print("=" * 70)

# Add project to path
project_path = Path(__file__).resolve().parent
sys.path.insert(0, str(project_path))

print("\n1. CHECKING .ENV FILE:")
print("-" * 70)
env_file = Path('.env')
if env_file.exists():
    print("   ✅ .env file exists")
    print(f"   Location: {env_file.absolute()}")
    
    # Read and show content (masked)
    with open(env_file, 'r') as f:
        content = f.read().strip()
        if content:
            print("   ✅ .env file has content")
            lines = content.split('\n')
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    if 'KEY' in key or 'SECRET' in key:
                        masked_value = value[:10] + '...' + value[-4:] if len(value) > 14 else '***'
                        print(f"   {key} = {masked_value}")
                    else:
                        print(f"   {line}")
        else:
            print("   ⚠️  .env file is EMPTY!")
else:
    print("   ❌ .env file NOT found!")
    print("   Create .env file in project root")

print("\n2. CHECKING PYTHON-DOTENV:")
print("-" * 70)
try:
    from dotenv import load_dotenv
    print("   ✅ python-dotenv installed")
    
    # Load .env
    loaded = load_dotenv()
    if loaded:
        print("   ✅ .env file loaded successfully")
    else:
        print("   ⚠️  .env file not loaded (might not exist)")
except ImportError:
    print("   ❌ python-dotenv NOT installed")
    print("   Run: pip install python-dotenv")

print("\n3. CHECKING ENVIRONMENT VARIABLES (BEFORE DJANGO):")
print("-" * 70)
api_key_before = os.environ.get('GEMINI_API_KEY')
if api_key_before:
    print(f"   ✅ GEMINI_API_KEY found: {api_key_before[:10]}...{api_key_before[-4:]}")
else:
    print("   ❌ GEMINI_API_KEY not found in environment")

print("\n4. CHECKING DJANGO SETTINGS:")
print("-" * 70)
try:
    # Setup Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vitas_project.settings')
    
    import django
    django.setup()
    
    print("   ✅ Django setup successful")
    
    from django.conf import settings
    
    # Check if settings loaded
    print("   ✅ Django settings loaded")
    
    # Try to get API key through Django settings
    api_key_django = os.environ.get('GEMINI_API_KEY')
    if api_key_django:
        print(f"   ✅ GEMINI_API_KEY accessible: {api_key_django[:10]}...{api_key_django[-4:]}")
    else:
        print("   ❌ GEMINI_API_KEY not accessible from Django")
        
except Exception as e:
    print(f"   ⚠️  Django setup issue: {str(e)[:100]}")
    print("   Trying alternative settings module name...")
    
    # Try alternative names
    for settings_module in ['config.settings', 'settings', 'vitas_ai.settings']:
        try:
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)
            import django
            django.setup()
            print(f"   ✅ Django setup successful with: {settings_module}")
            break
        except:
            continue

print("\n5. CHECKING SETTINGS.PY FILE:")
print("-" * 70)

# Find settings.py
settings_files = [
    'vitas_project/settings.py',
    'vitas_ai/settings.py',
    'config/settings.py',
    'settings.py'
]

found_settings = False
for settings_path in settings_files:
    if Path(settings_path).exists():
        found_settings = True
        print(f"   ✅ Found settings.py at: {settings_path}")
        
        # Read first 30 lines to check for load_dotenv
        with open(settings_path, 'r') as f:
            lines = f.readlines()[:30]
            content = ''.join(lines)
            
            if 'load_dotenv' in content:
                print("   ✅ load_dotenv() found in settings.py")
                
                # Check if it's before imports
                dotenv_line = None
                for i, line in enumerate(lines):
                    if 'load_dotenv' in line:
                        dotenv_line = i + 1
                        break
                
                if dotenv_line:
                    print(f"   load_dotenv() is at line {dotenv_line}")
                    if dotenv_line <= 10:
                        print("   ✅ load_dotenv() is near the top (good)")
                    else:
                        print("   ⚠️  load_dotenv() should be at the very top")
            else:
                print("   ❌ load_dotenv() NOT found in settings.py")
                print("\n   ADD THESE LINES AT THE TOP OF settings.py:")
                print("   " + "-" * 50)
                print("   from dotenv import load_dotenv")
                print("   import os")
                print("   load_dotenv()")
                print("   " + "-" * 50)
        break

if not found_settings:
    print("   ⚠️  Could not find settings.py file")

print("\n6. TESTING GEMINI API WITH CURRENT ENVIRONMENT:")
print("-" * 70)

api_key_test = os.environ.get('GEMINI_API_KEY')
if api_key_test:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key_test)
        
        # Try to find working model
        print("   Testing model access...")
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro'
        ]
        
        working = False
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("say 'test'")
                print(f"   ✅ API KEY WORKS with model: {model_name}")
                working = True
                break
            except:
                continue
        
        if not working:
            print("   ❌ API key configured but no working model found")
            print("   This might be a region or API enablement issue")
            
    except Exception as e:
        print(f"   ❌ API test failed: {str(e)[:100]}")
else:
    print("   ❌ No API key available for testing")

print("\n" + "=" * 70)
print("DIAGNOSIS & SOLUTION")
print("=" * 70)

# Provide solution based on findings
if not env_file.exists():
    print("\n❌ ISSUE: .env file missing")
    print("\nSOLUTION:")
    print("1. Create .env file in project root")
    print("2. Add: GEMINI_API_KEY=your-key-here")
elif not api_key_before:
    print("\n❌ ISSUE: .env file exists but not loading")
    print("\nSOLUTION:")
    print("1. Check .env file has: GEMINI_API_KEY=your-key")
    print("2. No quotes, no spaces around =")
    print("3. Make sure load_dotenv() is at top of settings.py")
elif not found_settings or 'load_dotenv' not in content:
    print("\n❌ ISSUE: Django settings.py not loading .env file")
    print("\nSOLUTION:")
    print("Add these lines at the VERY TOP of settings.py:")
    print("\nfrom dotenv import load_dotenv")
    print("import os")
    print("load_dotenv()")
else:
    print("\n✅ Environment variables are set up correctly!")
    print("\nIf still getting errors, the issue might be:")
    print("1. Django server needs restart")
    print("2. Model availability in your region")
    print("3. API needs to be enabled in Google Cloud Console")

print("\n" + "=" * 70)
