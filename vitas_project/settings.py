"""
ADD THESE LINES AT THE VERY TOP OF YOUR settings.py
(Before any other imports or code)

This ensures .env file is loaded before Django starts
"""

# ============================================
# STEP 1: ADD THIS AT THE VERY TOP
# ============================================

from pathlib import Path
import os
from dotenv import load_dotenv

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from project root
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# Debug: Print if .env loaded (remove after testing)
if os.path.exists(env_path):
    print(f"✅ .env file found at: {env_path}")
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        print(f"✅ GEMINI_API_KEY loaded: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("❌ GEMINI_API_KEY not found in .env file")
else:
    print(f"❌ .env file not found at: {env_path}")

# ============================================
# AFTER THIS, ADD YOUR REGULAR SETTINGS
# ============================================

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'your-secret-key-here'

# ... rest of your settings ...


# ============================================
# FULL EXAMPLE OF TOP OF settings.py:
# ============================================

"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# Debug print (remove after confirming it works)
if os.path.exists(env_path):
    print(f"✅ .env file found at: {env_path}")
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        print(f"✅ GEMINI_API_KEY loaded: {api_key[:10]}...{api_key[-4:]}")
else:
    print(f"❌ .env file not found at: {env_path}")

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-...'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    # ... rest of your apps
]

# ... rest of your settings
"""
from dotenv import load_dotenv
import os
load_dotenv()
"""
Django settings for vitas_project project.
COPY THIS ENTIRE FILE INTO: vitas_project/settings.py
"""


from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-vitas-ai-2024-change-this-in-production'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    
    # Third party apps
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    
    # Local apps
    'chatbot',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'allauth.account.middleware.AccountMiddleware',
]

ROOT_URLCONF = 'vitas_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            BASE_DIR / 'templates',
            BASE_DIR / 'chatbot' / 'templates',
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.static',  # ADD THIS
            ],
        },
    },
]

WSGI_APPLICATION = 'vitas_project.wsgi.application'


# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
import os

STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Django Allauth Configuration
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

SITE_ID = 1

# Allauth settings
ACCOUNT_AUTHENTICATION_METHOD = 'email'
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_UNIQUE_EMAIL = True
ACCOUNT_USERNAME_REQUIRED = True
ACCOUNT_EMAIL_VERIFICATION = 'none'  # Set to 'mandatory' in production
LOGIN_REDIRECT_URL = '/chat/'
ACCOUNT_LOGOUT_REDIRECT_URL = '/'
ACCOUNT_SIGNUP_REDIRECT_URL = '/chat/'

# Social Auth Settings
SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'SCOPE': [
            'profile',
            'email',
        ],
        'AUTH_PARAMS': {
            'access_type': 'online',
        },
        'APP': {
            'client_id': 'YOUR_GOOGLE_CLIENT_ID',  # Replace with your Google Client ID
            'secret': 'YOUR_GOOGLE_CLIENT_SECRET',  # Replace with your Google Client Secret
            'key': ''
        }
    }
}

# Login URL
LOGIN_URL = '/accounts/login/'

# Session settings
SESSION_COOKIE_AGE = 86400  # 1 day
SESSION_SAVE_EVERY_REQUEST = True