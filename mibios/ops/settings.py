"""
Django settings for mibios project.

Generated by 'django-admin startproject' using Django 2.2.12.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

from os import environ
from pathlib import Path

from . import get_secret_key


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = get_secret_key(Path('./secret.key'))

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# SECURE_HSTS_SECONDS = ??  (TODO)
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_SSL_REDIRECT = True  # set to False for runserver command
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'

ALLOWED_HOSTS = ['localhost']


# Application definition
INSTALLED_APPS = [
    'mibios.apps.AdminConfig',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'mibios.apps.MibiosConfig',
    'django_tables2',
    'rest_framework',
]


MIDDLEWARE = [
    'mibios.utils.StatsMiddleWare',
    'django.middleware.common.BrokenLinkEmailsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.RemoteUserMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

AUTHENTICATION_BACKENDS = ['mibios.ops.utils.RemoteUserBackend']

ROOT_URLCONF = 'mibios.ops.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'mibios.context_processors.base',
            ],
        },
    },
]

WSGI_APPLICATION = 'mibios.ops.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

# Name of snapshot directory
SNAPSHOT_DIR = Path('snapshots')

db_file = 'db.sqlite3'
db_mode = 'ro' if environ.get('MIBIOS_DB_RO') else 'rw'
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': f'file:{db_file}?mode={db_mode}',
        'OPTIONS': {'uri': True},
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

pwd_val_pkg = 'django.contrib.auth.password_validation'
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': pwd_val_pkg + '.UserAttributeSimilarityValidator',
    },
    {
        'NAME': pwd_val_pkg + '.MinimumLengthValidator',
    },
    {
        'NAME': pwd_val_pkg + '.CommonPasswordValidator',
    },
    {
        'NAME': pwd_val_pkg + '.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/Detroit'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'
DJANGO_TABLES2_TEMPLATE = "django_tables2/bootstrap.html"
STATIC_ROOT = 'static'
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} ({name}) '
                      'pid:{process:d}/{thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} ({name}) {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'formatter': 'verbose',
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': './debug.log',
        },
        'import_log_file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': './import.log',
            'formatter': 'verbose',
        },
        'mail_admins': {
            'level': 'ERROR',
            'class': 'django.utils.log.AdminEmailHandler',
        },
    },
    'loggers': {
        'django': {
            # log to file if DJANGO_LOG_LEVEL=DEBUG in env
            # FIXME: this does not work as we want to
            'handlers': ['file', 'mail_admins'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'mibios': {
            'handlers': ['console', 'file', 'mail_admins'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'dataimport': {
            'handlers': ['import_log_file'],
            'level': 'INFO',
        },
    },
}

REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.DjangoModelPermissions',
    ],
    'DEFAULT_PAGINATION_CLASS':
        'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
}

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        # 'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
        'TIMEOUT': 300,
    },
}

IGNORABLE_404_URLS = []
