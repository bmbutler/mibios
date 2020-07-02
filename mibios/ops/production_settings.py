"""
Settings for production environment for system-wide installations
"""
from .settings import *

STATIC_ROOT = '/usr/share/mibios/static'
DATABASES['default']['NAME'] = '/var/lib/mibios/db.sqlite3'

# Security related:
DEBUG = False
# SECURE_HSTS_SECONDS = ??  (TODO)
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'
