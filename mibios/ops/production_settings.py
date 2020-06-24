"""
Settings for production environment for system-wide installations
"""
from .settings import *

INSTALLED_APPS = [i for i in INSTALLED_APPS if i not in DEV_ONLY_APPS]
STATIC_ROOT = '/usr/share/mibios/static'
DATABASES['default']['NAME'] = '/var/lib/mibios/db.sqlite3'
# allowed hosts still need to be configured further:
ALLOWED_HOSTS = ['localhost']
AUTHENTICATION_BACKENDS = ['mibios.ops.utils.RemoteUserBackend']

# Security related:
DEBUG = False
# SECURE_HSTS_SECONDS = ??  (TODO)
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'
