"""
Settings for production environment
"""
from .settings import *

DEBUG = False
# TODO: secretly get SECRET_KEY
INSTALLED_APPS = [i for i in INSTALLED_APPS if i not in DEV_ONLY_APPS]
STATIC_ROOT = '/use/share/hmb/static'
DATABASES['default']['NAME'] = '/var/lib/hmb/db.sqlite3'
