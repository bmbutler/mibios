"""
Settings for production environment for system-wide installations
"""
from .settings import *

DEBUG = False
# TODO: secretly get SECRET_KEY
INSTALLED_APPS = [i for i in INSTALLED_APPS if i not in DEV_ONLY_APPS]
STATIC_ROOT = '/usr/share/mibios/static'
DATABASES['default']['NAME'] = '/var/lib/mibios/db.sqlite3'
# allowed hosts still need to be configured further:
ALLOWED_HOSTS = ['localhost']
