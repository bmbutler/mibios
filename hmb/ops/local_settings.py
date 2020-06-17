"""
Settings for production environment
"""
from .settings import *

DEBUG = False
# TODO: secretly get SECRET_KEY
INSTALLED_APPS = [i for i in INSTALLED_APPS if i not in DEV_ONLY_APPS]
STATIC_ROOT = './static'
DATABASES['default']['NAME'] = './db.sqlite3'
