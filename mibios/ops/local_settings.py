"""
Settings for production environment for use in manually installed setups, copy
this file to the web applications working directory as ./settings.py
"""
from .production_settings import *

DEBUG = False
STATIC_ROOT = './static'
DATABASES['default']['NAME'] = './db.sqlite3'
ALLOWED_HOSTS = ['localhost']
