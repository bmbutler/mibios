"""
Settings for production environment for use in manually installed setups, copy
this file to the web applications working directory as ./settings.py
"""
from mibios.ops.settings import *

DEBUG = False
# TODO: secretly get SECRET_KEY
INSTALLED_APPS = [i for i in INSTALLED_APPS if i not in DEV_ONLY_APPS]
STATIC_ROOT = './static'
DATABASES['default']['NAME'] = './db.sqlite3'
ALLOWED_HOSTS = ['localhost']
