"""
Settings for the development environment
"""
from mibios.ops.settings import *

DEBUG = True
INSTALLED_APPS.append('django_extensions')
MIDDLEWARE = ['mibios.ops.utils.RemoteUserInjection'] + MIDDLEWARE
ASSUME_IDENTITY = ('', 'heinro')
