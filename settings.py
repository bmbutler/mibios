"""
Django settings specific for glarm test site on alpena
"""
from mibios.ops.settings import *


# hook up the app
INSTALLED_APPS.append('mibios_omics.apps.AppConfig')
INSTALLED_APPS.append('django_extensions')  # dev only

# register logging
LOGGING['loggers']['mibios_omics'] = LOGGING['loggers']['mibios']

SITE_NAME = 'GLAMR'
SITE_NAME_VERBOSE = 'GLAMR DB test site'

# allow http via local server
SECURE_SSL_REDIRECT = False

DEBUG = True
ALLOWED_HOSTS.append('127.0.0.1')
MIDDLEWARE = ['mibios.ops.utils.RemoteUserInjection'] + MIDDLEWARE
ASSUME_IDENTITY = ('', 'heinro')
