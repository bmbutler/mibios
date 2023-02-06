"""
Django settings specific for glarm test site on alpena
"""
from pathlib import Path
from mibios.glamr.settings import *


# Set to True for development but never in production deployment
DEBUG = False

# Set this to False when running the runserver command on localhost
SECURE_SSL_REDIRECT = False

# Add additional apps here:
INSTALLED_APPS.append('django_extensions')

# User switch magic: needs the remote user injection middleware and set
# ASSUME_IDENTIY = ('alice', 'bob') so when user bob logs in through the web
# server the middleware will make it look as if alice is authenticated.  In
# development, e.g. when using the shell or runserver commands let
# ASSUME_IDENTITY = ('', 'bob') assume bob's identity.
#
#MIDDLEWARE = ['mibios.ops.utils.RemoteUserInjection'] + MIDDLEWARE
#ASSUME_IDENTITY = ('', 'heinro')

# List of contacts for site adminitrators
ADMINS = [("Robert", "heinro@umich.edu")]

# For production, set STATIC_ROOT to the directory containing static files,
# relative to your instance's base directory
# STATIC_ROOT = 'static'

# URL for static files
STATIC_URL = '/glamr/static/'

# Direcorty relative to the base where download files get stored
MEDIA_ROOT = '/glamr/media/'

# URL path for downloads
MEDIA_URL = '/glamr/media/'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'glamr',
        'USER': 'glamr_django',
        'HOST': 'database',
        'PORT': '5432',
    },
}

# Allowed host settings:
ALLOWED_HOSTS.append('127.0.0.1')
ALLOWED_HOSTS.append('webapp')
ALLOWED_HOSTS.append('vondamm.earth.lsa.umich.edu')

# Uncomment this do disable caching, for testing/debugging only
# CACHES['default']['BACKEND'] = 'django.core.cache.backends.dummy.DummyCache'

SITE_NAME = 'GLAMR'
SITE_NAME_VERBOSE = 'GLAMR DB prototype'

SCHEMA_PLOT_APPS = ['mibios_omics']

STATICFILES_DIRS = ['static_var']
FORCE_SCRIPT_NAME = '/glamr'
