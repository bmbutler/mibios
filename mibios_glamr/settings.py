"""
settings for the GLAMR app
"""
from mibios_omics.settings import *  # noqa:F403

# add our app
INSTALLED_APPS.append('mibios_glamr.apps.AppConfig')  # noqa:F405

# override mibios' urls since glamr has it's own
ROOT_URLCONF = 'mibios_glamr.urls0'

# swappable models
SAMPLE_GROUP_MODEL = 'mibios_glamr.Dataset'

DJANGO_TABLES2_TEMPLATE = 'django_tables2/bootstrap4.html'
