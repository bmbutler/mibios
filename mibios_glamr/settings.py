"""
settings for the GLAMR app
"""
from mibios_omics.settings import *  # noqa:F403

LOGGING['loggers']['mibios_glamr'] = LOGGING['loggers']['mibios']  # noqa:F405

# add our app
INSTALLED_APPS.append('mibios_glamr.apps.AppConfig')  # noqa:F405

# override mibios' urls since glamr has it's own
ROOT_URLCONF = 'mibios_glamr.urls0'

# swappable models
OMICS_SAMPLE_MODEL = 'mibios_glamr.Sample'
OMICS_SAMPLE_GROUP_MODEL = 'mibios_glamr.Dataset'

DJANGO_TABLES2_TEMPLATE = 'django_tables2/bootstrap4.html'

# path to so file without .so suffix, e.g. './spellfix'
# Setting this enables search suggestions
SPELLFIX_EXT_PATH = None
