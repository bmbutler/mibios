"""
Settings specific to mibios_seq module

This should be imported into the settings module at deployment
"""
from mibios.ops.settings import *

INSTALLED_APPS.append('mibios_seq.apps.AppConfig')
LOGGING['loggers']['mibios_seq'] = LOGGING['loggers']['mibios']
