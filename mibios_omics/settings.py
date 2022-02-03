"""
settings for the mibios_omics app
"""
from mibios.ops.settings import *  # noqa:F403

# hook up the apps
INSTALLED_APPS.append('mibios_umrad.apps.AppConfig')  # noqa:F405
INSTALLED_APPS.append('mibios_omics.apps.AppConfig')  # noqa:F405

# defaults for those swappable models (they are strings <appname>.<model_name>)
OMICS_SAMPLE_MODEL = 'mibios_omics.Sample'
OMICS_SAMPLE_GROUP_MODEL = 'mibios_omics.SampleGroup'

# register logging
LOGGING['loggers']['mibios_omics'] = LOGGING['loggers']['mibios']  # noqa:F405
