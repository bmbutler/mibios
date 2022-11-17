"""
settings for the mibios.omics app
"""
from os import environ

from mibios.ops.settings import *  # noqa:F403

# hook up the apps
INSTALLED_APPS.append('mibios.umrad.apps.AppConfig')  # noqa:F405
INSTALLED_APPS.append('mibios.omics.apps.AppConfig')  # noqa:F405

# defaults for those swappable models (they are strings <appname>.<model_name>)
OMICS_SAMPLE_MODEL = 'omics.Sample'
OMICS_DATASET_MODEL = 'omics.Dataset'

# register logging
LOGGING['loggers']['omics'] = LOGGING['loggers']['mibios']  # noqa:F405


def get_db_settings(db_dir='.', db_infix=''):
    """
    Call this to set DATABASE

    db_dir:  Directory where to store the DB, without trailing slash
    db_infix: optional infix to distinguish alternative DB
    """
    db_file = f'omics{db_infix}.sqlite3'
    db_mode = 'ro' if environ.get('MIBIOS_DB_RO') else 'rwc'

    return {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': f'file:{db_dir}/{db_file}?mode={db_mode}',
            'OPTIONS': {'uri': True},
        },
    }


DATABASES = get_db_settings()
