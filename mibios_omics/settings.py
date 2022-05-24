"""
settings for the mibios_omics app
"""
from os import environ

from mibios.ops.settings import *  # noqa:F403

# hook up the apps
INSTALLED_APPS.append('mibios_umrad.apps.AppConfig')  # noqa:F405
INSTALLED_APPS.append('mibios_omics.apps.AppConfig')  # noqa:F405

# defaults for those swappable models (they are strings <appname>.<model_name>)
OMICS_SAMPLE_MODEL = 'mibios_omics.Sample'
OMICS_SAMPLE_GROUP_MODEL = 'mibios_omics.SampleGroup'

# register logging
LOGGING['loggers']['mibios_omics'] = LOGGING['loggers']['mibios']  # noqa:F405


def get_db_settings(db_dir='.', db_infix=''):
    """
    Call this to set DATABASES

    db_dir:  Directory where to store the DBs, without trailing slash
    db_infix: optional infix to distinguish alternative DBs
    """
    ops_db_file = f'ops{db_infix}.sqlite3'
    omics_db_file = f'omics{db_infix}.sqlite3'
    omics_db_mode = 'ro' if environ.get('MIBIOS_DB_RO') else 'rwc'

    return {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': f'file:{db_dir}/{ops_db_file}?mode=rwc',
            'OPTIONS': {'uri': True},
        },
        'omics': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': f'file:{db_dir}/{omics_db_file}?mode={omics_db_mode}',
            'OPTIONS': {'uri': True},
        }
    }


DATABASES = get_db_settings()

DATABASE_ROUTERS = ['mibios_omics.DBRouter']
