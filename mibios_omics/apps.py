from django.apps import AppConfig

from mibios import __version__


class AppConfig(AppConfig):
    name = 'mibios_omics'
    verbose_name = 'mibios omics data'
    version = __version__
