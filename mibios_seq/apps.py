from django.apps import AppConfig

from mibios import __version__


class AppConfig(AppConfig):
    name = 'mibios_seq'
    verbose_name = 'sequencing data'
    version = __version__
