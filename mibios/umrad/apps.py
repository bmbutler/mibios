import django.apps

from mibios import __version__


class AppConfig(django.apps.AppConfig):
    name = 'mibios.umrad'
    verbose_name = 'UMRAD'
    version = __version__
