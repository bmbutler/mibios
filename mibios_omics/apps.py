from django.apps import AppConfig as _AppConfig


from mibios import __version__

class AppConfig(_AppConfig):
    name = 'mibios_omics'
    verbose_name = 'omics'
    version = __version__
