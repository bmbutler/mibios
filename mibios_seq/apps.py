from django.apps import AppConfig

from mibios import __version__, get_registry


class AppConfig(AppConfig):
    name = 'mibios_seq'

    def ready(self):
        get_registry().apps[self.name] = dict(
            version=__version__,
        )
