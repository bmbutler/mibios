from django.apps import AppConfig

from mibios import __version__, get_registry


class AppConfig(AppConfig):
    name = 'mibios_seq'

    def ready(self):
        super().ready()
        registry = get_registry()
        registry.apps[self.name] = dict(
            version=__version__,
        )
