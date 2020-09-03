from django.apps import AppConfig

from mibios import __version__, get_registry


class AppConfig(AppConfig):
    name = 'mibios_seq'
    version = __version__

    def ready(self):
        super().ready()
        registry = get_registry()
        registry.apps[self.name] = self
