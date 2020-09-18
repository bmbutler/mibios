from importlib import import_module

from django import apps
from django.contrib.admin.apps import AdminConfig as UpstreamAdminConfig
from django.utils.module_loading import import_string

from .registry import Registry


class MibiosConfig(apps.AppConfig):
    name = 'mibios'
    verbose_name = 'Microbiome Data System'

    def ready(self):
        super().ready()

        # set up registry
        registry = Registry()
        registry.name = self.name
        registry.verbose_name = self.verbose_name
        import_module('mibios')._registry = registry

        # register models here, since django has found them already
        Model = import_string(self.name + '.models.Model')
        for i in apps.apps.get_models():
            if issubclass(i, Model):
                if hasattr(i, 'get_child_info') and i.get_child_info():
                    # model has children, skip
                    continue
                registry.add(i)

                # register all apps with mibios.Models
                app = i._meta.app_label
                if app not in registry.apps:
                    registry.apps[app] = apps.apps.get_app_config(app)

        # register datasets
        for app_conf in registry.apps.values():
            try:
                registry.add_dataset_module(
                    app_conf.name + '.dataset',
                    app_conf.label,
                )
            except ImportError:
                pass

        # register table view plugins
        for app_conf in registry.apps.values():
            try:
                registry.add_table_view_plugins(app_conf.name + '.views')
            except ImportError:
                pass

        # admin setup below:
        admin = import_string('django.contrib.admin')
        admin.site.register_all()


class AdminConfig(UpstreamAdminConfig):
    default_site = 'mibios.admin.AdminSite'
    # name = 'mibios.admin.site'
