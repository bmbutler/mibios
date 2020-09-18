from importlib import import_module
import inspect

from django import apps
from django.contrib.admin.apps import AdminConfig as UpstreamAdminConfig
from django.utils.module_loading import import_string


class Registry():

    name = None
    verbose_name = None
    model_class = None
    dataset_class = None
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        self.model_class = import_string('mibios.models.Model')
        self.dataset_class = import_string('mibios.dataset.Dataset')

        # datasets are maps from a name to a Dataset subclass singleton object
        self.datasets = {}
        # models are maps from name to Model subclass
        self.models = {}
        # apps: a map from app label to the app's AppConfig instance
        self.apps = {}
        # plugins: maps model name to plugin class
        self.table_view_plugins = {}

    def get_models(self, app=None):
        return [
            i for i in self.models.values()
            if app is None or app == i._meta.app_label
        ]

    def get_model_names(self, app=None):
        return [
            k for k, v in self.models.items()
            if app is None or app == v._meta.app_label
        ]

    def get_datasets(self, app=None):
        return [
            i for i in self.datasets.values()
            if app is None or app == i.app_label
        ]

    def get_dataset_names(self, app=None):
        return [
            k for k, v in self.datasets.items()
            if app is None or app == v.app_label
        ]

    def get_names(self, app=None):
        return self.get_model_names(app=app) + self.get_dataset_names(app=app)

    def __setitem__(self, key, value):
        if isinstance(value, self.dataset_class):
            self.datasets[key] = value
        elif issubclass(value, self.model_class):
            self.models[key] = value
        else:
            raise ValueError('can only register datasets or models')

    def __getitem__(self, key):
        try:
            return self.datasets[key]
        except KeyError:
            return self.models[key]

    def add(self, item):
        if isinstance(item, self.dataset_class):
            self[item.name] = item
        elif issubclass(item, self.model_class):
            self[item._meta.model_name] = item
        else:
            raise ValueError('can only register datasets or models')

    def add_dataset_module(self, module_path, app_label):
        """
        Populate the registry with dataset from given module dotted path
        """
        dataset_module = import_module(module_path)
        for _, klass in inspect.getmembers(dataset_module, inspect.isclass):
            if issubclass(klass, self.dataset_class):
                if klass is self.dataset_class:
                    continue
                # add singleton instance
                obj = klass()
                obj.app_label = app_label
                self[klass.name] = obj

    def add_models(self):
        """
        Autoregister all defined Model subclasses
        """
        for i in self.model_class.__subclasses__():
            self.add(i)

    def add_table_view_plugins(self, module_path):
        views = import_module(module_path)
        parent = import_string('mibios.views.TableViewPlugin')
        for _, klass in inspect.getmembers(views, inspect.isclass):
            if not issubclass(klass, parent):
                continue

            if klass is parent:
                continue

            self.table_view_plugins[klass.model_class._meta.model_name] = klass


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
