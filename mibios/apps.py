from importlib import import_module
import inspect

from django import apps
from django.contrib.admin.apps import AdminConfig as UpstreamAdminConfig
from django.utils.module_loading import import_string


class Registry():

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
        # apps: a map from package name to dict of meta data
        self.apps = {}

    def get_models(self):
        return list(self.models.values())

    def get_model_names(self):
        return list(self.models.keys())

    def get_datasets(self):
        return list(self.datasets.values())

    def get_dataset_names(self):
        return list(self.datasets.keys())

    def get_names(self):
        return self.get_model_names() + self.get_dataset_names()

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

    def add_dataset_module(self, module_path):
        """
        Populate the registry with dataset from given module dotted path

        This should be called by implementing apps from their AppConfig.ready()
        """
        dataset_module = import_module(module_path)
        for _, klass in inspect.getmembers(dataset_module, inspect.isclass):
            if issubclass(klass, self.dataset_class):
                if klass is self.dataset_class:
                    continue
                # add singleton instance
                self[klass.name] = klass()

    def add_models(self):
        """
        Autoregister all defined Model subclasses
        """
        for i in self.model_class.__subclasses__():
            self.add(i)


class MibiosConfig(apps.AppConfig):
    name = 'mibios'
    verbose_name = 'Microbiome Data System'

    def ready(self):
        super().ready()

        # set up registry
        Model = import_string(self.name + '.models.Model')
        mibios = import_module('mibios')
        setattr(mibios, '_registry', Registry())
        registry = mibios.get_registry()

        # register models here, since django has found them already
        for i in apps.apps.get_models():
            if issubclass(i, Model):
                if hasattr(i, 'get_child_info') and i.get_child_info():
                    # model has children, skip
                    continue
                registry.add(i)

        # admin setup below:
        admin = import_string('django.contrib.admin')
        admin.site.register_all()


class AdminConfig(UpstreamAdminConfig):
    default_site = 'mibios.admin.AdminSite'
    # name = 'mibios.admin.site'
