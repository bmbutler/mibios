import inspect
from importlib import import_module

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
