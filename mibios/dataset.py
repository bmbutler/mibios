"""
Definitions for special datasets
"""
from .models import Model


class UserDataError(Exception):
    pass


class Dataset():
    name = None
    model = None
    fields = []
    filter = {}
    excludes = []
    blanks = ['']
    manager = 'published'

    # instance container for subclass singletons
    __instance = {}

    def __new__(cls):
        """
        Create per-subclass singleton instance
        """
        if cls.__name__ not in cls.__instance:
            cls.__instance[cls.__name__] = super().__new__(cls)
        return cls.__instance[cls.__name__]

    def __init__(self):
        if not self.fields:
            # default to all normal fields of model
            self.fields = self.model.get_fields().names
            self.fields = [(i,) for i in self.fields]


class Registry():

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
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
        if isinstance(value, Dataset):
            self.datasets[key] = value
        elif issubclass(value, Model):
            self.models[key] = value
        else:
            raise ValueError('can only register datasets or models')

    def __getitem__(self, key):
        try:
            return self.datasets[key]
        except KeyError:
            return self.models[key]

    def add(self, item):
        if isinstance(item, Dataset):
            self[item.name] = item
        elif issubclass(item, Model):
            self[item._meta.model_name] = item
        else:
            raise ValueError('can only register datasets or models')

    def add_datasets(self):
        """
        Autoregister representatives for all defined Dataset subclasses
        """
        for i in Dataset.__subclasses__():
            self[i.name] = i()

    def add_models(self):
        """
        Autoregister all defined Model subclasses
        """
        for i in Model.__subclasses__():
            self.add(i)


registry = Registry()
