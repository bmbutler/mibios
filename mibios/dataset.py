"""
Definitions for special datasets
"""
from pathlib import Path
import re

from django.apps import apps


class UserDataError(Exception):
    pass


class Dataset():
    name = None
    model = None
    fields = []
    filter = {}
    excludes = []
    missing_data = []
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
            self.fields = apps.get_app_config('mibios').get_model(self.model) \
                    .get_fields().names
            self.fields = [(i,) for i in self.fields]


DATASET = {}
for i in Dataset.__subclasses__():
    DATASET[i.name] = i()
