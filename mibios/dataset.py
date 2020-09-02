"""
Definitions for special datasets
"""


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
