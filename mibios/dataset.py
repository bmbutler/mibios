from inspect import getdoc, getmembers, ismethod
"""
Definitions for special datasets
"""
PARSE_BLANK = 'parse_blank'


class UserDataError(Exception):
    pass


class Dataset():
    name = None
    model = None
    app_label = None
    fields = []
    filter = {}
    excludes = []
    blanks = ['']
    manager = 'curated'

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
        # sentinel return value for parse_FOO methods to indicate to skip field
        self.IGNORE_THIS_FIELD = object()
        if not self.fields:
            # default to all normal fields of model
            self.fields = self.model.get_fields().names
            self.fields = [(i,) for i in self.fields]

    def get_doc(self):
        """
        Collect docstrings for dataset class and parse_FOO methods
        """
        PARSE_METH_PREFIX = 'parse_'
        doc = getdoc(self) or ''
        methdocs = [
            (name[len(PARSE_METH_PREFIX):], getdoc(meth))
            for name, meth
            in getmembers(self, predicate=ismethod)
            if name.startswith(PARSE_METH_PREFIX)
        ]
        if methdocs:
            if doc:
                doc += '\n\n'
            doc += '\n--- Column-specific parser info ---\n'
            for col, txt in methdocs:
                doc += f'\nColumn "{col}":\n{txt}\n'

        return doc
