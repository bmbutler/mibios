__version__ = ''  # to be overwritten by`setup.py build`


# constants used in TableView and forms, e.g. keyword recognized in the URL
# querystring, declared here to avoid circular imports
QUERY_FILTER = 'filter'
QUERY_EXCLUDE = 'exclude'
QUERY_NEGATE = 'inverse'
QUERY_SHOW = 'show'
QUERY_FORMAT = 'format'
QUERY_AVG_BY = 'avg-by'
QUERY_COUNT = 'count'
QUERY_SEARCH = 'search'
QUERY_Q = 'q'


_registry = None


def get_registry():
    if _registry is None:
        raise RuntimeError(
            "Registry is not yet set up.  It's only available after Django's "
            "apps are set up, i.e. after mibios' app config's ready() has "
            "returned."
        )
    return _registry
