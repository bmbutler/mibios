__version__ = ''  # to be overwritten by`setup.py build`

if __version__ == '':
    # get a version for development settings, in production the __version__
    # variable should have a hard-coded value.  For robustness all exceptions
    # are caught and __version__ is not set in case of errors.
    # get something like v0.9-339-g68a6ee6 via git-describe
    cmd = 'git describe --tag --match v* --always'
    try:
        import subprocess
        p = subprocess.run(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        __version__ = p.stdout.decode().split(maxsplit=1)[0]
    except Exception:
        pass

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
