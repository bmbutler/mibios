"""The project package for mibios"""
import os
import sys


VAR = 'DJANGO_SETTINGS_MODULE'

def manage(settings=None):
    """
    The original manage.py

    This is also the entry point for the manage script when installed vie
    setuptools.  In this case no argument is supplied and the prodcution
    settings are applied by default.  The usual manage.py script shoudl specify
    the development settings.
    """
    if settings is None:
        if VAR not in os.environ:
            # set a default
            settings = 'mibios.ops.production_settings'

    os.environ.setdefault(VAR, settings)
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
