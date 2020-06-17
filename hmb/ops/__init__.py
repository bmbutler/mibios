"""The project package for hmb"""
import os
import sys


def manage(settings='hmb.ops.production_settings'):
    """
    The original manage.py

    This is also the entry point for the manage script when installed vie
    setuptools.  In this case no argument is supplied and the prodcution
    settings are applied by default.  The usual manage.py script shoudl specify
    the development settings.
    """
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings)
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
