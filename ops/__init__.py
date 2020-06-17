"""The project package for hmb"""
import os
import sys


def manage():
    """
    The original manage.py
    """
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ops.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
