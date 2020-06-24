"""The project package for mibios"""
import os
import sys

from django.core.management.utils import get_random_secret_key


VAR = 'DJANGO_SETTINGS_MODULE'
DEFAULT_SETTINGS = 'mibios.ops.settings'
LOCAL_SETTINGS = 'settings'


def manage(settings=None):
    """
    The original manage.py

    This is also the entry point for the manage script when installed vie
    setuptools.  In this case no argument is supplied and the prodcution
    settings are applied by default.  The usual manage.py script shoudl specify
    the development settings.
    """
    if settings is None:
        if VAR in os.environ:
            # already in env, 2nd priority
            pass
        else:
            if os.path.exists(LOCAL_SETTINGS + '.py'):
                # for local/manual deployment
                settings = LOCAL_SETTINGS
                sys.path = [''] + sys.path
            else:
                # last resort but usually for development
                settings = DEFAULT_SETTINGS
    else:
        # passed as argument, has top priority
        pass

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


def get_secret_key(keyfile):
    """
    Read key from given file

    If file does not exist, generate a key randomly and store it in the file
    first.
    """
    if not keyfile.exists():
        keyfile.touch(mode=0o600, exist_ok=False)
        keyfile.write_text(get_random_secret_key())
        keyfile.chmod(0o400)

    return keyfile.read_text()
