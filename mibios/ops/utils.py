import logging

from django.conf import settings
from django.contrib.auth import backends


log = logging.getLogger(__name__)


class RemoteUserBackend(backends.RemoteUserBackend):
    create_unknown_user = False

    def clean_username(self, username):
        """
        Allow server admin to pretend to be a different user
        """
        try:
            real, assumed = settings.ASSUME_IDENTITY
        except Exception:
            pass
        else:
            if username == real:
                log.debug(
                    'Setting REMOTE_USER from {} to {}'.format(real, assumed)
                )
                username = assumed

        return username


class RemoteUserInjection:
    """
    Middleware to manipulate the REMOTE_USER variable on incoming requests

    Do not use this in production.  For development, we most commonly want to
    an authorized user to be logged in when testing views etc. and thus prevent
    the authentication framework to give us a login with the anonymous user,
    who shouldn't be able to access those views.  This middleware should run at
    the earliest point.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Code to be executed for each request before
        # the view (and later middleware) are called.
        """
        Set REMOTE_USER if configured
        """
        try:
            _, assumed = settings.ASSUME_IDENTITY
        except Exception:
            pass
        else:
            log.debug('Setting REMOTE_USER to {}'.format(assumed))
            request.META['REMOTE_USER'] = assumed
            print('MIDDLEWARE BORK:', type(request.META))
            print('MIDDLEWARE BORK:', request.META['REMOTE_USER'])

        return self.get_response(request)
