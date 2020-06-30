from django.conf import settings
from django.contrib.auth import backends


class RemoteUserBackend(backends.RemoteUserBackend):
    create_unknown_user = False

    def clean_username(self, username):
        """
        Allow server admin to pretend to be a different user
        """
        try:
            real, assumed = settings.ASSUME_USER_IDENTITY
        except Exception:
            pass
        else:
            if username == real:
                username = assumed

        return username
