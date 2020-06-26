from django.contrib.auth import backends


class RemoteUserBackend(backends.RemoteUserBackend):
    create_unknown_user = False

