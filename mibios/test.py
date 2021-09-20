from django.conf import settings
from django.contrib.auth.models import User
from django.core.exceptions import ImproperlyConfigured
from django.test import RequestFactory
from django.urls import resolve


def get_view(url, user=None):
    """
    Return view instance corresponding to given URL

    Convenience helper for testing and debugging.  Returns the view about as it
    is after setup() but before dispatch() is called.
    """

    if user is None:
        user = User.objects.first()
    elif isinstance(user, str):
        user = User.objects.get(username=user)

    try:
        server_name = settings.ALLOWED_HOSTS[0]
    except IndexError as e:
        raise ImproperlyConfigured('ALLOWED_HOST needs to be populated') from e

    req = RequestFactory(SERVER_NAME=server_name).get(url)
    req.user = user

    m = resolve(req.path_info)

    # roughly do what View.as_view() puts into the view function:
    view = m.func.view_class(**m.func.view_initkwargs)
    view.setup(req, *m.args, **m.kwargs)

    return view
