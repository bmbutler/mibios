"""
url declarations for the mibios_glamr app
"""
# FIXME: this module is, for now, called url0.  If we were to call it the usual
# url, then the mibios.url will try to include it.  This, when combined with
# setting ROOT_URLCONF to it, as we would do when running as the GLAMR webapp,
# will result in a loop.  Calling the module url0 avoids getting into that loop
# in the first place.  We should revise if the automatic include in mibios.url
# still makes sense.
from django.urls import include, path

from mibios import urls as mibios_urls
from . import views


urlpatterns = [
    path('', views.DemoFrontPageView.as_view(), name='demo_frontpage'),
    path('tables/', include(mibios_urls))
]
