from django.apps import apps
from django.contrib import admin
from django.urls import reverse_lazy


app_config = apps.get_app_config('mibios')


class AdminSite(admin.AdminSite):
    site_header = app_config.verbose_name + ' Administration'
    site_url = reverse_lazy('top')


site = AdminSite(name=app_config.name)

for i in app_config.get_models():
    site.register(i)
