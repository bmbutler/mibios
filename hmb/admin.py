from django.apps import apps
from django.contrib import admin


app_config = apps.get_app_config('hmb')


class AdminSite(admin.AdminSite):
    site_header = app_config.verbose_name + ' Administration'
    site_url = '/{}/'.format(app_config.name)


site = AdminSite(name=app_config.name)

for i in app_config.get_models():
    site.register(i)
