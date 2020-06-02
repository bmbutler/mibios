from django.contrib import admin

from django.apps import apps


class MyAdminSite(admin.AdminSite):
    site_header = 'The Administration'


admin_site = MyAdminSite(name='myadmin')

for i in apps.get_app_config('hmb').get_models():
    admin_site.register(i)
