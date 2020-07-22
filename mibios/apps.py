from django import apps
from django.contrib.admin.apps import AdminConfig as UpstreamAdminConfig


class MibiosConfig(apps.AppConfig):
    name = 'mibios'
    verbose_name = 'Microbiome Data System'

    def ready(self):
        super().ready()
        from django.contrib import admin
        admin.site.register_all()


class AdminConfig(UpstreamAdminConfig):
    default_site = 'mibios.admin.AdminSite'
    # name = 'mibios.admin.site'
