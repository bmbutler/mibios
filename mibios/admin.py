from django.apps import apps
from django.contrib import admin
from django.urls import reverse_lazy


app_config = apps.get_app_config('mibios')


class AdminSite(admin.AdminSite):
    site_header = app_config.verbose_name + ' Administration'
    site_url = reverse_lazy('top')


site = AdminSite(name=app_config.name)


class ModelAdmin(admin.ModelAdmin):
    exclude = ['history']
    actions = None

    def save_model(self, request, obj, form, change):
        obj.add_change_record(user=request.user)
        super().save_model(request, obj, form, change)

    def delete_model(self, request, obj):
        obj.add_change_record(user=request.user, is_deleted=True)
        super().delete_model(request, obj)


for i in app_config.get_models():
    if i._meta.model_name == 'changerecord':
        continue
    site.register(i, ModelAdmin)
