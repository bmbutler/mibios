from django.apps import apps
from django.contrib import admin
from django.db.transaction import atomic
from django.urls import reverse_lazy

from .dataset import registry
from .views import HistoryView


app_config = apps.get_app_config('mibios')


class AdminSite(admin.AdminSite):
    site_header = 'Administration'
    index_title = 'Data Curation'
    site_url = reverse_lazy('top')

    def register_all(self):
        """
        Register all the admins

        To be called from AppConfig.ready().  Normally, registration is done
        when this module is imported, but is seems even when following the
        documentation at
        https://docs.djangoproject.com/en/2.2/ref/contrib/admin/#overriding-the-default-admin-site
        we lose all registrations, possibly because site gets re-instantiated
        later, maybe has to do with module auto-discovery.
        """
        for i in registry.get_models():
            self.register(i, ModelAdmin)

    def get_app_list(self, request):
        """
        Get the app list but change order a bit

        The auth app shall go last
        """
        auth_admin = None
        app_list = []
        for i in super().get_app_list(request):
            if i['app_label'] == 'auth':
                auth_admin = i
            else:
                app_list.append(i)
        if auth_admin is not None:
            app_list.append(auth_admin)
        return app_list


class ModelAdmin(admin.ModelAdmin):
    exclude = ['history']

    def get_list_display(self, request):
        return self.model.get_fields().names

    def save_model(self, request, obj, form, change):
        obj.add_change_record(user=request.user)
        super().save_model(request, obj, form, change)

    def delete_model(self, request, obj):
        obj.add_change_record(user=request.user, is_deleted=True)
        super().delete_model(request, obj)

    @atomic
    def delete_queryset(self, request, queryset):
        for i in queryset:
            i.add_change_record(is_deleted=True, user=request.user)
            # and since Model.delete() won't be called:
            i.change.serialize()
            i.change.save()

        super().delete_queryset(request, queryset)

    def history_view(self, request, object_id, extra_context=None):
        record = self.model.objects.get(pk=object_id)
        return HistoryView.as_view()(
                request, record=record, extra_context=extra_context)
