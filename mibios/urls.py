from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
from rest_framework import routers

from . import views, get_registry


rest_router = routers.DefaultRouter()
for i in get_registry().get_models():
    rest_router.register(i._meta.model_name, i.get_rest_api_viewset_class())

archive_urls = [
    # URLs starting with archive/
    path('', views.SnapshotListView.as_view(), name='snapshot_list'),
    path('<str:name>/', views.SnapshotView.as_view(), name='snapshot'),
    path('<str:name>/<str:app>/<str:table>',
         views.SnapshotTableView.as_view(), name='snapshot_table'),
    path('<str:name>/<str:app>/<str:table>/export/<str:format>/',
         views.ExportSnapshotTableView.as_view(),
         name='export_snapshot_table'),
]

data_name_urls = [
    # URLs starting <str:data_name>/
    path('', views.TableView.as_view(), name='table'),
    path('mean/<str:avg_by>/', views.AverageView.as_view(),
         name='average'),
    path('import/', views.ImportView.as_view(), name='import'),
    path('export-form/', views.ExportFormView.as_view(),
         name='export_form'),
    path('export/', views.ExportView.as_view(),
         name='export'),
    path('show-hide-form/', views.ShowHideFormView.as_view(),
         name='show_hide_form'),
    path('mean/<str:avg_by>/export-form/',
         views.AverageExportFormView.as_view(), name='average_export_form'),
    path('mean/<str:avg_by>/export/',
         views.AverageExportView.as_view(), name='average_export'),
    path('history/deleted/', views.DeletedHistoryView.as_view(),
         name='deleted_history'),
    path('<str:natural>/history/', views.HistoryView.as_view(),
         name='record_history'),
]

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.FrontPageView.as_view(), name='top'),
    path('history/', views.CompactHistoryView.as_view(),
         name='compact_history'),
    re_path(
        # e.g.: /history/1234-2345/
        r'^history/(?P<first>[0-9]+)[-](?P<last>[0-9]+)/',
        views.DetailedHistoryView.as_view(),
        name='detailed_history',
    ),
    # path('test/', TestView.as_view(), name='test'),
    path('archive/', include(archive_urls)),
    path('api/', include(rest_router.urls)),
    path(settings.MEDIA_URL.lstrip('/') + 'imported/<int:year>/<str:name>',
         views.ImportFileDownloadView.as_view(), name='import_file_download'),
    path('log/<int:import_file_pk>/', views.LogView.as_view(), name='log'),
    # fixed string paths go above this comment
    path('<str:data_name>/', include(data_name_urls)),
]


def get_app_urls():
    """
    Get url patterns from registered apps

    Includes each registered app's urls module if there is one.
    """
    ret = []
    for i in get_registry().apps.values():
        if i.name == 'mibios':
            # skip including this module, avoids infinite recursion
            continue

        try:
            ret.append(path('', include(i.name + '.urls')))
        except ModuleNotFoundError:
            # app has no urls module, skip
            pass
    return ret


urlpatterns += get_app_urls()
