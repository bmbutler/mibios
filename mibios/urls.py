from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from rest_framework import routers

from . import views
from .dataset import registry


rest_router = routers.DefaultRouter()
for i in registry.get_models():
    rest_router.register(i._meta.model_name, i.get_rest_api_viewset_class())

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.FrontPageView.as_view(), name='top'),
    # path('test/', TestView.as_view(), name='test'),
    path('archive/', views.SnapshotListView.as_view(), name='snapshot_list'),
    path('archive/<str:name>/', views.SnapshotView.as_view(), name='snapshot'),
    path('archive/<str:name>/<str:table>', views.SnapshotTableView.as_view(),
         name='snapshot_table'),
    path('archive/<str:name>/<str:table>/export/<str:format>/',
         views.ExportSnapshotTableView.as_view(),
         name='export_snapshot_table'),
    path('api/', include(rest_router.urls)),
    path(settings.MEDIA_URL.lstrip('/') + 'imported/<int:year>/<str:name>',
         views.ImportFileDownloadView.as_view(), name='import_file_download'),
    # fixed string paths go above this comment
    path('<str:dataset>/', views.TableView.as_view(), name='queryset_index'),
    path('<str:dataset>/mean/<str:avg_by>/', views.AverageView.as_view(),
         name='average'),
    path('<str:dataset>/import/', views.ImportView.as_view(), name='import'),
    path('<str:dataset>/export/<str:format>/', views.ExportView.as_view(),
         name='export'),
    path('<str:dataset>/mean/<str:avg_by>/export/<str:format>/',
         views.AverageExportView.as_view(), name='average_export'),
    path('<str:dataset>/history/deleted/', views.DeletedHistoryView.as_view(),
         name='deleted_history'),
    path('<str:dataset>/<str:natural>/history/', views.HistoryView.as_view(),
         name='record_history'),
]
