from django.contrib import admin
from django.urls import path

from . import views


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
    # fixed string paths go above this comment
    path('<str:dataset>/', views.TableView.as_view(), name='queryset_index'),
    path('<str:dataset>/import/', views.ImportView.as_view(), name='import'),
    path('<str:dataset>/export/<str:format>/', views.ExportView.as_view(),
         name='export'),
    path('<str:dataset>/history/deleted/', views.DeletedHistoryView.as_view(),
         name='deleted_history'),
    path('<str:dataset>/<str:natural>/history/', views.HistoryView.as_view(),
         name='record_history'),

]
