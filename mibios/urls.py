# from django.apps import apps
from django.urls import path

from .admin import site as admin_site
from .views import ExportView, HistoryView, ImportView, TableView, TestView


urlpatterns = [
    # take over admin history view with our own
    path('admin/mibios/<str:dataset>/<int:pk>/history/', HistoryView.as_view(),
         name='history'),
    path('admin/', admin_site.urls),
    path('', TableView.as_view(), name='top'),
    # path('test/', TestView.as_view(), name='test'),
    path('<str:dataset>/', TableView.as_view(), name='queryset_index'),
    path('<str:dataset>/import/', ImportView.as_view(), name='import'),
    path('<str:dataset>/export/<str:format>/', ExportView.as_view(),
         name='export'),
]
