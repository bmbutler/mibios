# from django.apps import apps
from django.urls import path

from .views import ExportView, TableView


app_name = 'hmb'
urlpatterns = [
    path('', TableView.as_view(), name='top'),
    path('<str:dataset>/', TableView.as_view(), name='queryset_index'),
    path('<str:dataset>/export/<str:format>/', ExportView.as_view(),
         name='export'),
]
