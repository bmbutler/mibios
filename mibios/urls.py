from django.contrib import admin
from django.urls import path

from . import views


urlpatterns = [
    # take over admin history view with our own
    path('admin/<str:app>/<str:dataset>/<int:pk>/history/',
         views.HistoryView.as_view(), name='history'),
    path('admin/', admin.site.urls),
    path('', views.FrontPageView.as_view(), name='top'),
    # path('test/', TestView.as_view(), name='test'),
    path('<str:dataset>/', views.TableView.as_view(), name='queryset_index'),
    path('<str:dataset>/import/', views.ImportView.as_view(), name='import'),
    path('<str:dataset>/export/<str:format>/', views.ExportView.as_view(),
         name='export'),
    path('<str:dataset>/history/deleted/', views.DeletedHistoryView.as_view(),
         name='deleted_history'),
]
