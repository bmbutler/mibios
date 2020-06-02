from django.apps import apps
from django.urls import path

from .views import ModelIndexView


app_name = 'hmb'
urlpatterns = [
        path('', ModelIndexView.as_view(), name='top_index'),
    ]

for i in apps.get_app_config(app_name).get_models():
    p = i._meta.model_name + '/'
    name = i._meta.model_name + '_index'
    urlpatterns.append(path(p, ModelIndexView.as_view(model=i), name=name))

del i, p, name
