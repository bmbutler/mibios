"""mibios URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import include, path
from django.conf.urls import handler404

nofavex = Exception(
    'Dear web browser manufacturers: Please don\'t ask for favicon.ico on '
    'every request'
)

urlpatterns = [
    path('favicon.ico', handler404, {'exception': nofavex}),  # reduce log spam
    path('', include('mibios.urls')),  # catch all, go last
]
