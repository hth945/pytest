"""bookLib URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^search/', views.PCB_search, name='PCB_search'),
    url(r'^upPCB/', views.upPCB, name='PCB_up'),
    url(r'^login/', views.user_login, name='user_login'),
    url(r'^dingLogin/', views.ding_url, name='dingLogin'),
    url(r'^logout/', views.user_logout, name='user_logout'),
    url(r'^register/', views.user_register, name='user_register'),
    url(r'^set_password/', views.set_password, name='set_password'),
    url(r'^about/', views.about, name='about'),
    # url(r'^login/', views.user_login, name='user_login'),
    # url(r'^search/', views.book_search, name='book_search'),
    # url(r'^$', views.hello),
]



