from django.urls import path

from .views import index, generate_report, details

urlpatterns = [
    path('home', index, name='index'),
    path('report', generate_report, name="report"),
    path('details/<str:row>', details, name='details'),
]
