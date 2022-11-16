from django.urls import path

from .views import index, details, generate_report

urlpatterns = [
    path('home', index, name='index'),
    path('details/<str:row>/', details, name="details"),
    path('report', generate_report, name="report"),
]
