from django.urls import path

from .views import index, generate_report, details, save_phishing, save_no_phishing

urlpatterns = [
    path('home', index, name='index'),
    path('report', generate_report, name="report"),
    path('details/<str:row>', details, name='details'),
    path('save_phishing/<str:row>', save_phishing, name='save_phishing'),
    path('save_no_phishing/<str:row>', save_no_phishing, name='save_no_phishing'),
]
