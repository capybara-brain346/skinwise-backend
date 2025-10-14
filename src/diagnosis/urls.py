from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("predict", views.predict_form, name="predict_form"),
    path("analyze", views.analyze_form, name="analyze_form"),
    path("predict.pdf", views.predict_pdf, name="predict_pdf"),
    path("analyze.pdf", views.analyze_pdf, name="analyze_pdf"),
    path("api/predict", views.predict_api, name="predict_api"),
    path("api/analyze", views.analyze_api, name="analyze_api"),
    path("health", views.health, name="health"),
    path("classes", views.classes, name="classes"),
]
