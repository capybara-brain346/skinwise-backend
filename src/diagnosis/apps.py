from django.apps import AppConfig
from django.db.models.signals import post_migrate


class DiagnosisConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "diagnosis"

    def ready(self):
        try:
            from .services import (
                init_s3_service,
                ensure_model_loaded,
                ensure_gemini_loaded,
            )

            init_s3_service()
            ensure_model_loaded()
            ensure_gemini_loaded()
        except Exception:
            pass
