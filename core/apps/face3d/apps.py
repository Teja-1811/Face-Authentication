from django.apps import AppConfig


class Face3DConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.face3d'
    
    def ready(self):
        import apps.face3d.signals
