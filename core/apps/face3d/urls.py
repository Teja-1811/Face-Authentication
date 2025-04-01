from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

from django.http import HttpResponse
from django.conf.urls import handler404

def favicon_view(request):
    return HttpResponse(status=204)


urlpatterns = [
    #path('', views.face_construction, name='face_construction'),
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('verify-spoof/', views.verify_spoof, name='verify-spoof'),
    path("login/", views.login_view, name="login"),
    path("p_login/", views.password_login, name='p_login'),
    path("face-login/", views.face_login, name="face_login"),
    path("dashboard/", views.dashboard_view, name="dashboard"),  # Add dashboard page
    path("logout/", views.logout_view, name="logout"),  # Logout route
] 
urlpatterns += [path('favicon.ico', favicon_view)]
handler404 = "apps.face3d.views.custom_404_view" 

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)