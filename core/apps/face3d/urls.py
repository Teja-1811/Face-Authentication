from django.urls import path
from . import views
from .views import CustomPasswordResetConfirmView
from django.conf import settings
from django.conf.urls.static import static

from django.http import HttpResponse
from django.conf.urls import handler404
from django.contrib.auth import views as auth_views

def favicon_view(request):
    return HttpResponse(status=204)


urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('verify-spoof/', views.verify_spoof, name='verify-spoof'),
    path("login/", views.login_view, name="login"),
    path("p_login/", views.password_login, name='p_login'),
    path("face-login/", views.face_login, name="face_login"),
    path("dashboard/", views.dashboard_view, name="dashboard"),  # Add dashboard page
    path("logout/", views.logout_view, name="logout"),  # Logout route
    
    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='password_reset.html'), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', CustomPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
    
    path('reset-face/', views.reset_face, name='reset_face'),
] 
urlpatterns += [path('favicon.ico', favicon_view)]
handler404 = "apps.face3d.views.custom_404_view" 

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)