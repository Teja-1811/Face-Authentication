from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from django.core.mail import send_mail
from django.utils.timezone import localtime

@receiver(user_logged_in)
def send_login_alert_email(sender, request, user, **kwargs):
    ip = get_client_ip(request)
    time = localtime().strftime('%Y-%m-%d %H:%M:%S')

    send_mail(
        subject="🔐 New Login Detected",
        message=f"Hi {user.first_name} {user.last_name},\n\n"
                f"You logged in at {time} \n from IP: {ip} : "
                f"If this wasn't you, change your password immediately.",
        from_email="bhanuteja18112001@gmail.com",
        recipient_list=[user.email],
        fail_silently=False,
    )

def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0]
    return request.META.get("REMOTE_ADDR")
