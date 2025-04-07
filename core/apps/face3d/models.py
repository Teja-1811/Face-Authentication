from django.contrib.auth.models import AbstractUser
from django.db import models
import json

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    face_embedding = models.TextField(null=True, blank=True)  # Store face embeddings
    face_3d_model = models.FileField(upload_to="3d_faces/", null=True, blank=True)  # Store 3D model file
    login_attempts = models.IntegerField(default=0)
    otp = models.CharField(max_length=6, default='000000')

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return self.email

    def set_face_embedding(self, embedding):
        """Save face embedding as JSON string."""
        self.face_embedding = json.dumps(embedding)
        self.save()

    def get_face_embedding(self):
        """Retrieve face embedding as a NumPy array."""
        if self.face_embedding:
            return json.loads(self.face_embedding)
        return None
