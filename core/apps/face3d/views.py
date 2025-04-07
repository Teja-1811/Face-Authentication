import os
import cv2
import numpy as np
import base64
import json
import face_recognition
from deepface import DeepFace
import mediapipe as mp
import open3d as o3d
import tempfile
from django.shortcuts import render, redirect
from django.http import HttpResponseNotFound
from django.core.mail import send_mail
from django.contrib.auth import login, authenticate, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib import messages
from django.conf import settings
from django.core.mail import send_mail
from django.contrib.auth.views import PasswordResetConfirmView
from django.urls import reverse_lazy
from django.utils.timezone import now

from .models import CustomUser
from .otp import generate_otp

mp_face_mesh = mp.solutions.face_mesh

def get_client_ip(request):
    """Extract IP address from request headers."""
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0].strip()
    else:
        ip = request.META.get("REMOTE_ADDR")
    return ip

def custom_404_view(request, exception):
    return HttpResponseNotFound('<h1>404 - Page Not Found</h1>')

def home(request):
    return render(request, "home.html")

def decode_base64_image(image_data):
    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            print("Error: Image decoding failed!")
        else:
            print("Image successfully decoded!")
        return image
    except Exception as e:
        print("Error decoding image:", e)
        return None

def generate_3d_face(image):
    """Generate 3D model from image using MediaPipe Face Mesh."""
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image for face mesh
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        return None  # No face detected

    # Extract 3D face mesh points
    landmarks = results.multi_face_landmarks[0].landmark
    points = [(lm.x, lm.y, lm.z) for lm in landmarks]

    # Save as a simple .obj file (Wavefront OBJ format)
    obj_data = "o FaceModel\n"
    for x, y, z in points:
        obj_data += f"v {x} {y} {z}\n"

    return obj_data  # Return the .obj model data

@csrf_exempt
def register(request):
    if request.method == "POST":
        try:
            if request.content_type == "application/json":
                data = json.loads(request.body)
                fname = data.get("fname")
                lname = data.get("lname")
                email = data.get("email")
                password = data.get("password")
                image_data = data.get("image")  # Base64 image from frontend
            else:
                fname = request.POST.get("fname")
                lname = request.POST.get("lname")
                email = request.POST.get("email")
                password = request.POST.get("password")
                image_data = request.POST.get("face_image")  # Base64 image from frontend
            
            if not all([fname, lname, email, password, image_data]):
                return JsonResponse({"error": "All fields are required."}, status=400)

            face_image = decode_base64_image(image_data)
            if face_image is None:
                return JsonResponse({"error": "Invalid image format!"}, status=400)

            rgb_frame = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            print(f"Decoded image shape: {rgb_frame.shape}")

            # Try face recognition detection (HOG model)
            face_locations = face_recognition.face_locations(rgb_frame)
            print(f"Detected Faces (HOG): {face_locations}")

            # If no faces detected, try CNN model
            if not face_locations:
                print("Trying CNN model...")
                face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

            # If still no faces, try OpenCV Haar Cascade
            if not face_locations:
                print("Trying OpenCV Cascade...")
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    #print(f"OpenCV detected {len(faces)} faces.")
                    face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

            # If still no faces detected, return error
            if not face_locations:
                return JsonResponse({"error": "No face detected. Ensure proper lighting and face is visible."}, status=400)

            # Encode face
            face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)

                        
            face_embedding = json.dumps(face_encodings[0].tolist())
            
           # Generate 3D face model
            face_3d_obj = generate_3d_face(face_image)
            if not face_3d_obj:
                return JsonResponse({"error": "Could not generate 3D face."}, status=400)
            
            #print(face_3d_obj)
            # Save 3D model to a file
            file_name = f"{email}.obj"
            file_path = os.path.join(settings.MEDIA_ROOT, "3d_faces", file_name)

            with open(file_path, "w") as f:
                f.write(face_3d_obj)

            user = CustomUser.objects.create(
                username=email,
                email=email,
                first_name=fname,
                last_name=lname,
                password=make_password(password),
                face_embedding=face_embedding,
                face_3d_model=f"3d_faces/{file_name}"
            )
            send_mail(
                subject="Welcome to 3D Face Auth System!",
                message=f"Hii {fname} {lname}!\n\nThank you for registering with us!",
                from_email=None,
                recipient_list=[email],
                fail_silently=False,
            )
            messages.success(request, "Registration successful. Please log in.")
            return redirect("home")
        
        except Exception as e:
            messages.error(request, "Registration Failed. Please Try Again.")
            return redirect("home")

    return render(request, "register.html")

@csrf_exempt
def login_view(request):
    return render(request, "login.html")

@csrf_exempt
def face_login(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_email = data.get("email")
            image_data = data.get("image")
            print(user_email)
            #print(image_data)
            face_image = decode_base64_image(image_data)
            
            user = CustomUser.objects.get(username = user_email) 
            if user is None:
                return JsonResponse({"error" : "User not registered"}, status=400)
            
            if face_image is None:
                return JsonResponse({"error": "Invalid face image."}, status=400)
            
            uploaded_encoding = np.array(face_recognition.face_encodings(face_image)[0])
            stored_encoding = np.array(json.loads(user.face_embedding))
            
            match = face_recognition.compare_faces([stored_encoding], uploaded_encoding, tolerance=0.6)
            
            if match[0]:
                login(request, user)
                return JsonResponse({
                    "success": True,
                    "message": "Face authentication successful!",
                    "username": f"{user.first_name} {user.last_name}"
                })
            return JsonResponse({"error": "Face not recognized. Try again."}, status=401)
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    
    return JsonResponse({"error": "Invalid request."}, status=400)

def password_login(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            email = data.get("email")
            password = data.get("password")
            
            user = authenticate(username=email, password=password)
            if user is not None:
                otp = generate_otp()
                user.otp = otp
                user.save()

                send_mail(
                    subject="🔐 Your OTP Code",
                    message=f"Hi {user.first_name},\n\nYour OTP code is: {otp}\nIt expires in 5 minutes.\n\nBest,\nSecurity Team",
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[user.email],
                    fail_silently=False,
                )

                return JsonResponse({
                    "success": True,
                    "message": "OTP sent to your email. Please verify.",
                    "require_otp": True
                })
            else:
                try:
                    user = CustomUser.objects.get(email=email)
                    user.login_attempts += 1
                    user.save()
                    if user.login_attempts >= 3:
                        ip_address = get_client_ip(request)
                        timestamp = now().strftime('%Y-%m-%d %H:%M:%S')
                        send_mail(
                            subject="🚨 Login Attempt Failed",
                            message=(
                                f"Hi {user.first_name} {user.last_name},\n\n"
                                f"A failed login attempt was detected on your account.\n\n"
                                f"Details:\n"
                                f"- Time: {timestamp}\n"
                                f"- IP Address: {ip_address}\n\n"
                                f"If this wasn't you, please change your password or contact support immediately.\n\n"
                                f"Best,\nSecurity Team"
                            ),
                            from_email=settings.DEFAULT_FROM_EMAIL,
                            recipient_list=[user.email],
                            fail_silently=False,
                        )
                except CustomUser.DoesNotExist:
                    pass
                return JsonResponse({"error": "Invalid email or password."}, status=400)
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid request format."}, status=400)
    return render(request, 'p_login.html')

@csrf_exempt
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("/")

@login_required
def dashboard_view(request):
    user = request.user
    model_path = user.face_3d_model.url if user.face_3d_model else None
    
    return render(request, "dashboard.html", {"name": user.first_name, "model_path": model_path})

@csrf_exempt
def verify_spoof(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            image_data = data.get("image")
            #print(image_data)

            if not image_data:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Decode Base64 Image
            image_bytes = base64.b64decode(image_data.split(",")[1])
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({"error": "Invalid image format"}, status=400)

            # Convert Image to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Load Pretrained Spoof Detection Model (Basic LBP)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return JsonResponse({"error": "No face detected"}, status=400)

            # Simple Heuristic: Blurriness Check
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:
                return JsonResponse({"error": "Spoof detected: Blurry image"}, status=400)

            return JsonResponse({"success": True, "message": "No spoof detected"})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=400)

@login_required
def reset_face(request):
    if request.method == "POST":
        try:
            if request.content_type == "application/json":
                data = json.loads(request.body)
                image_data = data.get("image")
            else:
                image_data = request.POST.get("face_image")

            if not image_data:
                return JsonResponse({"error": "Face image is required."}, status=400)

            face_image = decode_base64_image(image_data)
            if face_image is None:
                return JsonResponse({"error": "Invalid image format!"}, status=400)

            rgb_frame = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if not face_locations:
                face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

            if not face_locations:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

            if not face_locations:
                return JsonResponse({"error": "No face detected. Ensure proper lighting and face is visible."}, status=400)

            face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)

            face_embedding = json.dumps(face_encodings[0].tolist())

            # Generate 3D face model
            face_3d_obj = generate_3d_face(face_image)
            if not face_3d_obj:
                return JsonResponse({"error": "Could not generate 3D face."}, status=400)

            file_name = f"{request.user.email}.obj"
            file_path = os.path.join(settings.MEDIA_ROOT, "3d_faces", file_name)

            with open(file_path, "w") as f:
                f.write(face_3d_obj)

            # Update user's face embedding and model
            user = request.user
            user.face_embedding = face_embedding
            user.face_3d_model = f"3d_faces/{file_name}"
            user.save()

            return JsonResponse({"success": True, "message": "Face data reset successful!", "redirect_url": "/dashboard/"})


        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return render(request, "reset_face.html")

class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'password_reset_confirm.html'
    success_url = reverse_lazy('password_reset_complete')

    def form_valid(self, form):
        response = super().form_valid(form)

        # Send password reset success email
        if self.user and self.user.email:
            send_mail(
                subject='Your Password Has Been Changed',
                message='Hi, your password has been successfully changed. If this wasn’t you, please reset your password or contact support immediately.',
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[self.user.email],
                fail_silently=False,
            )
            messages.success(self.request, "Password changed successfully")
        
        return response
    

@csrf_exempt
def verify_otp(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            email = data.get("email")
            otp = data.get("otp")

            try:
                user = CustomUser.objects.get(email=email)
                if user.otp == otp:
                    login(request, user)
                    user.login_attempts = 0
                    user.otp = ""  # Clear OTP after successful login
                    user.save()
                    return JsonResponse({
                        "success": True,
                        "message": "Login successful via OTP!",
                        "username": f"{user.first_name} {user.last_name}"
                    })
                else:
                    return JsonResponse({"error": "Invalid OTP."}, status=400)

            except CustomUser.DoesNotExist:
                return JsonResponse({"error": "User does not exist."}, status=400)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid request format."}, status=400)
