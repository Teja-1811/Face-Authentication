from django.shortcuts import render, redirect
from django.http import HttpResponseNotFound
from django.core.mail import send_mail
import os
import cv2
import numpy as np
import base64
import json
import face_recognition
from django.contrib.auth import login, authenticate, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib import messages
from django.conf import settings
from deepface import DeepFace

from .models import CustomUser

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
                    print(f"OpenCV detected {len(faces)} faces.")
                    face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

            # If still no faces detected, return error
            if not face_locations:
                return JsonResponse({"error": "No face detected. Ensure proper lighting and face is visible."}, status=400)

            # Encode face
            face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)

                        
            face_embedding = json.dumps(face_encodings[0].tolist())
            
            user = CustomUser.objects.create(
                username=email,
                email=email,
                first_name=fname,
                last_name=lname,
                password=make_password(password),
                face_embedding=face_embedding
            )
            messages.success(request, "Registration successful. Please log in.")
            return redirect("home")
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

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
                login(request, user)
                return JsonResponse({
                    "success": True,
                    "message": "Authentication successful!",
                    "username": f"{user.first_name} {user.last_name}"
                })
            else:
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
    user = CustomUser.objects.get(username=request.user)
    name = f"{user.first_name} {user.last_name}"
    print(name)
    return render(request, "dashboard.html", {"name": name})

def face_construction(request):
    return render(request, 'face_construction.html')

import cv2
import numpy as np
import base64
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def verify_spoof(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            image_data = data.get("image")

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
