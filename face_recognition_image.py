import face_recognition
import cv2
import os

# ---------------------------------------------------
# 1. Load known images and encode them
# ---------------------------------------------------

known_faces_dir = "known_faces"
known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png", ".jpeg", ".HEIC")):
        name = os.path.splitext(filename)[0]   # e.g., "mohamed.jpg" → "mohamed"
        img_path = os.path.join(known_faces_dir, filename)

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)
        else:
            print(f"[WARNING] No face found in {filename}")

# ---------------------------------------------------
# 2. Load the unknown image
# ---------------------------------------------------

unknown_image = face_recognition.load_image_file("WhatsApp Image 2025-12-08 at 10.03.22.jpeg")
unknown_encodings = face_recognition.face_encodings(unknown_image)

if len(unknown_encodings) == 0:
    print("No face detected in the unknown image!")
    exit()

unknown_encoding = unknown_encodings[0]

# ---------------------------------------------------
# 3. Compare the unknown face to known faces
# ---------------------------------------------------

results = face_recognition.compare_faces(known_encodings, unknown_encoding)
face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)

best_match_index = face_distances.argmin()

if results[best_match_index]:
    print(f"✔ The person in the image is: {known_names[best_match_index]}")
else:
    print("❌ No match found")
