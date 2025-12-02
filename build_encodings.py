import os
import cv2
import face_recognition
import pickle

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"

known_encodings = []
known_names = []

# Allowed image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)

    # Skip if not a folder
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing person: {person_name}")

    for image_name in os.listdir(person_folder):
        _, ext = os.path.splitext(image_name.lower())
        if ext not in IMAGE_EXTENSIONS:
            continue  # skip non-images

        image_path = os.path.join(person_folder, image_name)
        print(f"   -> {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"   [WARN] Could not read {image_path}")
            continue

        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find faces in the image
        boxes = face_recognition.face_locations(rgb, model="hog")

        if len(boxes) == 0:
            print("      [WARN] No face found in this image, skipping.")
            continue

        # Compute encodings (may be multiple faces, but usually one)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print(f"[INFO] Found {len(known_encodings)} face encodings in total.")

# Save encodings to file
data = {
    "encodings": known_encodings,
    "names": known_names,
}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encodings saved to {ENCODINGS_FILE}")
