import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ---------- InsightFace ----------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(320, 320))  # faster than 640

# ---------- Settings ----------
THRESHOLD = 0.38
SCALE = 0.5          # 0.5 means run model on half-resolution frame
RUN_EVERY = 5       # run detection+recognition every N frames

# ---------- Load known faces ----------
known_embeddings = []
known_names = []

def load_known_faces(base_path="dataset"):
    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) > 0:
                known_embeddings.append(faces[0].embedding.astype(np.float32))
                known_names.append(person)

    print(f"Loaded {len(known_embeddings)} known faces")

load_known_faces()

if len(known_embeddings) == 0:
    raise RuntimeError("No known faces loaded. Check your dataset folder and images.")

# Pre-normalize known embeddings for fast cosine
known_mat = np.stack(known_embeddings, axis=0)
known_mat /= (np.linalg.norm(known_mat, axis=1, keepdims=True) + 1e-10)

# ---------- Webcam (low latency on Windows) ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_id = 0
last_results = []  # list of (bbox(x1,y1,x2,y2), name, sim)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % RUN_EVERY == 0:
        # Downscale for faster inference
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        faces = app.get(small)

        results = []
        for f in faces:
            emb = f.embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-10)

            sims = known_mat @ emb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            name = "Unknown"
            if best_sim >= THRESHOLD:
                name = known_names[best_idx]

            # Scale bbox back to original frame coordinates
            x1, y1, x2, y2 = (f.bbox / SCALE).astype(int)
            results.append(((x1, y1, x2, y2), name, best_sim))

        last_results = results

    # Draw last results
    for (x1, y1, x2, y2), name, sim in last_results:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {sim:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Fast Face ID", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
