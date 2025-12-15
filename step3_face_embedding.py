import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis

# ---------- InsightFace ----------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- MediaPipe ----------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

cap = cv2.VideoCapture(0)

FACE_SIZE = 112
MARGIN = 0.2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            mx = int(bw * MARGIN)
            my = int(bh * MARGIN)

            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x1 + bw + 2 * mx)
            y2 = min(h, y1 + bh + 2 * my)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))

            # InsightFace expects RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            faces = app.get(face_rgb)

            if len(faces) > 0:
                embedding = faces[0].embedding

                # Show vector info
                print("Embedding shape:", embedding.shape)
                print("First 10 values:", embedding[:10])

                cv2.putText(
                    frame,
                    "Embedding OK",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Face Crop", face_crop)

    cv2.imshow("STEP 3 - Face Embedding", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
