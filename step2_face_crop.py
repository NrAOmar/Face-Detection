import cv2
import mediapipe as mp

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

cap = cv2.VideoCapture(0)

FACE_SIZE = 160       # standard size (FaceNet-friendly)
MARGIN = 0.2          # add margin around face

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

            # Convert to pixel coordinates
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Add margin
            mx = int(bw * MARGIN)
            my = int(bh * MARGIN)

            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x1 + bw + 2 * mx)
            y2 = min(h, y1 + bh + 2 * my)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # Resize for recognition models
            face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))

            # Show cropped face
            cv2.imshow("Cropped Face", face_crop)

            # Draw box on original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("STEP 2 - Face Crop", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

