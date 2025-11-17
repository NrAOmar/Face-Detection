import cv2
import haar_detector
import dnn_detector
import helpers
import math

camera_in_use = 2 # start with camera in the lab

# Open camera (macOS AVFoundation). Try 2 then 1 then 0.
cap = cv2.VideoCapture(camera_in_use, cv2.CAP_AVFOUNDATION)
while not cap.isOpened() and camera_in_use > 0:
    camera_in_use -= 1
    cap = cv2.VideoCapture(camera_in_use, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get frame_dnn size
scale, frame_width, frame_height = helpers.get_frame_size(camera_in_use)

print(f"frame_width {frame_width}")
print(f"frame_height {frame_height}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 20.0  # safe fallback
print(f"Recording at {fps} FPS")

# Video writer
out_haar = cv2.VideoWriter(
    'output_haar.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)
out_dnn = cv2.VideoWriter(
    'output_dnn.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

print("Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = helpers.detect_faces_multi_rotation_mapped(frame, haar_detector.face_classifier, step=90)

    display = frame.copy()

    for (bbox, angle) in detections:
        x, y, w, h = bbox
        # optionally check bounds/clamp to image size
        x = max(0, min(x, frame.shape[1]-1))
        y = max(0, min(y, frame.shape[0]-1))
        w = max(1, min(w, frame.shape[1]-x))
        h = max(1, min(h, frame.shape[0]-y))

        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display, f"{angle}°", (x, max(0, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    total = len(detections)
    cv2.putText(display, f"Total faces (all angles): {total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    show = cv2.resize(display, (0,0), fx=0.6, fy=0.6)
    cv2.imshow("Multi-Rotation Detection (mapped)", show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out_dnn.release()
out_haar.release()
cv2.destroyAllWindows()