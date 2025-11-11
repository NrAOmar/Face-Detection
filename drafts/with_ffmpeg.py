import cv2
import time

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Start with fallback FPS (just to init VideoWriter)
fallback_fps = 20.0
out = cv2.VideoWriter(
    'output.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fallback_fps,
    (frame_width, frame_height)
)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.moveWindow('Camera', 100, 100)

print("Recording... Press 'q' to stop.")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    out.write(frame)
    cv2.imshow("Camera", frame)

    frame_count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Compute the actual FPS
end_time = time.time()
elapsed_time = end_time - start_time
actual_fps = frame_count / elapsed_time
print(f"Measured FPS: {actual_fps:.2f}")

cap.release()
out.release()
cv2.destroyAllWindows()

# Re-encode video with correct FPS using ffmpeg (if installed)
import subprocess
try:
    subprocess.run([
        "ffmpeg", "-y", "-i", "output.mp4",
        "-filter:v", f"setpts={(fallback_fps/actual_fps):.4f}*PTS",
        "output_corrected.mp4"
    ])
    print("✅ Saved corrected video as output_corrected.mp4")
except FileNotFoundError:
    print("⚠️ ffmpeg not found — install it with 'brew install ffmpeg' for auto-correction.")
