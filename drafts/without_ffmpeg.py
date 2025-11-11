import cv2
import time

# --- Open the camera ---
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

# --- Load face detector ---
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- Setup display window ---
cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.moveWindow('Camera', 100, 100)

print("Recording... Press 'q' to stop.")

# --- Measure real FPS for the first second ---
start_time = time.time()
frame_count = 0
while frame_count < 30:  # capture 30 frames to estimate FPS
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
elapsed = time.time() - start_time
real_fps = frame_count / elapsed if elapsed > 0 else 15.0
print(f"Measured real FPS: {real_fps:.2f}")

# --- Prepare VideoWriter using measured FPS ---
out = cv2.VideoWriter(
    'output.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    real_fps,
    (frame_width, frame_height)
)

frame_delay = 1 / real_fps  # time to wait per frame

# --- Record loop ---
while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Write frame
    out.write(frame)

    # Show preview (scaled down for screen)
    display = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow("Camera", display)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Wait to maintain consistent FPS
    elapsed_loop = time.time() - loop_start
    if elapsed_loop < frame_delay:
        time.sleep(frame_delay - elapsed_loop)

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording saved as output.mp4")
