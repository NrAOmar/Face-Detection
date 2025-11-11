# import numpy as np
import cv2

camera_in_use = 0 # start with camera in the lab

# Open camera (macOS AVFoundation). Try 2 then 1 then 0.
cap = cv2.VideoCapture(camera_in_use, cv2.CAP_AVFOUNDATION)
while not cap.isOpened() and camera_in_use > 0:
    camera_in_use -= 1
    cap = cv2.VideoCapture(camera_in_use, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"frame_width {frame_width}")
print(f"frame_height {frame_height}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 20.0  # safe fallback
print(f"Recording at {fps} FPS")

# Define the codec and create VideoWriter object
# 'mp4v' is the codec for .mp4 files
out = cv2.VideoWriter('output.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

print("Recording... Press 'q' to stop.")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create a resizable window (so it doesn't overflow your screen)
cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.moveWindow('Camera', 0, 0)    # x, y position on screen

scale = 0.3
if (camera_in_use == 1):
    scale = 0.5

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_classifier.detectMultiScale(
        frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Write the frame into the file 'output.mp4'
    out.write(frame)

    # Show the frame
    display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    cv2.imshow("Camera", display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
out.release()
cv2.destroyAllWindows()