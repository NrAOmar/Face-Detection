import cv2
import haar_detector
import dnn_detector
import helpers

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
        print("Failed to grab frame.")
        break

#Haar
    frame_haar = haar_detector.detect_faces(frame.copy(), out_haar)
    # Show the frame
    display_frame_haar = cv2.resize(frame_haar, (0, 0), fx=scale, fy=scale)
    cv2.imshow('Camera (Haar)', display_frame_haar)

#DNN
    frame_dnn = dnn_detector.detect_faces(frame.copy(), out_dnn)
    # Show the frame
    display_frame_dnn = cv2.resize(frame_dnn, (0, 0), fx=scale, fy=scale)
    cv2.imshow('Camera (DNN)', display_frame_dnn)

    # Wait for close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_dnn.release()
out_haar.release()
cv2.destroyAllWindows()