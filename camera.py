import cv2
import helpers
import time

camera_in_use = 2 # start with camera in the lab
stop_flag = False
latest_frame = None

# Open camera (macOS AVFoundation). Try 2 then 1 then 0.
cap = cv2.VideoCapture(camera_in_use)
while not cap.isOpened() and camera_in_use > 0:
    camera_in_use -= 1
    cap = cv2.VideoCapture(camera_in_use)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


def get_frame_size():
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    scale = 1.0
    return (scale, frame_width, frame_height)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 20.0  # safe fallback
print(f"Recording at {fps} FPS")

# Get frame_dnn size
frame_size = get_frame_size()
# Source - https://stackoverflow.com/a
# Posted by Hannes Ovrén
# Retrieved 2025-11-22, License - CC BY-SA 2.5

out_frame_size = (int(frame_size[0] * frame_size[1]), 
                  int(frame_size[0] * frame_size[2]))

# Video writer
out_haar = cv2.VideoWriter(
    'output_haar.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    out_frame_size
)
out_dnn = cv2.VideoWriter(
    'output_dnn.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    out_frame_size
)

print("Recording... Press 'q' to stop.")

def camera_loop():
    global latest_frame, stop_flag
    # cap = cv2.VideoCapture(1)

    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame.copy()
        else:
            time.sleep(0.001)

    cap.release()
