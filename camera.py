import cv2
import helpers
import time

stop_flag = False
caps = {}
cameras_in_use = 3 # start with camera in the lab, then mac, then iphone
frame_size = None

def get_frame_size(cap):
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    scale = 1.0
    return (scale, frame_width, frame_height)

def get_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 20.0  # safe fallback
    # print(f"Camera " + cap + " is recording at {fps} FPS")
    print(f"Camera is recording at {fps} FPS")
    return fps

def camera_loop():
    global caps, stop_flag, cameras_in_use, frame_size

    # Open camera (macOS AVFoundation). Try 2 then 1 then 0.
    for i in range(0, cameras_in_use):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            caps[cap] = (None, get_fps(cap))
            print("Recording... Press 'Esc' to stop.")
            if i == 0:
                frame_size = get_frame_size(cap)
        else:
            print("Error: Could not open camera " + str(i))
            cameras_in_use -= 1

    if len(caps) == 0:
        exit()

    while not stop_flag:
        for cap, values in caps.items():
            ret, frame = cap.read()
            if ret:
                caps[cap] = (frame.copy(), values[1])
            else:
                time.sleep(0.001)

    for cap, values in caps.items():
        cap.release()
