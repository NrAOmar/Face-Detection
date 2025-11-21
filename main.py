import cv2
import threading
import time

TARGET_FPS = 20
FRAME_PERIOD = 1.0 / TARGET_FPS

latest_frame = None
stop_flag = False

def camera_loop():
    global latest_frame, stop_flag

    cap = cv2.VideoCapture(1)

    last_time = time.time()

    while not stop_flag:
        now = time.time()
        if now - last_time >= FRAME_PERIOD:
            last_time = now

            ret, frame = cap.read()
            if not ret:
                continue

            latest_frame = frame.copy()

            # Display
            cv2.imshow("Camera", latest_frame)
            cv2.waitKey(1)  # needed for imshow

    cap.release()
    cv2.destroyAllWindows()

# -------------------------
# Start the camera thread
# -------------------------
cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()

# -------------------------
# Main loop (heavy processing)
# -------------------------
try:
    while True:
        if latest_frame is not None:
            # Do heavy processing here
            processed = latest_frame  # placeholder

            # simulate heavy work
            time.sleep(0.2)  # 200ms = 5 FPS processing
        else:
            time.sleep(0.01)

except KeyboardInterrupt:
    stop_flag = True
    cam_thread.join()
