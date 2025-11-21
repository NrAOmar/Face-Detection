import cv2
import threading
import time

FRAME_RATE = 20
FRAME_PERIOD = 1.0 / FRAME_RATE

latest_frame = None
stop_flag = False


def camera_loop():
    """Thread that grabs frames continuously."""
    global latest_frame, stop_flag

    cap = cv2.VideoCapture(1)

    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        else:
            time.sleep(0.001)

    cap.release()


# Start camera thread
cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()


# MAIN THREAD: fixed FPS display
last_time = time.time()

try:
    while True:
        now = time.time()
        if now - last_time >= FRAME_PERIOD:
            last_time = now

            if latest_frame is not None:
                cv2.imshow("Camera", latest_frame)

            # keep window responsive
            if cv2.waitKey(1) == 27:
                break

        else:
            # tiny sleep to not burn CPU
            time.sleep(0.001)

except KeyboardInterrupt:
    pass

stop_flag = True
cam_thread.join()
cv2.destroyAllWindows()
