import cv2
import threading
import time
import helpers
import haar_detector

DISPLAY_FPS = 20
DISPLAY_PERIOD = 1.0 / DISPLAY_FPS

latest_frame = None
processed_frame = None
boxes_haar = None
processed_timestamp = 0.0
stop_flag = False


scale, frame_width, frame_height = helpers.get_frame_size(1)
# -------------------------------------------
# 1) CAMERA THREAD: always fast
# -------------------------------------------
def camera_loop():
    global latest_frame, stop_flag
    cap = cv2.VideoCapture(1)

    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            latest_frame = frame.copy()
        else:
            time.sleep(0.001)

    cap.release()


# -------------------------------------------
# 2) PROCESSING THREAD: heavy operations
# -------------------------------------------
def processing_loop(angle):
    global boxes_haar, processed_timestamp, latest_frame, stop_flag

    while not stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        # frame = latest_frame.copy()

        # -----------------------------
        # 🔥 Your heavy processing here
        # -----------------------------
        # Example: slow grayscale

        # frame_rotated2, rotation_matrix = helpers.rotate_image(frame, 90)
        
        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame, angle)
        faces = haar_detector.detect_faces(frame_rotated)
        boxes = helpers.construct_boxes(faces, rotation_matrix, angle)

        # -----------------------------

        boxes_haar = boxes
        processed_timestamp = time.time()


# ------------------------------------------------
# Start background threads
# ------------------------------------------------
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=processing_loop, daemon=True).start()

angle_step = 45
for angle in range(0, 360, angle_step):
    threading.Thread(target=processing_loop, args=(angle,), daemon=True).start()
    # threads.append(t)

boxes = []
# ------------------------------------------------
# 3) DISPLAY LOOP — ALWAYS 20 FPS, NO LAG
# ------------------------------------------------
last_display = time.time()

try:
    while True:
        now = time.time()
        if now - last_display >= DISPLAY_PERIOD:
            last_display = now

            if latest_frame is None:
                continue

            # ----------------------------------------
            # Decide what to show:
            # processed frame is valid if < 0.5s old
            # ----------------------------------------
            
            if boxes_haar is not None and (now - processed_timestamp) < 0.1:
                boxes.extend(boxes_haar)
                while len(boxes) > 12:
                    boxes.pop(0)
                print(len(boxes))
                latest_frame = helpers.add_boxes(latest_frame, boxes)
                # frame_rotated = helpers.add_boxes(frame_rotated, boxes, 90 % 360)
            
            # latest_frame = cv2.resize(latest_frame, (0, 0), fx=scale, fy=scale)
            # frame_rotated = cv2.resize(frame_rotated, (0, 0), fx=scale, fy=scale)
            cv2.imshow("Output", latest_frame)
            # cv2.imshow("Rotated", frame_rotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.001)

except KeyboardInterrupt:
    pass

stop_flag = True
cv2.destroyAllWindows()
