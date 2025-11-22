import cv2
import threading
import time
import helpers
import haar_detector
import dnn_detector
import camera
from camera import stop_flag

# Features to enable
flag_rotation = True
flag_haar = True
flag_dnn = True
flag_enhancement = False
flag_lowPassFilter = False
flag_biometric = False




latest_frame = None
rotated_frame = None
processed_frame = None
stop_flag = False


# Model file names
PROTOTXT = "deploy.prototxt"
CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# URLs used by the official OpenCV sample
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Try to make sure the model files exist (skip if offline and already present)
have_proto = dnn_detector.ensure_file(PROTOTXT, PROTOTXT_URL)
have_model = dnn_detector.ensure_file(CAFFEMODEL, CAFFEMODEL_URL)

if not (have_proto and have_model):
    print("Model files missing. Place deploy.prototxt and the caffemodel next to this script.")
    # You can still continue if you already have them elsewhere and set absolute paths.
    # exit()


# -------------------------------------------
# 1) CAMERA THREAD: always fast
# -------------------------------------------

boxes_by_angle = {}
timestamps_by_angle = {}
lock = threading.Lock()

# -------------------------------------------
# 2) PROCESSING THREAD: heavy operations
# -------------------------------------------
def haar_loop(angle):
    global latest_frame, stop_flag

    while not stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        tmp_start = time.time()
        # frame = latest_frame.copy()

        # -----------------------------
        # 🔥 Your heavy processing here
        # -----------------------------
        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), angle)        
        faces = haar_detector.detect_faces(frame_rotated)
        boxes = helpers.construct_boxes(faces, (angle,))
        # -----------------------------

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[("haar", angle)] = boxes
            timestamps_by_angle[("haar", angle)] = time.time()
        
        tmp_end = time.time()
        print(f"duration = {tmp_end - tmp_start}")

def dnn_loop(angle):
    global latest_frame, stop_flag
    
    while not stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        tmp_start = time.time()
        # frame = latest_frame.copy()

        # -----------------------------
        # 🔥 Your heavy processing here
        # -----------------------------
        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), camera.frame_size, angle)        
        faces, confidence = dnn_detector.detect_faces(frame_rotated, (PROTOTXT, CAFFEMODEL))
        boxes = helpers.construct_boxes(faces, (angle, confidence))
        # -----------------------------

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[("dnn", angle)] = boxes
            timestamps_by_angle[("dnn", angle)] = time.time()
        
        tmp_end = time.time()
        print(f"duration = {tmp_end - tmp_start}")


# ------------------------------------------------
# Start background threads
# ------------------------------------------------
threads = []
threading.Thread(target=camera.camera_loop, daemon=True).start()
# threading.Thread(target=haar_loop, args=(340,), daemon=True).start()
# threading.Thread(target=haar_loop, args=(20,), daemon=True).start()

angle_step = 20
for angle in range(0, 360, angle_step):
    threading.Thread(target=haar_loop, args=(angle,), daemon=True).start()
    threading.Thread(target=dnn_loop, args=(angle,), daemon=True).start()

combined_boxes = []
# ------------------------------------------------
# 3) DISPLAY LOOP — ALWAYS 20 FPS, NO LAG
# ------------------------------------------------
last_display = time.time()

try:
    while True:
        now = time.time()
        latest_frame = camera.latest_frame
        rotated_frame = camera.rotated_frame
        if now - last_display >= 1/camera.fps:
            last_display = now

            if latest_frame is None:
                continue

           # ----------------------------------------
            # Decide what to show:
            # processed frame is valid if < 0.5s old
            # ----------------------------------------
            new_combined_boxes = []

            with lock:
                for key, boxes in boxes_by_angle.items():
                    ts = timestamps_by_angle.get(key, 0)
                    if now - ts < 0.5:
                        # store each box with its timestamp
                        for box in boxes:
                            new_combined_boxes.append((box, ts))

            # Remove old boxes (older than 0.5s)
            combined_boxes = [(box, ts) for box, ts in combined_boxes if now - ts < 0.2]

            # Add new boxes
            combined_boxes.extend(new_combined_boxes)
            # combined_boxes = combined_boxes[-12:]

            # Extract just the box coordinates for drawing
            boxes_to_draw = [box for box, ts in combined_boxes]

            output_frame = helpers.add_boxes(latest_frame.copy(), boxes_to_draw)
            # rotated_frame = helpers.add_boxes(rotated_frame.copy(), boxes_to_draw, False)
            cv2.imshow("Camera (Haar)", output_frame)
            # cv2.imshow("Rotated (Haar)", rotated_frame)
            if camera.out_haar != "":
                camera.out_haar.write(output_frame)
            else:
                print("no haar frame found")

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.001)

except KeyboardInterrupt:
    pass

stop_flag = True
camera.out_dnn.release()
camera.out_haar.release()
cv2.destroyAllWindows()
