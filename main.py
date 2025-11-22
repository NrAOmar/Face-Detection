import cv2
import threading
import time
import helpers
import haar_detector


# Features to enable
flag_rotation = True
flag_haar = True
flag_dnn = True
flag_enhancement = False
flag_lowPassFilter = False
flag_biometric = False
camera_in_use = 2 # start with camera in the lab

# Open camera (macOS AVFoundation). Try 2 then 1 then 0.
cap = cv2.VideoCapture(camera_in_use)
while not cap.isOpened() and camera_in_use > 0:
    camera_in_use -= 1
    cap = cv2.VideoCapture(camera_in_use)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get frame_dnn size
scale, frame_width, frame_height = helpers.get_frame_size(camera_in_use)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 20.0  # safe fallback
print(f"Recording at {fps} FPS")

# Video writer
out_haar = cv2.VideoWriter(
    'output_haar.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (int(scale * frame_width), int(scale * frame_height))
)
out_dnn = cv2.VideoWriter(
    'output_dnn.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (int(scale * frame_width), int(scale * frame_height))
)

print("Recording... Press 'q' to stop.")

latest_frame = None
processed_frame = None
stop_flag = False

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


boxes_by_angle = {}
timestamps_by_angle = {}
lock = threading.Lock()

# -------------------------------------------
# 2) PROCESSING THREAD: heavy operations
# -------------------------------------------
def processing_loop(angle):
    global latest_frame, stop_flag

    while not stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        # frame = latest_frame.copy()

        # -----------------------------
        # 🔥 Your heavy processing here
        # -----------------------------
        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), angle)        
        faces = haar_detector.detect_faces(frame_rotated)
        boxes = helpers.construct_boxes(helpers.get_frame_size(1), faces, rotation_matrix, angle)
        # -----------------------------

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[angle] = boxes
            timestamps_by_angle[angle] = time.time()


# ------------------------------------------------
# Start background threads
# ------------------------------------------------
threads = []
threading.Thread(target=camera_loop, daemon=True).start()

angle_step = 20
for angle in range(0, 360, angle_step):
    t = threading.Thread(target=processing_loop, args=(angle,), daemon=True)
    t.start()
    threads.append(t)

combined_boxes = []
# ------------------------------------------------
# 3) DISPLAY LOOP — ALWAYS 20 FPS, NO LAG
# ------------------------------------------------
last_display = time.time()

try:
    while True:
        now = time.time()
        if now - last_display >= 1/fps:
            last_display = now

            if latest_frame is None:
                continue

           # ----------------------------------------
            # Decide what to show:
            # processed frame is valid if < 0.5s old
            # ----------------------------------------
            new_combined_boxes = []

            with lock:
                for angle, boxes in boxes_by_angle.items():
                    ts = timestamps_by_angle.get(angle, 0)
                    if now - ts < 0.5:
                        # store each box with its timestamp
                        for box in boxes:
                            new_combined_boxes.append((box, ts))

            # Remove old boxes (older than 0.5s)
            combined_boxes = [(box, ts) for box, ts in combined_boxes if now - ts < 0.2]

            # Add new boxes
            combined_boxes.extend(new_combined_boxes)

            # Extract just the box coordinates for drawing
            boxes_to_draw = [box for box, ts in combined_boxes]

            output_frame = helpers.add_boxes(latest_frame.copy(), boxes_to_draw)
            cv2.imshow("Camera (Haar)", output_frame)
            if out_haar != "":
                out_haar.write(output_frame)
            else:
                print("no haar frame found")

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.001)

except KeyboardInterrupt:
    pass

stop_flag = True
out_dnn.release()
out_haar.release()
cv2.destroyAllWindows()
