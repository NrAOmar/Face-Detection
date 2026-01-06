import cv2
import threading
import time
import numpy as np

import camera
import helpers
import haar_detector
import dnn_detector
from plot_windows import display_frames_in_grid
from helpers import known_embeddings, known_names


# =================================================
# Configuration flags
# =================================================
FLAG_ROTATION = True
FLAG_HAAR = True
FLAG_DNN = True
FLAG_FUSION = True
FLAG_BIOMETRIC = True
FLAG_MULTIPLE_CAMERAS = True

ANGLE_STEP = 120
MAX_KEEP_TIME = 0.5
THRESHOLD = 0.38


# =================================================
# Shared state (results only)
# =================================================
results_lock = threading.Lock()

# (camera_id, model, angle) -> (boxes, timestamp)
boxes_by_key = {}

# camera_id -> list of (xyxy, name, sim)
labeled_faces = {}


# =================================================
# Load known faces
# =================================================
helpers.load_known_faces()

if len(known_embeddings) == 0:
    raise RuntimeError("No known faces loaded")

known_mat = np.stack(known_embeddings).astype(np.float32)
known_mat /= (np.linalg.norm(known_mat, axis=1, keepdims=True) + 1e-10)


# =================================================
# Detection threads
# =================================================
def haar_worker(camera_id: int, angle: int):
    while not camera.stop_flag:
        frame, _ = camera.get_latest_frame(camera_id)
        if frame is None:
            time.sleep(0.001)
            continue

        rotated, _ = helpers.rotate_image(frame, angle)
        faces = haar_detector.detect_faces(rotated)
        boxes = helpers.construct_boxes(faces, angle)

        with results_lock:
            boxes_by_key[(camera_id, "haar", angle)] = (boxes, time.time())


def dnn_worker(camera_id: int, angle: int):
    while not camera.stop_flag:
        frame, _ = camera.get_latest_frame(camera_id)
        if frame is None:
            time.sleep(0.001)
            continue

        rotated, _ = helpers.rotate_image(frame, angle)
        faces, confs = dnn_detector.detect_faces(rotated)
        boxes = helpers.construct_boxes(faces, angle, confs)

        with results_lock:
            boxes_by_key[(camera_id, "dnn", angle)] = (boxes, time.time())


def identify_worker(camera_id: int):
    last_results = []

    while not camera.stop_flag:
        frame, _ = camera.get_latest_frame(camera_id)
        if frame is None:
            time.sleep(0.001)
            continue

        with results_lock:
            boxes = last_results.copy()

        labeled = []
        for box in boxes:
            emb = helpers.embed_from_box(frame, box)
            if emb is None:
                continue

            sims = known_mat @ emb
            idx = int(np.argmax(sims))
            sim = float(sims[idx])

            name = known_names[idx] if sim >= THRESHOLD else "Unknown"
            labeled.append((helpers.to_xyxy(box), name, sim))

        with results_lock:
            labeled_faces[camera_id] = labeled

        time.sleep(0.01)


# =================================================
# Startup
# =================================================
if not FLAG_MULTIPLE_CAMERAS:
    camera.cameras_in_use = 1

camera.start_cameras()

time.sleep(1.0)  # allow cameras to warm up

num_cameras = camera.cameras_in_use
angles = [0] if not FLAG_ROTATION else list(range(0, 360, ANGLE_STEP))

for cam_id in range(num_cameras):
    for angle in angles:
        if FLAG_HAAR:
            threading.Thread(
                target=haar_worker,
                args=(cam_id, angle),
                daemon=True
            ).start()

        if FLAG_DNN:
            threading.Thread(
                target=dnn_worker,
                args=(cam_id, angle),
                daemon=True
            ).start()

    if FLAG_BIOMETRIC:
        threading.Thread(
            target=identify_worker,
            args=(cam_id,),
            daemon=True
        ).start()


# =================================================
# Display loop (real-time, no lag)
# =================================================
last_display = [0.0] * num_cameras

try:
    while not camera.stop_flag:
        for cam_id in range(num_cameras):
            frame, fps = camera.get_latest_frame(cam_id)
            if frame is None:
                continue

            now = time.time()
            if now - last_display[cam_id] < 1.0 / fps:
                continue

            boxes_all = []

            with results_lock:
                for (cid, _, _), (boxes, ts) in boxes_by_key.items():
                    if cid == cam_id and now - ts < MAX_KEEP_TIME:
                        boxes_all.extend(boxes)

            if FLAG_FUSION:
                boxes_final = helpers.merge_boxes_with_iou(boxes_all, 0.4)
            else:
                boxes_final = boxes_all

            view_all = helpers.add_boxes_all(frame.copy(), boxes_all)
            view_final = helpers.add_boxes(frame.copy(), boxes_final)

            if FLAG_BIOMETRIC:
                id_view = frame.copy()
                for (x1, y1, x2, y2), name, sim in labeled_faces.get(cam_id, []):
                    cv2.rectangle(id_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        id_view,
                        f"{name} {sim:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            else:
                id_view = frame

            display_frames_in_grid([
                ("Original", cam_id, frame),
                ("All detections", cam_id, view_all),
                ("Fused", cam_id, view_final),
                ("Identified", cam_id, id_view),
            ])

            last_display[cam_id] = now

        if cv2.waitKey(1) & 0xFF == 27:
            camera.stop_flag = True

        time.sleep(0.001)

except KeyboardInterrupt:
    camera.stop_flag = True

cv2.destroyAllWindows()
