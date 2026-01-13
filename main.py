import cv2
import threading
import time
import numpy as np

import camera
import helpers
import haar_detector
import dnn_detector
from plot_windows import display_frames_in_grid


# Configuration flags
FLAG_ROTATION = True
FLAG_HAAR = True
FLAG_DNN = True
FLAG_FUSION = True
FLAG_BIOMETRIC = True
FLAG_MULTIPLE_CAMERAS = False

ANGLE_STEP_HAAR = 120
ANGLE_STEP_DNN = 360
MAX_KEEP_TIME = 1
THRESHOLD = 0.38


# Shared state (results only)
results_lock = threading.Lock()

# (camera_id, model, angle) -> (boxes, timestamp)
boxes_by_key = {}

# camera_id -> list of (xyxy, name, sim)
labeled_faces = {}

# camera_id -> last frame (shared, read-only)
latest_frames = {}
frames_lock = threading.Lock()
starting_camera = camera.starting_camera

boxes_merged = {}

# Detection threads
def haar_worker(camera_id: int, angle: int):
    while not camera.stop_flag:
        with frames_lock:
            frame = latest_frames.get(camera_id)

        if frame is None:
            time.sleep(0.001)
            continue

        rotated, _ = helpers.rotate_image(frame, angle, camera.frame_sizes.get(camera_id))
        faces = haar_detector.detect_faces(rotated)
        boxes = helpers.construct_boxes(faces, angle, camera.frame_sizes.get(camera_id))
        
        if FLAG_FUSION:
            boxes_filtered = helpers.dnn_filter_boxes(frame, boxes, camera.frame_sizes.get(camera_id), conf_thr=0.2)
        else:
            boxes_filtered = boxes

        with results_lock:
            boxes_by_key[(camera_id, "haar_not_filtered", angle)] = (boxes, time.time())
            boxes_by_key[(camera_id, "haar_filtered", angle)] = (boxes_filtered, time.time())
        

def dnn_worker(camera_id: int, angle: int):
    while not camera.stop_flag:
        with frames_lock:
            frame = latest_frames.get(camera_id)

        if frame is None:
            time.sleep(0.001)
            continue

        rotated, _ = helpers.rotate_image(frame, angle, camera.frame_sizes.get(camera_id))
        faces, confs = dnn_detector.detect_faces(rotated)
        boxes = helpers.construct_boxes(faces, angle, camera.frame_sizes.get(camera_id), confs)

        with results_lock:
            boxes_by_key[(camera_id, "dnn", angle)] = (boxes, time.time())


def identify_worker(camera_id: int):
    while not camera.stop_flag:
    # for cam_id in range(starting_camera, num_cameras):
        with frames_lock:
            frame = latest_frames.get(camera_id)
            boxes_labeled = boxes_merged.get(camera_id)
        
        if frame is None or boxes_labeled is None:
            continue

        for mb in boxes_labeled:
            emb = helpers.embed_from_box(frame.copy(), mb)
            if emb is None:
                continue
            sims = known_mat @ emb

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            mb["similarity"] = best_sim
            if best_sim >= THRESHOLD:
                mb["name"] = known_names[best_idx]
            else:
                mb["name"] = "Unknown"

        labeled_faces[camera_id] = boxes_labeled

# Startup
if FLAG_MULTIPLE_CAMERAS:
    camera.start_cameras()
else:
    camera.start_cameras(1) # only start 1 camera

if FLAG_BIOMETRIC:
    helpers.prepare_models()

    # Load known faces
    (known_embeddings, known_names) = helpers.load_known_faces()

    if len(known_embeddings) == 0:
        raise RuntimeError("No known faces loaded")

    known_mat = np.stack(known_embeddings).astype(np.float32)
    known_mat /= (np.linalg.norm(known_mat, axis=1, keepdims=True) + 1e-10)


time.sleep(3.0)  # allow cameras to warm up

num_cameras = camera.cameras_in_use
if not FLAG_MULTIPLE_CAMERAS:
    starting_camera = camera.starting_camera
    num_cameras = starting_camera + 1

angles_haar = [0] if not FLAG_ROTATION else list(range(0, 360, ANGLE_STEP_HAAR))
angles_dnn  = [0] if not FLAG_ROTATION else list(range(0, 360, ANGLE_STEP_DNN))

for cam_id in range(starting_camera, num_cameras):
    for angle in angles_haar:
        if FLAG_HAAR:
            threading.Thread(
                target=haar_worker,
                args=(cam_id, angle),
                daemon=True
            ).start()

    for angle in angles_dnn:
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


# Display loop (real-time, no lag)
last_display = [0.0] * num_cameras

try:
    while not camera.stop_flag:
        for cam_id in range(starting_camera, num_cameras):
            frame, fps = camera.get_latest_frame(cam_id)

            if frame is None:
                continue

            with frames_lock:
                latest_frames[cam_id] = frame

            now = time.time()
            if now - last_display[cam_id] < 1.0 / fps:
                continue

            boxes_all = []
            view_all = frame.copy()
            total_faces = 0
            with results_lock:
                for (cid, model, _), (boxes, ts) in boxes_by_key.items():
                    if cid == cam_id and now - ts < MAX_KEEP_TIME:
                        if model != "haar_filtered":
                            (view_all, num_faces) = helpers.add_boxes_all(view_all, boxes, camera.frame_sizes.get(cam_id))
                            total_faces += num_faces

                        if model != "haar_not_filtered":
                            boxes_all.extend(boxes)
            
            cv2.putText(view_all, f"Faces: {total_faces}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            with frames_lock:
                boxes_merged[cam_id] = helpers.merge_boxes_with_iou(boxes_all, camera.frame_sizes.get(cam_id), 0.1, 0.5)
                view_final = helpers.add_boxes(frame.copy(), boxes_merged[cam_id], camera.frame_sizes.get(cam_id))

            if FLAG_BIOMETRIC:
                id_view = helpers.add_boxes(frame.copy(), labeled_faces.get(cam_id, []), camera.frame_sizes.get(cam_id))
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
