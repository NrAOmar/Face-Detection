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
FLAG_HAAR = True # not tested
FLAG_DNN = True # not tested
FLAG_FUSION = True # not tested
FLAG_BIOMETRIC = False
FLAG_MULTIPLE_CAMERAS = True # TODO: change implementation to the old architecture if only 1 camera

ANGLE_STEP = 45
MAX_KEEP_TIME = 0.5
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
        boxes_filtered = helpers.dnn_filter_boxes(frame, boxes, margin= 0, conf_thr=0.2)
        
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
    last_good_name = "Unknown"
    last_good_sim = 0.0
    last_good_time = 0.0
    hold_seconds = 0 # TODO: what does this do? does the threading take its place?
    while not camera.stop_flag:
        for cam_id in range(num_cameras):
            with frames_lock:
                frame, fps = camera.get_latest_frame(cam_id)
            
            if frame is None:
                continue

            now = time.time()

            labeled_boxes2 = []
            mbs = boxes_final.copy()
            for mb in mbs:
                emb = helpers.embed_from_box(frame.copy(), mb)
                if emb is None:
                    continue
                sims = known_mat @ emb

                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])

                # Decide name for THIS frame
                if best_sim >= THRESHOLD:
                    name = known_names[best_idx]
                    # update "last good"
                    last_good_name = name
                    last_good_sim = best_sim
                    last_good_time = now
                else:
                    # If we recently had a confident name, hold it for a bit
                    if last_good_name != "Unknown" and (now - last_good_time) <= hold_seconds:
                        name = last_good_name
                        # Optional: show the last good sim instead of the low current sim
                        best_sim = float(last_good_sim)
                    else:
                        name = "Unknown"

                labeled_boxes2.append((helpers.to_xyxy(mb), name, best_sim)) # TODO: change this to only output the name not the box, then add the name to the merged_boxes somehow
                while len(labeled_boxes2) > len(mbs):
                    labeled_boxes2.pop(0)
                
                # print(labeled_boxes2)
                with results_lock:
                    # for i, labeled_box in enumerate(labeled_boxes):
                    #     if labeled_box[3] == camera_number:
                    #         labeled_boxes.pop(i)
                    
                    labeled_faces[camera_id] = labeled_boxes2


# Startup
if not FLAG_MULTIPLE_CAMERAS:
    camera.cameras_in_use = 1

camera.start_cameras()

if FLAG_BIOMETRIC:
    helpers.prepare_models()

    # Load known faces
    (known_embeddings, known_names) = helpers.load_known_faces()

    if len(known_embeddings) == 0:
        raise RuntimeError("No known faces loaded")

    known_mat = np.stack(known_embeddings).astype(np.float32)
    known_mat /= (np.linalg.norm(known_mat, axis=1, keepdims=True) + 1e-10)

    boxes_final = []

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


# Display loop (real-time, no lag)
last_display = [0.0] * num_cameras

try:
    while not camera.stop_flag:
        for cam_id in range(num_cameras):
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
            with results_lock:
                for (cid, model, _), (boxes, ts) in boxes_by_key.items():
                    if cid == cam_id and now - ts < MAX_KEEP_TIME:
                        if model != "haar_filtered":
                            view_all = helpers.add_boxes_all(view_all, boxes, camera.frame_sizes.get(cam_id))

                        if model != "haar_not_filtered":
                            boxes_all.extend(boxes)

            if FLAG_FUSION:
                boxes_final = helpers.merge_boxes_with_iou(boxes_all, 0.4)
            else:
                boxes_final = boxes_all

            view_final = helpers.add_boxes(frame.copy(), boxes_final, camera.frame_sizes.get(cam_id))

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
