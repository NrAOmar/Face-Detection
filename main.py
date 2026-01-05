import cv2
import threading
import time
import helpers
import haar_detector
import dnn_detector
from plot_windows import display_frames_in_grid
import camera
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os
import mediapipe as mp
import math
from helpers import known_embeddings, known_names


# Features to enable
flag_rotation = True
flag_haar = True
flag_dnn = True
flag_modelsFusion = True    # TODO: this flag is currently not working for false,
                            # maybe because merged_boxes is not the same format as boxes_to_draw
flag_biometric = True
# flag_enhancement = False # TODO: add enhancement to the input images
# flag_lowPassFilter = False

latest_frame = None
# processed_frame = None
display_rotated_frame = None
display_detected_all = None
display_detected_final = None
identified_frame = None

angle_step = 45
angle_to_display = angle_step # show first rotation step
MAX_KEEP_TIME = 0.5

labeled_boxes = []
rotated_boxes = []
boxes_by_angle = {}

lock = threading.Lock()


# HAAR_CONF = 0.5
# CONF_EPS = 1e-3       # tolerance for float compare
# DNN_VERIFY_THR = 0.6  # adjust if too strict
THRESHOLD = 0.38

helpers.load_known_faces()

if len(known_embeddings) == 0:
    raise RuntimeError("No known faces loaded. Check your dataset folder and images.")

# known_embeddings: list of (512,) vectors
# known_names: same length
known_mat = np.stack(known_embeddings).astype(np.float32)
known_mat /= (np.linalg.norm(known_mat, axis=1, keepdims=True) + 1e-10)


def haar_loop(angle):
    global latest_frame

    while not camera.stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), angle)        
        faces = haar_detector.detect_faces(frame_rotated)
        boxes = helpers.construct_boxes(faces, angle)
        boxes = helpers.dnn_filter_boxes(latest_frame.copy(), boxes, margin= 0, conf_thr=0.2)

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[("haar", angle)] = (boxes, time.time())

def dnn_loop(angle):
    global latest_frame, display_rotated_frame, rotated_boxes
    
    while not camera.stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), angle)        
        faces, conf_list = dnn_detector.detect_faces(frame_rotated)
        boxes = helpers.construct_boxes(faces, angle, conf_list)

        if (angle == angle_to_display):
            display_rotated_frame = frame_rotated.copy()
            rotated_boxes = boxes.copy()
        else:
            rotated_boxes = []

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[("dnn", angle)] = (boxes, time.time())
        
def identify_faces():
    global latest_frame, merged_boxes, labeled_boxes

    last_good_name = "Unknown"
    last_good_sim = 0.0
    last_good_time = 0.0
    hold_seconds = 0.5 # TODO: what does this do? does the threading take its place?
    while not camera.stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        now = time.time()

        # labeled_boxes = []
        for mb in merged_boxes:
            emb = helpers.embed_from_box(latest_frame.copy(), mb)
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

            with lock:
                labeled_boxes.append((helpers.to_xyxy(mb), name, best_sim)) # TODO: change this to only output the name not the box, then add the name to the merged_boxes somehow
                while len(labeled_boxes) > len(merged_boxes):
                    labeled_boxes.pop(0)

threading.Thread(target=camera.camera_loop, daemon=True).start()

if not flag_rotation:
    angle_step = 360

for angle in range(0, 360, angle_step):
    if (flag_haar):
        threading.Thread(target=haar_loop, args=(angle,), daemon=True).start()
    if (flag_dnn):
        threading.Thread(target=dnn_loop, args=(angle,), daemon=True).start()

if flag_biometric:
    threading.Thread(target=identify_faces, daemon=True).start() # TODO: change to create a thread for each face

last_display = time.time()
try:
    while not camera.stop_flag:
        now = time.time()
        latest_frame = camera.latest_frame
        
        if latest_frame is None:
            continue

        if now - last_display < 1/camera.fps:
            continue

        last_display = now

        boxes_to_draw = []
        with lock:
            for key, (boxes, ts) in boxes_by_angle.items():
                if now - ts < MAX_KEEP_TIME:
                    boxes_by_angle[key] = (boxes, ts)
                    boxes_to_draw.extend(boxes)
        
        if flag_modelsFusion: # Get one merged box per face
            merged_boxes = helpers.merge_boxes_with_iou(boxes_to_draw, iou_threshold=0.4)
            merged_boxes = helpers.filter_boxes_by_confidence(merged_boxes, min_conf=0.5)
        else:
            merged_boxes = boxes_to_draw

        display_detected_all = helpers.add_boxes_all(latest_frame.copy(), boxes_to_draw)
        display_detected_final = helpers.add_boxes(latest_frame.copy(), merged_boxes) # TODO: change format of merged_boxes to be same as boxes_to_draw
        display_rotated_frame = helpers.add_boxes_all(display_rotated_frame, rotated_boxes)

        if flag_biometric:
            identified_frame = latest_frame.copy(); 
            for (x1, y1, x2, y2), name, sim in labeled_boxes:
                cv2.rectangle(identified_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(identified_frame, f"{name} {sim:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        display_frames_in_grid(
            ["Original", "Rotated", "Detected Combined output", "Detected (HAAR & DNN)", "Identified"],
            [latest_frame, display_rotated_frame, display_detected_all, display_detected_final, identified_frame]
        )

        if camera.out_haar != "":
            camera.out_haar.write(display_detected_final)
        else:
            print("no haar frame found")

        if cv2.waitKey(1) & 0xFF == 27:
            camera.stop_flag = True

        time.sleep(0.001)
except KeyboardInterrupt:
    pass

camera.out_dnn.release()
camera.out_haar.release()
cv2.destroyAllWindows()
