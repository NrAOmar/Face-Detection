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
flag_enhancement = False
flag_lowPassFilter = False
flag_biometric = False

latest_frame = None
rotated_boxes = []
display_rotated_frame = None
processed_frame = None
angle_to_display = 90

boxes_by_angle = {}
timestamps_by_angle = {}
lock = threading.Lock()
frame_id = 0
last_labeled = []


# ---------- Settings ----------
THRESHOLD = 0.38
SCALE = 1          # 0.5 means run model on half-resolution frame
RUN_EVERY =  10    # run detection+recognition every N fram


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
        boxes = helpers.construct_boxes(faces, angle, rotation_matrix)
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
        boxes = helpers.construct_boxes(faces, angle, rotation_matrix, conf_list)

        if (angle == angle_to_display):
            display_rotated_frame = frame_rotated.copy()
            rotated_boxes = helpers.construct_boxes_old(faces, angle)
        else:
            rotated_boxes = []

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[("dnn", angle)] = (boxes, time.time())
        
threads = []
threading.Thread(target=camera.camera_loop, daemon=True).start()
# threading.Thread(target=haar_loop, args=(angle_to_display,), daemon=True).start()
# threading.Thread(target=dnn_loop, args=(angle_to_display,), daemon=True).start()
# threading.Thread(target=haar_loop, args=(20,), daemon=True).start()

angle_step = 120
for angle in range(0, 360, angle_step):
    if (flag_haar):
        threading.Thread(target=haar_loop, args=(angle,), daemon=True).start()
    if (flag_dnn):
        threading.Thread(target=dnn_loop, args=(angle,), daemon=True).start()

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
                if now - ts < 0.4:
                    boxes_by_angle[key] = (boxes, ts)
                    boxes_to_draw.extend(boxes)
        
        # Get one merged box per face
        merged_boxes = helpers.merge_boxes_with_iou(boxes_to_draw, iou_threshold=0.4)
        merged_boxes = helpers.filter_boxes_by_confidence(merged_boxes, min_conf=0.5)
        
        HAAR_CONF = 0.5
        CONF_EPS = 1e-3       # tolerance for float compare
        DNN_VERIFY_THR = 0.6  # adjust if too strict

        verified = []

        if merged_boxes:
            labeled = helpers.identify_boxes_id_only(latest_frame.copy(), merged_boxes, known_mat, known_names, THRESHOLD,1)
            last_labeled = labeled
        else:
            labeled = []
            last_labeled = []
                    
        identified_frame = latest_frame.copy(); 

        for (x1, y1, x2, y2), name, sim in labeled:
            cv2.rectangle(identified_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(identified_frame, f"{name} {sim:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # print("merged boxes")
        # print(merged_boxes)
        detected_all = helpers.add_boxes_all(latest_frame.copy(), boxes_to_draw, False)
        detected_final = helpers.add_boxes(latest_frame.copy(), merged_boxes)

        if (display_rotated_frame is None):
            display_rotated_frame = latest_frame.copy()
        else:
            display_rotated_frame = helpers.add_boxes_all(display_rotated_frame.copy(), rotated_boxes, False)

        display_frames_in_grid(
            ["Original", "Identified", "Rotated", "Detected Combined output", "Detected (HAAR & DNN)"],
            [latest_frame, identified_frame, display_rotated_frame, detected_all, detected_final]
        )

        if camera.out_haar != "":
            camera.out_haar.write(detected_final)
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
