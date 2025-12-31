import cv2
import threading
import time
import helpers
import haar_detector
import dnn_detector
from plot_windows import display_frames_in_grid
import camera
from camera import stop_flag
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
stop_flag = False
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
    global latest_frame, stop_flag

    while not stop_flag:
        # if not is_angle_active(angle):
        #     time.sleep(0.02)   # small sleep to reduce CPU
        #     continue

        if latest_frame is None:
            time.sleep(0.001)
            continue

       

        tmp_start = time.time()

        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), angle)        
        faces = haar_detector.detect_faces(frame_rotated)
        boxes = helpers.construct_boxes(faces, angle, rotation_matrix)
        boxes = helpers.dnn_filter_boxes(latest_frame.copy(), boxes, margin= 0, conf_thr=0.2)
        # print(boxes)

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[("haar", angle)] = boxes
            timestamps_by_angle[("haar", angle)] = time.time()
        
        tmp_end = time.time()
        # print(angle)

def dnn_loop(angle):
    global latest_frame, display_rotated_frame, rotated_boxes, stop_flag
    
    while not stop_flag:
        # if not is_angle_active(angle):
        #     time.sleep(0.02)   # small sleep to reduce CPU
        #     continue

        if latest_frame is None:
            time.sleep(0.001)
            continue

        tmp_start = time.time()

        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), angle)        
        faces, conf_list = dnn_detector.detect_faces(frame_rotated)
        boxes = helpers.construct_boxes(faces, angle, rotation_matrix, conf_list)

        if (angle == angle_to_display):
            display_rotated_frame = frame_rotated.copy()
            rotated_boxes = helpers.construct_boxes_old(faces, angle)
        else:
            # display_rotated_frame = latest_frame.copy()
            rotated_boxes = []
        # print(rotated_boxes)

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[("dnn", angle)] = boxes
            timestamps_by_angle[("dnn", angle)] = time.time()
        
        tmp_end = time.time()
        # print(angle)

threads = []
threading.Thread(target=camera.camera_loop, daemon=True).start()
# threading.Thread(target=haar_loop, args=(angle_to_display,), daemon=True).start()
# threading.Thread(target=dnn_loop, args=(angle_to_display,), daemon=True).start()
# threading.Thread(target=haar_loop, args=(20,), daemon=True).start()

angle_step = 120
for angle in range(0, 360, angle_step):
    threading.Thread(target=haar_loop, args=(angle,), daemon=True).start()
    threading.Thread(target=dnn_loop, args=(angle,), daemon=True).start()

combined_boxes = []
last_display = time.time()
try:
    while True:
        now = time.time()
        latest_frame = camera.latest_frame
        
        if now - last_display >= 1/camera.fps:
            last_display = now

            if latest_frame is None:
                continue

            new_combined_boxes = []
            with lock:
                for key, boxes in boxes_by_angle.items():
                    ts = timestamps_by_angle.get(key, 0)
                    if now - ts < 0.4:
                        # store each box with its timestamp
                        for box in boxes:
                            new_combined_boxes.append((box, ts))

            # Remove old boxes (older than 0.5s)
            combined_boxes = [(box, ts) for box, ts in combined_boxes if now - ts < 0.4]
            # print(combined_boxes)

            # Add new boxes
            combined_boxes.extend(new_combined_boxes)
            # combined_boxes = combined_boxes[-12:]

            # Extract just the box coordinates for drawing
            boxes_to_draw = [box for box, ts in combined_boxes]
            
            # Get one merged box per face
            merged_boxes = helpers.merge_boxes_with_iou(boxes_to_draw, iou_threshold=0.4)
            merged_boxes = helpers.filter_boxes_by_confidence(merged_boxes, min_conf=0.5)
            # if len(merged_boxes) == 0:
            #     print("No merged boxes yet")
            #     continue
            
            HAAR_CONF = 0.5
            CONF_EPS = 1e-3       # tolerance for float compare
            DNN_VERIFY_THR = 0.6  # adjust if too strict

            verified = []
           

            # for b in merged_boxes:
            #     print(merged_boxes)
            #     conf = float(b.get("conf", 0.0))
            #     print("Yala n2ool bsmellah ")
            #     print(conf)
                
            #     # Haar-only: confidence is basically 0.5
            #     if abs(conf - HAAR_CONF) <= CONF_EPS:
            #         print("da5al gowa awl function")
            #         # Verify using DNN on the crop
            #         if helpers.dnn_confirms_box(latest_frame.copy(), b, margin=0.25, conf_thr=DNN_VERIFY_THR):
            #             print("da5al gowa tany functionwl haar got detected")
            #             verified.append(b)
            #         else:
            #             pass  # discard Haar false positive
            #             print("da5al gowa tany function bas el haar msh detected")
            #     else:
            #         # DNN or merged with DNN, keep directly
            #         verified.append(b)

            # merged_boxes = verified


            
            if merged_boxes:
                if frame_id % RUN_EVERY == 0:
                    labeled = helpers.identify_boxes_id_only(latest_frame.copy(), merged_boxes, known_mat, known_names, THRESHOLD,1)
                    last_labeled = labeled
                else:
                    labeled = last_labeled
            else:
                labeled = []
                last_labeled = []
                        
            identified_frame =latest_frame.copy(); 

            for (x1, y1, x2, y2), name, sim in labeled:
                cv2.rectangle(identified_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(identified_frame, f"{name} {sim:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # if len(merged_boxes) > 0:
            #     last_face_seen = now
            #      # Face is present, go back to only angle 0
            #     if not is_angle_active(0) or len(active_angles) != 1:
            #         set_active_angles([0])
            # else:   
            #     # No face right now
            #     no_face_time = now - last_face_seen

            #     # If we have not seen a face for 0.2s, enter search mode for 1.0s
            #     if no_face_time > 0.2 and now > search_mode_until:
            #         search_mode_until = now + 1.0
            #         set_active_angles([0, 90, 180, 270])
            
            # print("merged boxes")
            # print(merged_boxes)
            detected_all = helpers.add_boxes_all(latest_frame.copy(), boxes_to_draw, False)
            detected_final = helpers.add_boxes(latest_frame.copy(), merged_boxes)

            try:
                if (display_rotated_frame == None):
                    display_rotated_frame = latest_frame.copy()
                else:
                    display_rotated_frame = helpers.add_boxes_all(display_rotated_frame.copy(), rotated_boxes, False)
            except:
                display_rotated_frame = helpers.add_boxes_all(display_rotated_frame.copy(), rotated_boxes, False)

            # faces_haar = haar_detector.detect_faces(rotated_frame)
            # boxes_haar = helpers.construct_boxes(faces_haar, angle_to_display)
            # detected_rotated = helpers.add_boxes(rotated_frame.copy(), boxes_haar)
            
            # faces_dnn = haar_detector.detect_faces(rotated_frame)
            # boxes_dnn = helpers.construct_boxes(faces_dnn, (angle_to_display,))
            # detected_rotated = helpers.add_boxes(detected_rotated.copy(), boxes_dnn, False)
            display_frames_in_grid(
                ["Original", "Identified", "Rotated", "Detected Combined output", "Detected (HAAR & DNN)"],
                [latest_frame, identified_frame, display_rotated_frame, detected_all, detected_final]
            )
            frame_id += 1

            if camera.out_haar != "":
                camera.out_haar.write(detected_final)
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
