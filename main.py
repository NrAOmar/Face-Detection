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
import os


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

# active_angles = set([0])         # angles currently enabled
# search_mode_until = 0.0          # time until which search mode is active
# last_face_seen = time.time()     # last time we saw at least 1 merged face

# def is_angle_active(angle):
#     with lock:
#         return angle in active_angles

# def set_active_angles(angles):
#     with lock:
#         active_angles.clear()
#         active_angles.update(angles)

frame_id = 0
last_labeled = []


def to_xyxy(box):
    # Your merged boxes are dicts with x1,y1,x2,y2
    if isinstance(box, dict):
        return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

    # Fallback for list/tuple/np-array boxes
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return int(x1), int(y1), int(x2), int(y2)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter + 1e-10
    return inter / union




# ---------- InsightFace ----------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(320, 320))  # faster than 640

# ---------- Settings ----------
THRESHOLD = 0.38
SCALE = 0.5          # 0.5 means run model on half-resolution frame
RUN_EVERY = 3       # run detection+recognition every N frames

# ---------- Load known faces ----------
known_embeddings = []
known_names = []

def load_known_faces(base_path="dataset"):
    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) > 0:
                known_embeddings.append(faces[0].embedding.astype(np.float32))
                known_names.append(person)

    print(f"Loaded {len(known_embeddings)} known faces")

load_known_faces()

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

        # Write results ONLY for this angle
        with lock:
            boxes_by_angle[("haar", angle)] = boxes
            timestamps_by_angle[("haar", angle)] = time.time()
        
        tmp_end = time.time()
        # print(f"duration = {tmp_end - tmp_start}")

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
        # print(f"duration = {tmp_end - tmp_start}")

threads = []
threading.Thread(target=camera.camera_loop, daemon=True).start()
# threading.Thread(target=haar_loop, args=(angle_to_display,), daemon=True).start()
# threading.Thread(target=dnn_loop, args=(angle_to_display,), daemon=True).start()
# threading.Thread(target=haar_loop, args=(20,), daemon=True).start()

angle_step = 90
for angle in range(0, 360, angle_step):
    threading.Thread(target=haar_loop, args=(angle,), daemon=True).start()
    threading.Thread(target=dnn_loop, args=(angle,), daemon=True).start()

combined_boxes = []
last_display = time.time()

def identify_merged_boxes(frame, merged_boxes, app, known_mat, known_names,
                          sim_threshold=0.38, iou_threshold=0.20):
    ins_faces = app.get(frame)  # InsightFace detects + embeddings on full frame

    labeled = []
    for mb in merged_boxes:
        mb_xyxy = to_xyxy(mb)

        best_face = None
        best_iou = 0.0
        for f in ins_faces:
            fb = tuple(map(int, f.bbox))  # (x1,y1,x2,y2)
            s = iou(mb_xyxy, fb)
            if s > best_iou:
                best_iou = s
                best_face = f

        name = "Unknown"
        best_sim = 0.0

        if best_face is not None and best_iou >= iou_threshold:
            emb = best_face.embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-10)

            sims = known_mat @ emb
            idx = int(np.argmax(sims))
            best_sim = float(sims[idx])

            if best_sim >= sim_threshold:
                name = known_names[idx]

        labeled.append((mb_xyxy, name, best_sim, best_iou))

    return labeled


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

            # Add new boxes
            combined_boxes.extend(new_combined_boxes)
            # combined_boxes = combined_boxes[-12:]

            # Extract just the box coordinates for drawing
            boxes_to_draw = [box for box, ts in combined_boxes]
            
            # Get one merged box per face
            merged_boxes = helpers.merge_boxes_with_iou(boxes_to_draw, iou_threshold=0.4)
            merged_boxes = helpers.filter_boxes_by_confidence(merged_boxes, min_conf=0.6)
            # if len(merged_boxes) == 0:
            #     print("No merged boxes yet")
            #     continue
            

            if len(merged_boxes) > 0:
                if frame_id % RUN_EVERY == 0:
                    labeled = identify_merged_boxes(latest_frame, merged_boxes, app, known_mat, known_names,
                                                sim_threshold=THRESHOLD, iou_threshold=0.20)
                    last_labeled = labeled
                else:
                    labeled = last_labeled
            else:
                labeled = []
                last_labeled = []
                        
            identified_frame =latest_frame.copy(); 

            for (x1, y1, x2, y2), name, sim, box_iou in labeled:
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
