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


REC_PATH = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l", "w600k_r50.onnx")
rec_model = get_model(REC_PATH)
rec_model.prepare(ctx_id=0)  # CPU

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)



frame_id = 0
last_labeled = []

def align_crop_by_eyes(crop_bgr):
    h, w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return crop_bgr  # no landmarks, return as-is

    lm = res.multi_face_landmarks[0].landmark

    # Eye corners indices (common stable points)
    left = lm[33]   # left eye outer corner
    right = lm[263] # right eye outer corner

    xL, yL = int(left.x * w), int(left.y * h)
    xR, yR = int(right.x * w), int(right.y * h)

    angle = math.degrees(math.atan2(yR - yL, xR - xL))  # roll angle
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(crop_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
    return aligned

# def embed_from_box(frame_bgr, box, margin=0.20):
#     x1, y1, x2, y2 = to_xyxy(box)

#     w = x2 - x1
#     h = y2 - y1
#     mx = int(w * margin)
#     my = int(h * margin)

#     x1 = max(0, x1 - mx)
#     y1 = max(0, y1 - my)
#     x2 = min(frame_bgr.shape[1], x2 + mx)
#     y2 = min(frame_bgr.shape[0], y2 + my)

#     crop = frame_bgr[y1:y2, x1:x2]
#     crop = align_crop_by_eyes(crop)
#     if crop.size == 0:
#         return None

#     crop112 = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)

#     feat = rec_model.get_feat(crop112).flatten().astype(np.float32)
#     feat /= (np.linalg.norm(feat) + 1e-10)
#     return feat


def embed_from_box(frame_bgr, box, margin=0.20):
    t0 = time.perf_counter()

    x1, y1, x2, y2 = to_xyxy(box)
    w = x2 - x1
    h = y2 - y1
    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(frame_bgr.shape[1], x2 + mx)
    y2 = min(frame_bgr.shape[0], y2 + my)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    t1 = time.perf_counter()
    crop = align_crop_by_eyes(crop)
    t2 = time.perf_counter()

    crop112 = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
    t3 = time.perf_counter()

    feat = rec_model.get_feat(crop112).flatten().astype(np.float32)
    t4 = time.perf_counter()

    feat /= (np.linalg.norm(feat) + 1e-10)

    # print(
    #     f"crop:{(t1-t0)*1000:.1f}ms  "
    #     f"align:{(t2-t1)*1000:.1f}ms  "
    #     f"resize:{(t3-t2)*1000:.1f}ms  "
    #     f"feat:{(t4-t3)*1000:.1f}ms"
    # )
    return feat



def identify_boxes_id_only(frame_bgr, merged_boxes, known_mat, known_names,
                           threshold=0.38, hold_seconds=1):
    global last_good_name, last_good_sim, last_good_time

    labeled = []
    now = time.time()

    for mb in merged_boxes:
        t0 = time.perf_counter()
        emb = embed_from_box(frame_bgr, mb)
        t1 = time.perf_counter()
        if emb is None:
            continue

        sims = known_mat @ emb
        t2 = time.perf_counter()
        # print("embed:", (t1 - t0) * 1000, "ms  sim:", (t2 - t1) * 1000, "ms")

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        # Decide name for THIS frame
        if best_sim >= threshold:
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

        labeled.append((to_xyxy(mb), name, best_sim))

    return labeled





def to_xyxy(box):
    # Your merged boxes are dicts with x1,y1,x2,y2
    if isinstance(box, dict):
        return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

    # Fallback for list/tuple/np-array boxes
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return int(x1), int(y1), int(x2), int(y2)


# ---------- InsightFace ----------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(320, 320))  # faster than 640

print("has models:", hasattr(app, "models"))
if hasattr(app, "models"):
    print("model keys:", app.models.keys())


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

angle_step = 360
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
            

            if merged_boxes:
                if frame_id % RUN_EVERY == 0:
                    labeled = identify_boxes_id_only(latest_frame, merged_boxes, known_mat, known_names, THRESHOLD)
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
