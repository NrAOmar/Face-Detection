import cv2
import threading
import time
import helpers
import haar_detector
import dnn_detector
from plot_windows import display_frames_in_grid
import camera
from camera import stop_flag
from insightface.app import FaceAnalysis
import os
import numpy as np

# Features to enable
flag_rotation = True
flag_haar = True
flag_dnn = True
flag_enhancement = False
flag_lowPassFilter = False
flag_biometric = False

latest_frame = None
display_id_frame = None
display_rotated_frame = None
processed_frame = None
stop_flag = False
angle_to_display = 90

boxes_by_angle = {}
timestamps_by_angle = {}
lock = threading.Lock()


def load_known_faces():
    global latest_frame, display_id_frame, stop_flag
    print("I am in face_id")
    # ---------- InsightFace ----------
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320, 320))  # faster than 640

    # ---------- Settings ----------
    THRESHOLD = 0.38
    SCALE = 0.5          # 0.5 means run model on half-resolution frame
    RUN_EVERY = 10       # run detection+recognition every N frames

    # ---------- Load known faces ----------
    known_embeddings = []
    known_names = []
    base_path="dataset"
    for person in os.listdir(base_path):
        print("I am in face_id for loop 1")
        person_path = os.path.join(base_path, person)
        if not os.path.isdir(person_path):
            print("I am in face_id for loop 2")
            continue

        print("I am in face_id for loop 3")
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            print("I am in face_id for loop 4")
            if img is None:
                continue

            print("I am in face_id for loop 5")
            faces = app.get(img)
            if len(faces) > 0:
                known_embeddings.append(faces[0].embedding.astype(np.float32))
                known_names.append(person)
            print("I am in face_id for loop 6")

    if len(known_embeddings) == 0:
        raise RuntimeError("No known faces loaded. Check your dataset folder and images.")

    print(f"Loaded {len(known_embeddings)} known faces")

    # Pre-normalize known embeddings for fast cosine
    known_mat = np.stack(known_embeddings, axis=0)
    known_mat /= (np.linalg.norm(known_mat, axis=1, keepdims=True) + 1e-10)

    last_results = []  # list of (bbox(x1,y1,x2,y2), name, sim)

    while not stop_flag:
        print("I am in face_id while loop")
        # Downscale for faster inference
        display_id_frame = latest_frame.copy()
        small = cv2.resize(display_id_frame, (0, 0), fx=SCALE, fy=SCALE)
        faces = app.get(small)

        results = []
        for f in faces:
            emb = f.embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-10)

            sims = known_mat @ emb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            name = "Unknown"
            if best_sim >= THRESHOLD:
                name = known_names[best_idx]

            # Scale bbox back to original frame coordinates
            x1, y1, x2, y2 = (f.bbox / SCALE).astype(int)
            results.append(((x1, y1, x2, y2), name, best_sim))

        last_results = results

    # Draw last results
    for (x1, y1, x2, y2), name, sim in last_results:
        cv2.rectangle(display_id_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_id_frame, f"{name} {sim:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def haar(angle):
    global latest_frame, display_rotated_frame, stop_flag
    tmp_start = time.time()

    frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), angle)        
    faces = haar_detector.detect_faces(frame_rotated)
    boxes = helpers.construct_boxes(faces, angle, rotation_matrix)

    if (angle == angle_to_display):
        display_rotated_frame = frame_rotated.copy()

    # Write results ONLY for this angle
    boxes_by_angle[("haar", angle)] = boxes
    timestamps_by_angle[("haar", angle)] = time.time()
    # with lock:
    #     boxes_by_angle[("haar", angle)] = boxes
    #     timestamps_by_angle[("haar", angle)] = time.time()
    
    tmp_end = time.time()
    # print(f"duration = {tmp_end - tmp_start}")

def haar_loop(angle):
    global latest_frame, display_rotated_frame, stop_flag

    while not stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        haar(angle)

def dnn_loop(angle):
    global latest_frame, stop_flag
    
    while not stop_flag:
        if latest_frame is None:
            time.sleep(0.001)
            continue

        tmp_start = time.time()

        frame_rotated, rotation_matrix = helpers.rotate_image(latest_frame.copy(), angle)        
        faces, conf_list = dnn_detector.detect_faces(frame_rotated)
        boxes = helpers.construct_boxes(faces, angle, rotation_matrix, conf_list)

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

# threading.Thread(target=load_known_faces, daemon=True).start()

angle_step = 20
for angle in range(angle_step, 360, angle_step):
    threading.Thread(target=haar_loop, args=(angle,), daemon=True).start()
    threading.Thread(target=dnn_loop, args=(angle,), daemon=True).start()
    continue

combined_boxes = []
last_display = time.time()

try:
    while True:
        now = time.time()
        latest_frame = camera.latest_frame
        
        # if now - last_display >= 1/camera.fps:
        if True:
            last_display = now

            if latest_frame is None:
                continue

            new_combined_boxes = []
            # haar(0)
            # dnn_loop(0)
            # print(boxes_by_angle)
            # with lock:
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
            
            # Get one merged box per face
            merged_boxes = helpers.merge_boxes_with_iou(boxes_to_draw, iou_threshold=0.4)
            merged_boxes = helpers.filter_boxes_by_confidence(merged_boxes, min_conf=0.4)
            # print(merged_boxes)
            
            # print("merged boxes")
            # print(merged_boxes)
            detected_all = helpers.add_boxes_all(latest_frame.copy(), boxes_to_draw, False)
            detected_final = helpers.add_boxes(latest_frame.copy(), merged_boxes)
            
            try:
                if (display_rotated_frame == None):
                    display_rotated_frame = latest_frame.copy()
                else:
                    display_rotated_frame = helpers.add_boxes_all(display_rotated_frame.copy(), boxes_to_draw, False)
            except:
                display_rotated_frame = helpers.add_boxes_all(display_rotated_frame.copy(), boxes_to_draw, False)

            # faces_haar = haar_detector.detect_faces(rotated_frame)
            # boxes_haar = helpers.construct_boxes(faces_haar, angle_to_display)
            # detected_rotated = helpers.add_boxes(rotated_frame.copy(), boxes_haar)
            
            # faces_dnn = haar_detector.detect_faces(rotated_frame)
            # boxes_dnn = helpers.construct_boxes(faces_dnn, (angle_to_display,))
            # detected_rotated = helpers.add_boxes(detected_rotated.copy(), boxes_dnn, False)
            display_frames_in_grid([
                "Original",
                # "Rotated",
                # "Detected Combined output",
                "Detected (HAAR & DNN)",
                # "Face ID",
                # "Detected Rotated"
            ],[
                latest_frame,
                # display_rotated_frame,
                # detected_all,
                detected_final,
                # display_id_frame,
                # detected_rotated
            ])

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
