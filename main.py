import cv2
import threading
import time
import helpers
import haar_detector
import dnn_detector
from plot_windows import display_frames_in_grid
import camera
from camera import stop_flag

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

def haar_loop(angle):
    global latest_frame, stop_flag

    while not stop_flag:
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
            merged_boxes = helpers.filter_boxes_by_confidence(merged_boxes, min_conf=0.6)
            
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

            display_frames_in_grid([
                "Original",
                "Rotated",
                "Detected Combined output",
                "Detected (HAAR & DNN)",
                # "Detected Rotated"
            ],[
                latest_frame,
                display_rotated_frame,
                detected_all,
                detected_final,
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
