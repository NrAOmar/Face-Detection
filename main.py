import cv2
import haar_detector
import dnn_detector
import helpers

# Features to enable
flag_rotation = True
flag_haar = True
flag_dnn = True
flag_enhancement = False
flag_lowPassFilter = False
flag_biometric = False
camera_in_use = 2 # start with camera in the lab

# Open camera (macOS AVFoundation). Try 2 then 1 then 0.
cap = cv2.VideoCapture(camera_in_use)
while not cap.isOpened() and camera_in_use > 0:
    camera_in_use -= 1
    cap = cv2.VideoCapture(camera_in_use)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get frame_dnn size
scale, frame_width, frame_height = helpers.get_frame_size(camera_in_use)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 20.0  # safe fallback
print(f"Recording at {fps} FPS")

# Video writer
out_haar = cv2.VideoWriter(
    'output_haar.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps/8,
    (frame_width, frame_height)
)
out_dnn = cv2.VideoWriter(
    'output_dnn.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps/8,
    (frame_width, frame_height)
)

print("Recording... Press 'q' to stop.")

angle_step = 45
if not flag_rotation:
    angle_step = 360

display_frame_haar = ""
flag_quit = False
while not flag_quit:
    # frame_haar = frame.copy()
    # frame_dnn = frame.copy()
    
    angle = 0
    boxes_haar_total = []
    boxes_dnn_total = []
    while angle < 360:
        # Capture frame
        ret, frame_original = cap.read()
        if not ret:
            break

        frame_rotated, rotation_matrix = helpers.rotate_image_without_cropping(frame_original.copy(), angle)
        
        # HAAR
        if flag_haar:
            faces = haar_detector.detect_faces(frame_rotated)
            boxes = helpers.construct_boxes(faces, rotation_matrix)
            for box in boxes:
                boxes_haar_total.append(box)
            frame_haar = helpers.add_boxes(frame_original, boxes_haar_total, angle % 360)

            # Show the frame
            display_frame_haar = cv2.resize(frame_haar, (0, 0), fx=scale, fy=scale)
            cv2.imshow('Camera (Haar)', display_frame_haar)
        
        # DNN
        if flag_dnn:
            faces = dnn_detector.detect_faces(frame_rotated)
            boxes = helpers.construct_boxes(faces, rotation_matrix)
            for box in boxes:
                boxes_dnn_total.append(box)
            frame_dnn = helpers.add_boxes(frame_original, boxes_dnn_total, angle % 360)

            # Show the frame
            display_frame_dnn = cv2.resize(frame_dnn, (0, 0), fx=scale, fy=scale)
            cv2.imshow('Camera (DNN)', display_frame_dnn)

        angle += angle_step
        if cv2.waitKey(1) & 0xFF == ord('q'):
            flag_quit = True

cap.release()
out_dnn.release()
out_haar.release()
cv2.destroyAllWindows()