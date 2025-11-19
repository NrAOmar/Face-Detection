import cv2
import haar_detector
import dnn_detector
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
import helpers
import time

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

# display_frame_haar = ""
flag_quit = False

@njit(parallel=True)
def haar_parallel(frame_original, angle):
    frame_rotated, rotation_matrix = helpers.rotate_image_without_cropping(frame_original, angle)
    
    faces = haar_detector.detect_faces(frame_rotated)
    boxes = helpers.construct_boxes(faces, rotation_matrix)
    
    return boxes

@njit(parallel=True)
def dnn_parallel(frame_original, angle):
    frame_rotated, rotation_matrix = helpers.rotate_image_without_cropping(frame_original, angle)
    
    faces = haar_detector.detect_faces(frame_rotated)
    boxes = helpers.construct_boxes(faces, rotation_matrix)
    
    return boxes

# executor = ThreadPoolExecutor(max_workers=360/angle_step)
executor = ThreadPoolExecutor(max_workers=6)

while not flag_quit:
    # Capture frame
    ret, frame_original = cap.read()
    if not ret:
        break

    # frame_haar = frame.copy()
    # frame_dnn = frame.copy()
    
    angle = 0
    boxes_haar = []
    boxes_dnn = []
    future_haar = []
    future_dnn = []
    # while angle_main < 360:

    # HAAR

    start_time = time.time()

    # if flag_haar:
        # future_haar.append(executor.submit(haar_parallel, frame_original, angle))
        # future_haar.append(executor.submit(haar_parallel, frame_original, angle))
        # future_haar.append(executor.submit(haar_parallel, frame_original, angle))
    
    b0 = executor.submit(haar_parallel, frame_original, angle)
    b1 = executor.submit(haar_parallel, frame_original, angle)
    b2 = executor.submit(haar_parallel, frame_original, angle)
    # if flag_dnn:
        # future_dnn.append(executor.submit(dnn_parallel, frame_original, angle))
        # future_dnn.append(executor.submit(dnn_parallel, frame_original, angle))
        # future_dnn.append(executor.submit(dnn_parallel, frame_original, angle))
        
    b3 = executor.submit(dnn_parallel, frame_original, angle)
    b4 = executor.submit(dnn_parallel, frame_original, angle)
    b5 = executor.submit(dnn_parallel, frame_original, angle)

    # # Trial 1
    # for b in future_haar:
    #     boxes_haar.append(b.result())
    # for b in future_dnn:
    #     boxes_dnn.append(b.result())

    # # Trial 2
    # boxes_haar.append(future_haar[0].result())
    # boxes_haar.append(future_haar[1].result())
    # boxes_haar.append(future_haar[2].result())
    # boxes_dnn.append(future_dnn[0].result())
    # boxes_dnn.append(future_dnn[1].result())
    # boxes_dnn.append(future_dnn[2].result())

    # # Trial 3
    # b1 = future_haar[0].result()
    # b2 = future_haar[0].result()
    # b3 = future_dnn[0].result()
    # b4 = future_dnn[0].result()
    # b5 = future_dnn[0].result()

    # # Trial 4
    box0 = b0.result()
    box1 = b1.result()
    box2 = b2.result()
    box3 = b3.result()
    box4 = b4.result()
    box5 = b5.result()

    boxes_haar.append(box0)
    boxes_haar.append(box1)
    boxes_haar.append(box2)
    boxes_dnn.append(box3)
    boxes_dnn.append(box4)
    boxes_dnn.append(box5)

    print(box1)
    print(boxes_haar)
    print(boxes_dnn)

    end_time = time.time()
    print(f"CPU tasks completed in {end_time - start_time:.2f} seconds")
    
    frame_haar = helpers.add_boxes(frame_original, boxes_haar, angle % 360)
    frame_dnn = helpers.add_boxes(frame_original, boxes_dnn, angle % 360)
    
    display_frame_haar = cv2.resize(frame_haar, (0, 0), fx=scale, fy=scale)
    display_frame_dnn = cv2.resize(frame_dnn, (0, 0), fx=scale, fy=scale)

    cv2.imshow('Camera (HAAR)', display_frame_haar)
    cv2.imshow('Camera (DNN)', display_frame_dnn)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        flag_quit = True

cap.release()
out_dnn.release()
out_haar.release()
cv2.destroyAllWindows()

