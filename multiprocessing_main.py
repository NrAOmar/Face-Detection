import cv2
import numpy as np
import haar_detector
import dnn_detector
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import itertools
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

ret, frame_original = cap.read()

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

angle_step = 90
if not flag_rotation:
    angle_step = 360

# display_frame_haar = ""
flag_quit = False

# @njit(parallel=True)
def haar_parallel_shared(angle):
    # Access the frame from the shared dictionary (requires dictionary object access)
    start_time = time.time()
    frame_original = shared_frame_dict['frame']
    # frame, angle = args
    
    frame_rotated, rotation_matrix = helpers.rotate_image_without_cropping(frame, angle)
    
    faces = haar_detector.detect_faces(frame_rotated)
    boxes = helpers.construct_boxes(faces, rotation_matrix)
    
    end_time = time.time()
    
    return boxes, angle, end_time - start_time

# @njit(parallel=True)
def dnn_parallel_shared(angle):
    # Access the frame from the shared dictionary (requires dictionary object access)
    start_time = time.time()
    frame_original = shared_frame_dict['frame']
    # frame, angle = args

    frame_rotated, rotation_matrix = helpers.rotate_image_without_cropping(frame, angle)
    
    faces = dnn_detector.detect_faces(frame_rotated)
    boxes = helpers.construct_boxes(faces, rotation_matrix)
    
    end_time = time.time()
    return boxes, angle, end_time - start_time

# executor = ThreadPoolExecutor(max_workers=360/angle_step)
if __name__ == '__main__':
# 1. Start a Manager
    with multiprocessing.Manager() as manager:
        # Create a shared dictionary to hold the frame
        shared_frame_dict = manager.dict() 
        # Create the Pool (Workers will be able to access the Manager)
        with multiprocessing.Pool() as pool:
            while not flag_quit:
                # Capture frame
                ret, frame_original = cap.read()
                if not ret:
                    break

                # 2. Update the Shared Memory with the new frame
                # NOTE: Even Manager.dict() requires pickling/copying the data 
                # when assigning the value, but it's often more efficient 
                # than passing as a task argument.
                shared_frame_dict['frame'] = frame_original.copy()
                
                angles = range(0, 360, angle_step)
                # task_arguments = list(itertools.product([frame_original], angles))

                start_time = time.time()

                # with Pool(processes=5) as pool:
                # results_haar = pool.imap(haar_parallel, task_arguments)
                # results_dnn = pool.imap(dnn_parallel, task_arguments)
                results_haar = pool.imap(dnn_parallel_shared, angles)
                results_dnn = pool.imap(dnn_parallel_shared, angles)
                
                for boxes, angle, duration in results_haar:
                    print(f" got boxes {boxes} at angle {angle} in {duration:.2f}s")
                    frame_haar = helpers.add_boxes(frame_original, boxes, angle % 360)

                for boxes, angle, duration in results_dnn:
                    print(f" got boxes {boxes} at angle {angle} in {duration:.2f}s")
                    frame_dnn = helpers.add_boxes(frame_original, boxes, angle % 360)

                end_time = time.time()
                print(f"CPU tasks completed in {end_time - start_time:.2f} seconds")
                
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

