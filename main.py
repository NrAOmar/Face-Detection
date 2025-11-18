import cv2
import haar_detector
import dnn_detector
import helpers
import math

camera_in_use = 2 # start with camera in the lab

# Open camera (macOS AVFoundation). Try 2 then 1 then 0.
cap = cv2.VideoCapture(camera_in_use, cv2.CAP_AVFOUNDATION)
while not cap.isOpened() and camera_in_use > 0:
    camera_in_use -= 1
    cap = cv2.VideoCapture(camera_in_use, cv2.CAP_AVFOUNDATION)

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
display_frame_haar = ""
break_flag = False
while not break_flag:

    # # Capture frame
    # frame_haar = frame.copy()
    # frame_dnn = frame.copy()
    
    angle = 0
    boxes_total = []
    while angle < 360:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_rotated, rotation_matrix = helpers.rotate_image(frame.copy(), angle)
        
        # HAAR
        faces = haar_detector.detect_faces(frame_rotated.copy())
        boxes = helpers.construct_boxes(faces, rotation_matrix)
        for box in boxes:
            boxes_total.append(box)
        frame_haar = helpers.add_boxes(frame.copy(), boxes_total, angle % 360)

        ## Show the frame
        display_frame_haar = cv2.resize(frame_haar, (0, 0), fx=scale, fy=scale)
        cv2.imshow('Camera (Haar)', display_frame_haar)
        
        # debugging
        # if (angle == 280):
            # boxes2 = helpers.construct_boxes(faces)
            # frame_rotated2 = helpers.add_boxes(frame_rotated.copy(), boxes2, angle % 360)
            # display_frame_haar_rotated = cv2.resize(frame_rotated2, (0, 0), fx=scale, fy=scale)
            # cv2.imshow('Camera (Rotated)', display_frame_haar_rotated)
        # if (angle == 360):
            ## Store the frames
        if out_haar != "":
            out_haar.write(display_frame_haar)
        else:
            print("no haar frame found")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break_flag = True
        


        # DNN
        # frame_dnn = dnn_detector.detect_faces(frame_rotated.copy(), out_dnn)
        
        # ## Show the frame
        # display_frame_dnn = cv2.resize(frame_dnn, (0, 0), fx=scale, fy=scale)
        # cv2.imshow('Camera (DNN)', display_frame_dnn)
        
        angle += angle_step

    # ## Store the frames
    # if out_haar != "":
    #     out_haar.write(display_frame_haar)
    # else:
    #     print("no haar frame found")

    # if out_dnn != "":
    #     out_dnn.write(display_frame_dnn)
    

cap.release()
out_dnn.release()
out_haar.release()
cv2.destroyAllWindows()