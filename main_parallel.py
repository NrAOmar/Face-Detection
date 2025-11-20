import cv2
import helpers
import haar_detector
import dnn_detector
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

# -------------------------------------------------------------------
#  Worker functions (executed in SEPARATE PROCESSES)
# -------------------------------------------------------------------

def haar_worker(frame, angle):
    """Runs Haar detection for one rotation."""
    frame_rotated, rot_matrix = helpers.rotate_image_without_cropping(frame, angle)
    faces = haar_detector.detect_faces(frame_rotated)
    boxes = helpers.construct_boxes(faces, rot_matrix)
    return boxes

def dnn_worker(frame, angle):
    """Runs DNN detection for one rotation."""
    frame_rotated, rot_matrix = helpers.rotate_image_without_cropping(frame, angle)
    faces = dnn_detector.detect_faces(frame_rotated)
    boxes = helpers.construct_boxes(faces, rot_matrix)
    return boxes


# -------------------------------------------------------------------
#  Main Program (must be inside __main__ for multiprocessing on macOS)
# -------------------------------------------------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    flag_rotation = True
    angle_step = 20 if flag_rotation else 360
    angles = [i for i in range(0, 360, angle_step)]

    camera_id = 1
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Could not open camera.")
        exit()

    scale, frame_width, frame_height = helpers.get_frame_size(camera_id)

    print("Running with ProcessPoolExecutor...")

    # Set number of processes
    executor = ProcessPoolExecutor(max_workers=len(angles)*2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Critical: pass a copy → avoids shared memory issues on macOS spawn
        frame_to_process = frame.copy()

        start = time.time()

        # Submit parallel Haar tasks
        haar_futures = [
            executor.submit(haar_worker, frame_to_process, a) for a in angles
        ]

        # Submit parallel DNN tasks
        dnn_futures = [
            executor.submit(dnn_worker, frame_to_process, a) for a in angles
        ]

        # Collect results
        # Flatten Haar
        boxes_haar_raw = [f.result() for f in haar_futures]
        boxes_haar = [box for sublist in boxes_haar_raw for box in sublist]

        # Flatten DNN
        boxes_dnn_raw = [f.result() for f in dnn_futures]
        boxes_dnn = [box for sublist in boxes_dnn_raw for box in sublist]

        print(f"Parallel time: {time.time() - start:.2f}s")

        # Draw results
        frame_haar = helpers.add_boxes(frame, boxes_haar, 0)
        frame_dnn  = helpers.add_boxes(frame, boxes_dnn, 0)

        cv2.imshow("HAAR (Parallel)", cv2.resize(frame_haar, (0, 0), fx=scale, fy=scale))
        cv2.imshow("DNN  (Parallel)", cv2.resize(frame_dnn, (0, 0), fx=scale, fy=scale))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()
