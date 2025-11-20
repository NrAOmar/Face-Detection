import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
import helpers
import haar_detector
import dnn_detector

# ------------------------------------------------------------
# Worker (runs in separate process)
# ------------------------------------------------------------
def worker_process(shm_name, shape, dtype, angle):
    """
    Reads shared frame, rotates it, runs Haar + DNN,
    returns merged boxes.
    """
    shm = SharedMemory(name=shm_name)
    frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # Rotate frame
    rotated, rot_matrix = helpers.rotate_image_without_cropping(frame, angle)

    # Haar
    faces_haar = haar_detector.detect_faces(rotated)
    boxes_haar = helpers.construct_boxes(faces_haar, rot_matrix)

    # DNN
    faces_dnn = dnn_detector.detect_faces(rotated)
    boxes_dnn = helpers.construct_boxes(faces_dnn, rot_matrix)

    return boxes_haar, boxes_dnn


# ------------------------------------------------------------
# Main Program
# ------------------------------------------------------------
if __name__ == "__main__":

    flag_rotation = True
    angle_step = 90 if flag_rotation else 360
    angles = [i for i in range(0, 360, angle_step)]

    camera_id = 1
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Could not open camera.")
        exit()

    scale, frame_width, frame_height = helpers.get_frame_size(camera_id)

    print("Running optimized parallel version...")

    # IMPORTANT: create the pool ONLY inside __main__
    executor = ProcessPoolExecutor(max_workers=len(angles))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # --------------------------------------------------------
        # Create shared memory for the frame (zero-copy sharing)
        # --------------------------------------------------------
        dtype = frame.dtype
        shape = frame.shape
        shm = SharedMemory(create=True, size=frame.nbytes)
        np_shm_frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        np_shm_frame[:] = frame[:]  # copy frame into shared memory

        # --------------------------------------------------------
        # Submit parallel angle tasks
        # --------------------------------------------------------
        futures = [
            executor.submit(worker_process, shm.name, shape, dtype, angle)
            for angle in angles
        ]

        # Wait for all workers to finish
        results = [f.result() for f in futures]

        # Free shared memory
        shm.close()
        shm.unlink()

        # --------------------------------------------------------
        # Merge Haar + DNN outputs from all angles
        # --------------------------------------------------------
        boxes_haar = []
        boxes_dnn = []

        for haar_list, dnn_list in results:
            boxes_haar.extend(haar_list)
            boxes_dnn.extend(dnn_list)

        print(f"Parallel time: {time.time() - start:.2f}s")

        # --------------------------------------------------------
        # Draw boxes
        # --------------------------------------------------------
        frame_haar = helpers.add_boxes(frame, boxes_haar, 0)
        frame_dnn  = helpers.add_boxes(frame, boxes_dnn, 0)

        cv2.imshow(
            "HAAR (Parallel)",
            cv2.resize(frame_haar, (0, 0), fx=scale, fy=scale)
        )
        cv2.imshow(
            "DNN  (Parallel)",
            cv2.resize(frame_dnn, (0, 0), fx=scale, fy=scale)
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()
