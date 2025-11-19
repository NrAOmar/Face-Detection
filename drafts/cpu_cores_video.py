import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange

# Example numba-accelerated operation
@njit(parallel=True)
def brighten(frame):
    for i in prange(frame.shape[0]):
        for j in prange(frame.shape[1]):
            for k in range(3):
                frame[i, j, k] = min(frame[i, j, k] + 40, 255)
    return frame

# Other operations (OpenCV built-in = already fast)
def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def edges(frame):
    return cv2.Canny(frame, 120, 200)

cap = cv2.VideoCapture(1)

executor = ThreadPoolExecutor(max_workers=3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Submit parallel tasks
    f1 = executor.submit(brighten, frame.copy())
    f2 = executor.submit(to_gray, frame.copy())
    f3 = executor.submit(edges, frame.copy())

    # Wait for all results
    bright_frame = f1.result()
    gray_frame   = f2.result()
    edges_frame  = f3.result()

    # Display all results
    cv2.imshow("Original", frame)
    cv2.imshow("Brightened", bright_frame)
    cv2.imshow("Gray", gray_frame)
    cv2.imshow("Edges", edges_frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()