import cv2
import queue
import threading
import time

# Public state (imported by main)
stop_flag = False

# Number of cameras you want to use (0,1,2,...)
# Example:
# 0 -> external iPhone camera
# 1 -> built-in webcam / phone
# 2 -> external camera
cameras_in_use = 2

# One queue per camera (latest frame only)
frame_queues = {}

# FPS per camera (best-effort)
fps_by_camera = {}

# Optional: frame size per camera
frame_sizes = {}

# Internal camera thread
def _camera_thread(camera_id: int):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print(f"[Camera {camera_id}] ERROR: Could not open camera")
        return

    # Try to get FPS (often unreliable on macOS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 20.0  # safe fallback

    fps_by_camera[camera_id] = fps

    # Read one frame to get size
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        frame_sizes[camera_id] = (w, h)
    else:
        frame_sizes[camera_id] = None

    # Queue holds ONLY the latest frame (low latency)
    q = queue.Queue(maxsize=1)
    frame_queues[camera_id] = q

    print(f"[Camera {camera_id}] Started ({fps:.1f} FPS), Frame size: ({w}, {h})")

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue

        # Always keep the newest frame
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass

        q.put(frame)

    cap.release()
    print(f"[Camera {camera_id}] Stopped")


# Public API
def start_cameras():
    """
    Starts one thread per camera.
    Call ONCE from main before using frame_queues.
    """
    for cam_id in range(cameras_in_use):
        t = threading.Thread(
            target=_camera_thread,
            args=(cam_id,),
            daemon=True
        )
        t.start()


def get_latest_frame(camera_id: int):
    """
    Non-blocking frame fetch.
    Returns (frame, fps) or (None, None)
    """
    q = frame_queues.get(camera_id)
    if q is None or q.empty():
        return None, None

    try:
        frame = q.get_nowait()
    except queue.Empty:
        return None, None

    return frame, fps_by_camera.get(camera_id, 20.0)
