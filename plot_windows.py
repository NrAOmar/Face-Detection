import cv2
import math
from screeninfo import get_monitors
import camera

def resize_with_aspect_ratio(image, target_w, target_h):
    """
    Resize an image to fit within (target_w x target_h) while keeping aspect ratio.
    Returns the resized image and its new dimensions.
    """
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, new_w, new_h

def display_frames_in_grid(frames_to_display, title_bar_height=29, margin=0):
    """
    Displays frames in a grid layout with screen fitting and proper handling
    of title bar height so no window goes outside the screen.
    """
    # Screen size
    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height

    # Arrange grid
    n = len(frames_to_display) * camera.cameras_in_use
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Deduct margins + title bars exactly
    usable_h = screen_h - ((rows + 1) * margin) - (rows * title_bar_height)
    usable_w = screen_w - ((cols + 1) * margin)

    win_h = usable_h // rows
    win_w = usable_w // cols

    for window_number, (window_name, camera_number, frame) in enumerate(frames_to_display):
        if frame is None or frame.size == 0:
            print(f"[WARNING] Frame for '{window_name}' '{camera_number}' is empty. Skipping.")
            continue

        # Resize frame
        resized, rw, rh = resize_with_aspect_ratio(frame, win_w, win_h)

        r = (window_number+(camera_number-camera.starting_camera)*len(frames_to_display)) // cols
        c = (window_number+(camera_number-camera.starting_camera)*len(frames_to_display)) % cols

        # Compute safe positions
        pos_x = margin + c * (win_w + margin)
        pos_y = margin + r * (win_h + title_bar_height + margin)

        cv2.namedWindow(window_name + str(camera_number), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name + str(camera_number), win_w, win_h)
        cv2.moveWindow(window_name + str(camera_number), pos_x, pos_y)
        cv2.imshow(window_name + str(camera_number), resized)
