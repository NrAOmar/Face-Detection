import cv2
import math
from screeninfo import get_monitors

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

def display_frames_in_grid(window_names, frames, title_bar_height=29, margin=0):
    """
    Displays frames in a grid layout with screen fitting and proper handling
    of title bar height so no window goes outside the screen.
    """
    assert len(window_names) == len(frames)

    # Filter invalid frames
    valid_pairs = []
    for name, frame in zip(window_names, frames):
        if frame is None or frame.size == 0:
            print(f"[WARNING] Frame for '{name}' is empty. Skipping.")
        else:
            valid_pairs.append((name, frame))

    if not valid_pairs:
        print("No valid frames to display.")
        return

    window_names, frames = zip(*valid_pairs)

    # Screen size
    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height

    # Arrange grid
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Deduct margins + title bars exactly
    usable_h = screen_h - ((rows + 1) * margin) - (rows * title_bar_height)
    usable_w = screen_w - ((cols + 1) * margin)

    win_h = usable_h // rows
    win_w = usable_w // cols

    for i, (name, frame) in enumerate(zip(window_names, frames)):
        # Resize frame
        resized, rw, rh = resize_with_aspect_ratio(frame, win_w, win_h)

        r = i // cols
        c = i % cols

        # Compute safe positions
        pos_x = margin + c * (win_w + margin)
        pos_y = margin + r * (win_h + title_bar_height + margin)

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, win_w, win_h)
        cv2.moveWindow(name, pos_x, pos_y)
        cv2.imshow(name, resized)
