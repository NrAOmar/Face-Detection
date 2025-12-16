import cv2
import helpers
import time
import numpy as np

camera_in_use = 2 # start with camera in the lab
stop_flag = False
latest_frame = None

# Open camera (macOS AVFoundation). Try 2 then 1 then 0.
cap = cv2.VideoCapture(camera_in_use)
while not cap.isOpened() and camera_in_use > 0:
    camera_in_use -= 1
    cap = cv2.VideoCapture(camera_in_use)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


def get_frame_size():
    # iPhone Camera
    # scale = 0.3
    # frame_width = 1920
    # frame_height = 1080
    # if (camera_in_use == 2): # Lab Camera
    #     scale = 0.5
    #     frame_width = 1280
    #     frame_height = 720
    # elif (camera_in_use == 1): # macOS Camera
    #     scale = 0.5
    #     frame_width = 1280
    #     frame_height = 720

    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    scale = 1.0
    return (scale, frame_width, frame_height)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 20.0  # safe fallback
print(f"Recording at {fps} FPS")

# Get frame_dnn size
frame_size = get_frame_size()
# Source - https://stackoverflow.com/a
# Posted by Hannes Ovrén
# Retrieved 2025-11-22, License - CC BY-SA 2.5

out_frame_size = (int(frame_size[0] * frame_size[1]), 
                  int(frame_size[0] * frame_size[2]))

# Video writer
out_haar = cv2.VideoWriter(
    'output_haar.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    out_frame_size
)
out_dnn = cv2.VideoWriter(
    'output_dnn.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    out_frame_size
)

print("Recording... Press 'q' to stop.")

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    a = cv2.subtract(a, np.mean(a) - 128)
    b = cv2.subtract(b, np.mean(b) - 128)
    result = cv2.merge((l, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )

def sharpen(img):
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)


# Pre-create CLAHE (do NOT recreate per frame)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Precompute gamma LUT
gamma = 1.2
inv_gamma = 1.0 / gamma
gamma_lut = np.array(
    [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
    dtype=np.uint8
)

def enhance_frame(frame):
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # CLAHE only on luminance
    y = clahe.apply(y)

    # Merge back
    ycrcb = cv2.merge((y, cr, cb))
    frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Gamma correction via LUT
    frame = cv2.LUT(frame, gamma_lut)

    return frame

def fast_contrast(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)

    ycrcb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def camera_loop():
    global latest_frame, stop_flag
    # cap = cv2.VideoCapture(1)

    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            # frame = white_balance(frame)
            # frame = enhance_contrast(frame)
            # frame = denoise(frame)
            # frame = sharpen(frame)
            frame = enhance_frame(frame)
            # frame = fast_contrast(frame)
            latest_frame = frame.copy()
        else:
            time.sleep(0.001)

    cap.release()
