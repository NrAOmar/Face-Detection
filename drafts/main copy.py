import cv2
import numpy as np
import os
import urllib.request

# Model file names
PROTOTXT = "deploy.prototxt"
CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# URLs used by the official OpenCV sample
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def ensure_file(path, url):
    if os.path.exists(path):
        return True
    try:
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print("Done.")
        return True
    except Exception as e:
        print(f"Could not download {path}. Error: {e}")
        return False

# Try to make sure the model files exist (skip if offline and already present)
have_proto = ensure_file(PROTOTXT, PROTOTXT_URL)
have_model = ensure_file(CAFFEMODEL, CAFFEMODEL_URL)
if not (have_proto and have_model):
    print("Model files missing. Place deploy.prototxt and the caffemodel next to this script.")
    # You can still continue if you already have them elsewhere and set absolute paths.
    # exit()

# Load network
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
conf_thr = 0.5  # raise to 0.6 if you see false positives

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
# frame_dnn_width  = int(cap.get(cv2.CAP_PROP_frame_dnn_WIDTH))
# frame_dnn_height = int(cap.get(cv2.CAP_PROP_frame_dnn_HEIGHT))
frame_width  = int(cap.get(4))
frame_height = int(cap.get(3))

# Video writer
out_haar = cv2.VideoWriter(
    'output_haar.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    20.0,
    (frame_width, frame_height)
)
out = cv2.VideoWriter(
    'output_dnn.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    20.0,
    (frame_width, frame_height)
)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Recording with DNN... Press 'q' to stop.")

# Windows
cv2.namedWindow('Camera (DNN)', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera (DNN)', 400, 450)
cv2.moveWindow('Camera (DNN)', 0, 0)
cv2.namedWindow('Camera (Haar)', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera (Haar)', 400, 450)
cv2.moveWindow('Camera (Haar)', 640, 0)

scale = 0.3
if (camera_in_use == 1):
    scale = 0.5
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break


#Haar

#DNN
    frame_dnn = frame.copy()

    (h, w) = frame_dnn.shape[:2]

    # Prepare blob for the DNN
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_dnn, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    faces_dnn = 0
    # Draw detections
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf >= conf_thr:
            faces_dnn += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clip to frame_dnn just in case
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cv2.rectangle(frame_dnn, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame_dnn, f"{conf:.2f}", (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(
        frame_dnn,                      # Image
        f"Faces: {faces_dnn}",          # Text
        (10, 30),                       # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,       # Font
        1,                              # Font scale
        (0, 0, 255),                    # Color (B,G,R) → red
        2                               # Thickness
    )

    # Write the frame into the file 'output.mp4'
    out_dnn.write(frame_dnn)

    # Show the frame
    display_frame_dnn = cv2.resize(frame_dnn, (0, 0), fx=scale, fy=scale)
    cv2.imshow('Camera (DNN)', display_frame_dnn)

    # Wait for close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_dnn.release()
out_haar.release()
cv2.destroyAllWindows()