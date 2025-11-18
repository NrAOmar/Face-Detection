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

cv2.namedWindow('Camera (DNN)', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera (DNN)', 400, 450)
cv2.moveWindow('Camera (DNN)', 0, 0)


def detect_faces(frame, out = ""):
    (h, w) = frame.shape[:2]

    # Prepare blob for the DNN
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    face_list = []    # to store all faces detected by the DNN

    faces = 0
    # Draw detections
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf >= conf_thr:
            faces += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clip to frame just in case
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # Compute width & height
            width  = x2 - x1
            height = y2 - y1
 
            # Store 
            face_list.append((x1, y1, width, height))


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display number of faces
    cv2.putText(
        frame,                          # Image
        f"Faces: {faces}",              # Text
        (10, 30),                       # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,       # Font
        1,                              # Font scale
        (0, 0, 255),                    # Color (B,G,R) → red
        2                               # Thickness
    )

    # Write the frame into the file 'output.mp4'
    if out != "":
        out.write(frame)
    return frame, face_list