import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Windows
cv2.namedWindow('Camera (Haar)', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera (Haar)', 400, 450)
cv2.moveWindow('Camera (Haar)', 640, 0)

def detect_faces(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    return faces