import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Windows
cv2.namedWindow('Camera (Haar)', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Camera (Haar)', 400, 450)
cv2.moveWindow('Camera (Haar)', 640, 0)

def detect_faces(frame, out = ""):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Display number of faces
    cv2.putText(
        frame,                          # Image
        f"Faces: {len(faces)}",         # Text
        (10, 30),                       # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,       # Font
        1,                              # Font scale
        (0, 0, 255),                    # Color (B,G,R) → red
        2                               # Thickness
    )

    # Write the frame into the file 'output.mp4'
    if out != "":
        out.write(frame)
    return frame