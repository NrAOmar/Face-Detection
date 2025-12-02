import cv2
import face_recognition
import pickle
import numpy as np

ENCODINGS_FILE = "encodings.pickle"

# Load the encodings from file
print("[INFO] Loading encodings...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print(f"[INFO] Loaded {len(known_encodings)} encodings.")

def identify_faces(frame, tolerance=0.45):
    """
    Takes a BGR frame (from OpenCV),
    returns (boxes, names) for each detected face.
    """

    # Convert frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    boxes = face_recognition.face_locations(rgb, model="hog")

    # Compute encodings for each face
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:
        # Compare with known encodings
        distances = face_recognition.face_distance(known_encodings, encoding)

        # If we have no known faces stored, mark as unknown
        if len(distances) == 0:
            names.append("Unknown")
            continue

        # Find the best match
        best_index = np.argmin(distances)
        best_distance = distances[best_index]

        if best_distance < tolerance:
            name = known_names[best_index]
        else:
            name = "Unknown"

        names.append(name)

    return boxes, names


def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, stopping.")
            break

        boxes, names = identify_faces(frame)

        # Draw results
        for (top, right, bottom, left), name in zip(boxes, names):
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw label
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Face Identification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
