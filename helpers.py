import cv2
import numpy as np

def get_frame_size(camera_in_use):
    # iPhone Camera
    scale = 0.3
    frame_width = 1920
    frame_height = 1080
    if (camera_in_use == 2): # Lab Camera
        scale = 0.5
        frame_width = 1280
        frame_height = 720
    elif (camera_in_use == 1): # macOS Camera
        scale = 0.5
        frame_width = 1280
        frame_height = 720

    return (scale, frame_width, frame_height)

# frame_dnn_width  = int(cap.get(cv2.CAP_PROP_frame_dnn_WIDTH))
# frame_dnn_height = int(cap.get(cv2.CAP_PROP_frame_dnn_HEIGHT))

def rotate_image(img, angle, scale = 1.0):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    # rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated, rotation_matrix

def rotate_points_back(points, M):
    """Rotate corner points back using the inverse rotation matrix."""
    M_inv = cv2.invertAffineTransform(M)
    ones = np.ones((points.shape[0], 1))
    pts_homo = np.hstack([points, ones])
    return (M_inv @ pts_homo.T).T






def rotate_image_without_cropping(img, angle = 90, scale = 1.0):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

    # Compute the new bounding dimensions
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to account for translation
    rot_mat[0, 2] += new_w / 2 - center[0]
    rot_mat[1, 2] += new_h / 2 - center[1]

    rotated_img = cv2.warpAffine(img, rot_mat, (new_w, new_h))
    return rotated_img, rot_mat










def map_rotated_rect_to_original(x, y, w, h, Minv):
    """
    Given a rectangle (x,y,w,h) in rotated-image coords and the inverse affine matrix Minv (2x3),
    map the rectangle corners back to the original image and return an axis-aligned bounding box
    (xmin, ymin, w_new, h_new) in original-image coordinates.
    """
    # corners in rotated image coords
    corners = np.array([
        [x,     y,     1],
        [x + w, y,     1],
        [x,     y + h, 1],
        [x + w, y + h, 1]
    ]).T  # shape (3, 4)

    # build 3x3 inverse matrix from 2x3 Minv for homogeneous multiplication
    Minv_full = np.vstack([Minv, [0, 0, 1]])  # shape (3,3)

    mapped = Minv_full @ corners  # shape (3,4)
    xs = mapped[0, :]
    ys = mapped[1, :]

    xmin = int(np.clip(xs.min(), 0, w-1))
    ymin = int(np.clip(ys.min(), 0, h-1))
    xmax = int(np.clip(xs.max(), 0, w-1))
    ymax = int(np.clip(ys.max(), 0, h-1))

    return xmin, ymin, xmax - xmin, ymax - ymin

def detect_faces_multi_rotation_mapped(frame, detector, step=10):
    """
    Rotate frame by multiples of `step` degrees, detect faces in each rotated image,
    and map bounding boxes back to the original frame coordinates.
    Returns list of tuples (mapped_bbox, angle) where mapped_bbox = (x,y,w,h).
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    mapped_detections = []

    for angle in range(0, 360, step):
        # rotation matrix and rotated image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))

        # detect on rotated image
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        if len(faces) == 0:
            continue

        # inverse affine to map back to original coords
        Minv = cv2.invertAffineTransform(M)

        for (x, y, fw, fh) in faces:
            mapped_box = map_rotated_rect_to_original(x, y, fw, fh, Minv)
            mapped_detections.append((mapped_box, angle))

    return mapped_detections







# ========================================
def add_boxes(frame, boxes, angle):
    # draw all boxes on original frame
    for (xmin, ymin, xmax, ymax) in boxes:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{angle}°", (xmin, max(0, ymin-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.putText(frame, f"Faces: {len(boxes)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame

def construct_boxes(faces, rot_mat=np.eye(3)[:-1]):
    boxes = []
    for (x, y, w, h) in faces:
        # corners in rotated frame
        corners = np.array([
            [x  , y  ],
            [x+w, y  ],
            [x  , y+h],
            [x+w, y+h],
        ], dtype=np.float32)

        # rotate corners back
        unrot_corners = rotate_points_back(corners, rot_mat)

        # fit axis-aligned box
        xmin = int(unrot_corners[:,0].min())
        ymin = int(unrot_corners[:,1].min())
        xmax = int(unrot_corners[:,0].max())
        ymax = int(unrot_corners[:,1].max())

        boxes.append((xmin, ymin, xmax, ymax))

    return boxes
