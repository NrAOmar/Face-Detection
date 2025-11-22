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

def rotate_points_back(points, M):
    M_inv = cv2.invertAffineTransform(M)
    ones = np.ones((points.shape[0], 1))
    pts_homo = np.hstack([points, ones])
    return (M_inv @ pts_homo.T).T

def get_rot_mat(img, angle, scale = 1.0, cropping = False):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

    new_w = w
    new_h = h

    # Compute the new bounding dimensions
    if not cropping:
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to account for translation
        rot_mat[0, 2] += new_w / 2 - center[0]
        rot_mat[1, 2] += new_h / 2 - center[1]

    return rot_mat, (new_w, new_h)

def rotate_image(img, angle, scale = 1.0, cropping=False):
    rot_mat, dimensions = get_rot_mat(img, angle, scale, cropping)
    rotated_img = cv2.warpAffine(img, rot_mat, dimensions)
    return rotated_img, rot_mat

def construct_boxes(frame_size, faces, rot_mat=np.eye(3)[:-1], angle=0):
    boxes = []
    frame_h, frame_w = frame_size[1:]
    
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
        xmin, xmax = np.clip([unrot_corners[:,0].min(), unrot_corners[:,0].max()], 0, frame_w-1).astype(int)
        ymin, ymax = np.clip([unrot_corners[:,1].min(), unrot_corners[:,1].max()], 0, frame_h-1).astype(int)

        boxes.append((xmin, ymin, xmax, ymax, angle))

    return boxes

def add_boxes(frame_original, boxes):
    # draw all boxes on original frame
    frame = frame_original.copy()
    for (xmin, ymin, xmax, ymax, angle) in boxes:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{angle}°", (xmin, max(0, ymin+16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.putText(frame, f"Faces: {len(boxes)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame