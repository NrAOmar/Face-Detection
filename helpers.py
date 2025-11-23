import cv2
import numpy as np
import camera

# frame_dnn_width  = int(cap.get(cv2.CAP_PROP_frame_dnn_WIDTH))
# frame_dnn_height = int(cap.get(cv2.CAP_PROP_frame_dnn_HEIGHT))

def rotate_points_back(points, M):
    M_inv = cv2.invertAffineTransform(M)
    ones = np.ones((points.shape[0], 1))
    pts_homo = np.hstack([points, ones])
    return (M_inv @ pts_homo.T).T

def get_rot_mat(angle, cropping = False):
    (scale, w, h) = camera.frame_size
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

def rotate_image(img, angle, cropping=False):
    rot_mat, dimensions = get_rot_mat(angle, cropping)
    rotated_img = cv2.warpAffine(img, rot_mat, dimensions)
    return rotated_img, rot_mat

# def construct_boxes(faces, texts):
#     boxes = []
    
#     for (x, y, w, h) in faces:
#         # corners in rotated frame
#         corners = np.array([
#             [x  , y  ],
#             [x+w, y  ],
#             [x  , y+h],
#             [x+w, y+h],
#         ], dtype=np.float32)
#         boxes.append((corners, texts))
    
#     return boxes
def construct_boxes(faces, angle, confidences=None):
    """
    faces: list of (x, y, w, h)
    angle: rotation angle used for detection
    confidences: list of confidence values, same length as faces, or None
    """
    boxes = []

    if confidences is None:
        confidences = [None] * len(faces)

    for (x, y, w, h), conf in zip(faces, confidences):
        corners = np.array([
            [x    , y    ],
            [x + w, y    ],
            [x    , y + h],
            [x + w, y + h],
        ], dtype=np.float32)

        if conf is None:
            meta = (angle,)                 # Haar: only angle
        else:
            meta = (angle, float(conf))     # DNN: angle + confidence

        boxes.append((corners, meta))

    return boxes

def add_boxes(frame, boxes, rotate_back=True):
    for (corners, texts) in boxes:
        # rotate corners back
        angle = texts[0]

        rot_mat, dimensions = get_rot_mat(angle)
        if rotate_back:
            corners = rotate_points_back(corners, rot_mat)

        # fit axis-aligned box
        xmin, ymin, xmax, ymax = corners
        xmin, xmax = np.clip([corners[:,0].min(), corners[:,0].max()], 0, camera.frame_size[1]-1).astype(int)
        ymin, ymax = np.clip([corners[:,1].min(), corners[:,1].max()], 0, camera.frame_size[2]-1).astype(int)
        
        color = (0, 255, 0)
        if len(texts) > 1:
            color = (255, 0, 0)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, f"{texts[1]}", (xmin, max(0, ymin-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        cv2.putText(frame, f"{angle}°", (xmin, max(0, ymin+16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, f"Faces: {len(boxes)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame

import numpy as np

def preprocess_boxes(boxes_to_draw):
    """
    Convert your (points, meta) boxes into a convenient dict with:
    x1, y1, x2, y2, w, h, cx, cy, angle, conf, points, meta
    """
    processed = []

    for points, meta in boxes_to_draw:
        # points: 4x2 array of corners [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = points[:, 0]
        ys = points[:, 1]

        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max())
        y2 = float(ys.max())

        w = x2 - x1
        h = y2 - y1

        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        # meta is (angle,) for Haar, (angle, conf) for DNN
        angle = float(meta[0]) if len(meta) >= 1 else None
        if len(meta) >= 2:
            conf = float(meta[1])  # DNN confidence
        else:
            conf = 1.0             # default for Haar (or any no-conf detector)

        processed.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "angle": angle,
            "conf": conf,
            "points": points,
            "meta": meta,
        })

    return processed
