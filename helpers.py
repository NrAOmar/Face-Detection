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
    (scale, h, w) = camera.frame_size
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

def construct_boxes(faces, texts):
    boxes = []
    
    for (x, y, w, h) in faces:
        # corners in rotated frame
        corners = np.array([
            [x  , y  ],
            [x+w, y  ],
            [x  , y+h],
            [x+w, y+h],
        ], dtype=np.float32)
        boxes.append((corners, texts))

    return boxes

def add_boxes(frame, boxes, rotate_back=True):
    for (corners, texts) in boxes:
        # rotate corners back
        angle = texts[0]

        if rotate_back:
            corners = rotate_points_back(corners, get_rot_mat(angle))

        # fit axis-aligned box
        xmin, ymin, xmax, ymax = corners
        xmin, xmax = np.clip([corners[:,0].min(), corners[:,0].max()], 0, camera.frame_size[1]-1).astype(int)
        ymin, ymax = np.clip([corners[:,1].min(), corners[:,1].max()], 0, camera.frame_size[2]-1).astype(int)
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{angle}°", (xmin, max(0, ymin+16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        if len(texts) > 1:
            cv2.putText(frame, f"{texts[1]}", (xmin, max(0, ymin-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.putText(frame, f"Faces: {len(boxes)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return frame