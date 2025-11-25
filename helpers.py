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
def construct_boxes_old(faces, angle, confidences=None):
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

def construct_boxes(faces, angle, rot_mat=None, confidences=None):
    """
    faces: list of (x, y, w, h) in the ROTATED frame
    angle: rotation angle used
    rot_mat: 2x3 matrix from get_rot_mat / rotate_image
    confidences: list of conf values or None
    """
    boxes = []
    if confidences is None:
        confidences = [None] * len(faces)

    for (x, y, w, h), conf in zip(faces, confidences):
        # corners in ROTATED frame
        corners_rot = np.array([
            [x    , y    ],
            [x + w, y    ],
            [x    , y + h],
            [x + w, y + h],
        ], dtype=np.float32)

        # map back to ORIGINAL frame if we have a rotation matrix
        if rot_mat is not None:
            corners = rotate_points_back(corners_rot, rot_mat)
        else:
            corners = corners_rot

        if conf is None:
            meta = (angle,)           # Haar
        else:
            meta = (angle, float(conf))   # DNN

        boxes.append((corners, meta))

    return boxes

def add_boxes_all(frame, boxes, rotate_back=True):
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
 

def add_boxes(frame, boxes, draw_conf=True, color=(0, 255, 0), rotate_back=True):
    """
    Draw simple axis-aligned boxes on the frame.

    boxes can be:
      - list of dicts with keys: x1, y1, x2, y2, conf (conf optional)
      - or list of tuples: (x1, y1, x2, y2) or (x1, y1, x2, y2, conf)
    """
    H, W = frame.shape[:2]

    normalized = []

    # Normalize all inputs to (x1, y1, x2, y2, conf_or_None)
    for b in boxes:
        if isinstance(b, dict):
            x1 = b["x1"]
            y1 = b["y1"]
            x2 = b["x2"]
            y2 = b["y2"]
            conf = b.get("conf", None)
        else:
            # assume tuple / list
            if len(b) == 4:
                x1, y1, x2, y2 = b
                conf = None
            elif len(b) == 5:
                x1, y1, x2, y2, conf = b
            else:
                # unexpected format, skip
                continue

        # angles = [member["angle"] for member in b["members"]]

        # corners = np.array([
        #     [x1, y1],
        #     [x2, y1],
        #     [x1, y2],
        #     [x2, y2],
        # ], dtype=np.float32)

        # if rotate_back:
        #     for angle in angles:
        #         rot_mat, dimensions = get_rot_mat(angle) 
        #         corners = rotate_points_back(corners, rot_mat)

        # clip to image boundaries and convert to int
        x1 = int(np.clip(x1, 0, W - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        y2 = int(np.clip(y2, 0, H - 1))

        # ensure proper order
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        normalized.append((x1, y1, x2, y2, conf))

    # Draw all boxes
    for (x1, y1, x2, y2, conf) in normalized:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if draw_conf and conf is not None:
            text = f"{conf:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Faces count
    cv2.putText(frame, f"Faces: {len(normalized)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame



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
            conf = 0.5            # default for Haar (or any no-conf detector)

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


def iou(box_a, box_b):
    """
    box_a, box_b: dicts from preprocess_boxes (must contain x1,y1,x2,y2)

    returns: intersection-over-union value between 0 and 1
    """
    # coordinates of intersection rectangle
    inter_left   = max(box_a["x1"], box_b["x1"])
    inter_top    = max(box_a["y1"], box_b["y1"])
    inter_right  = min(box_a["x2"], box_b["x2"])
    inter_bottom = min(box_a["y2"], box_b["y2"])

    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area

def cluster_boxes(box_infos, iou_threshold=0.5):
    """
    box_infos: list of box dicts (output of preprocess_boxes)
    iou_threshold: boxes with IOU above this value are considered same face

    returns: list of clusters, each cluster is a list of box dicts
    """
    n = len(box_infos)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        # start a new cluster from box i
        stack = [i]
        visited[i] = True
        cluster_indices = [i]

        while stack:
            j = stack.pop()
            box_j = box_infos[j]

            # try to connect j with all yet-unvisited boxes
            for k in range(n):
                if visited[k]:
                    continue
                box_k = box_infos[k]
                if iou(box_j, box_k) >= iou_threshold:
                    visited[k] = True
                    stack.append(k)
                    cluster_indices.append(k)

        # collect all boxes in this cluster
        cluster = [box_infos[idx] for idx in cluster_indices]
        clusters.append(cluster)

    return clusters


def fuse_cluster_weighted(cluster):
    """
    cluster: list of box dicts (one cluster = one face)

    returns: a new box dict representing the merged box
             (x1,y1,x2,y2,cx,cy,w,h,conf,cluster_size,...)
    """
    if not cluster:
        return None

    # --- geometry: we still use confidences as weights ---
    sum_w = 0.0
    sum_cx = 0.0
    sum_cy = 0.0
    sum_w_box = 0.0
    sum_h_box = 0.0

    for b in cluster:
        w_conf = float(b["conf"])
        # avoid zero weights, just in case
        if w_conf <= 0.0:
            w_conf = 1e-6

        sum_w += w_conf
        sum_cx += w_conf * b["cx"]
        sum_cy += w_conf * b["cy"]
        sum_w_box += w_conf * b["w"]
        sum_h_box += w_conf * b["h"]

    if sum_w == 0.0:
        # fallback: equal weights
        n = len(cluster)
        sum_w = float(n)
        sum_cx = sum(b["cx"] for b in cluster)
        sum_cy = sum(b["cy"] for b in cluster)
        sum_w_box = sum(b["w"] for b in cluster)
        sum_h_box = sum(b["h"] for b in cluster)

    cx_final = sum_cx / sum_w
    cy_final = sum_cy / sum_w
    w_final  = sum_w_box / sum_w
    h_final  = sum_h_box / sum_w

    x1_final = cx_final - w_final / 2.0
    y1_final = cy_final - h_final / 2.0
    x2_final = cx_final + w_final / 2.0
    y2_final = cy_final + h_final / 2.0

    # --- confidence to DISPLAY: use average, not sum ---
    raw_confs = [float(b["conf"]) for b in cluster]
    avg_conf = sum(raw_confs) / len(raw_confs)

    return {
        "x1": x1_final,
        "y1": y1_final,
        "x2": x2_final,
        "y2": y2_final,
        "w": w_final,
        "h": h_final,
        "cx": cx_final,
        "cy": cy_final,
        "conf": avg_conf,          # <--- now between ~0 and 1 (if inputs are)
        "cluster_size": len(cluster),
        "members": cluster,
    }

def merge_boxes_with_iou(boxes_to_draw, iou_threshold=0.4):
    """
    boxes_to_draw: list of (points, meta) as in your current code

    returns: list of merged boxes (dicts with x1,y1,x2,y2,etc.),
             one per detected face.
    """
    if not boxes_to_draw:
        return []

    # Step 1: preprocess
    box_infos = preprocess_boxes(boxes_to_draw)

    # Step 2: cluster by IOU
    clusters = cluster_boxes(box_infos, iou_threshold=iou_threshold)

    # Step 3: fuse each cluster
    merged = []
    for cluster in clusters:
        merged_box = fuse_cluster_weighted(cluster)
        if merged_box is not None:
            merged.append(merged_box)

    return merged

def filter_boxes_by_confidence(merged_boxes, min_conf=0.5):
    """
    Keep only boxes with confidence >= min_conf.
    merged_boxes: list of dicts (output of fuse_cluster_weighted)
    """
    return [b for b in merged_boxes if float(b.get("conf", 0.0)) >= min_conf]
