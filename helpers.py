import cv2
import numpy as np
import camera
import math
import mediapipe as mp
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
import os
import dnn_detector
import time

# frame_dnn_width  = int(cap.get(cv2.CAP_PROP_frame_dnn_WIDTH))
# frame_dnn_height = int(cap.get(cv2.CAP_PROP_frame_dnn_HEIGHT))

_face_mesh = None
_rec_model = None
_app = None

# ---------- Load known faces ----------
known_embeddings = []
known_names = []
last_good_name = "Unknown"
last_good_sim = 0.0
last_good_time = 0.0



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

def construct_boxes(faces, angle, confidences=None, rotate_back=True):
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

        if rotate_back:
            rot_mat, dimensions = get_rot_mat(angle)
            corners = rotate_points_back(corners, rot_mat)

        boxes.append((corners, meta))

    return boxes

def add_boxes_all(frame, boxes):
    for (corners, texts) in boxes:
        # rotate corners back
        angle = texts[0]

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
 

def add_boxes(frame, boxes):
    for b in boxes:

        x1 = b["x1"]
        y1 = b["y1"]
        x2 = b["x2"]
        y2 = b["y2"]
        conf = b.get("conf", None)

        # angles = [member["angle"] for member in b["members"]]

        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2],
        ], dtype=np.float32)

        # fit axis-aligned box
        xmin, ymin, xmax, ymax = corners
        xmin, xmax = np.clip([corners[:,0].min(), corners[:,0].max()], 0, camera.frame_size[1]-1).astype(int)
        ymin, ymax = np.clip([corners[:,1].min(), corners[:,1].max()], 0, camera.frame_size[2]-1).astype(int)
        
        color = (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        if conf is not None:
            text = f"{conf:.2f}"
            cv2.putText(frame, text, (xmin, max(0, ymin - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Faces count
    cv2.putText(frame, f"Faces: {len(boxes)}", (10, 30),
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


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        mp_face_mesh = mp.solutions.face_mesh
        _face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _face_mesh

def align_crop_by_eyes(crop_bgr):
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    face_mesh = _get_face_mesh()
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return crop_bgr

    lm = res.multi_face_landmarks[0].landmark
    left = lm[33]
    right = lm[263]

    xL, yL = int(left.x * w), int(left.y * h)
    xR, yR = int(right.x * w), int(right.y * h)

    angle = math.degrees(math.atan2(yR - yL, xR - xL))
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(crop_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
    return aligned

def to_xyxy(box):
    # Your merged boxes are dicts with x1,y1,x2,y2
    if isinstance(box, dict):
        return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

    # Fallback for list/tuple/np-array boxes
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return int(x1), int(y1), int(x2), int(y2)

def get_rec_model(ctx_id=0):
    global _rec_model
    if _rec_model is None:
        rec_path = os.path.join(
            os.path.expanduser("~"),
            ".insightface", "models", "buffalo_l", "w600k_r50.onnx"
        )
        _rec_model = get_model(rec_path)
        _rec_model.prepare(ctx_id=ctx_id)
    return _rec_model

def resize_to_112(img):
    h, w = img.shape[:2]
    if w < 320 or h < 320:
        interp = cv2.INTER_LANCZOS4  # or INTER_CUBIC
        return cv2.resize(img, (320, 320), interpolation=interp)
    else:
        return img
   

def clahe_luma(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    y = clahe.apply(y)
    out = cv2.merge([y, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

def upscale_bgr(img_bgr, scale=2):

    out, _ = _upsampler.enhance(img_bgr, outscale=2)
    return out


def embed_from_box(frame_bgr, box, margin=0.20):
    x1, y1, x2, y2 = to_xyxy(box)

    w = x2 - x1
    h = y2 - y1
    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(frame_bgr.shape[1], x2 + mx)
    y2 = min(frame_bgr.shape[0], y2 + my)

    crop = frame_bgr[y1:y2, x1:x2]
    crop_normal = align_crop_by_eyes(crop)
    if crop.size == 0:
        return None

    # crop = clahe_luma(crop_normal)
    crop112 = resize_to_112(crop_normal) # This function I use it to make a proper face recognition with out further faces wont be recognized.
    
   

    # cv2.imshow("Crop Normal", crop_normal)
    # cv2.imshow("Super Resolution", crop112)
    # cv2.waitKey(1)  # 1 ms so it doesn’t block
    rec_model = get_rec_model(ctx_id=0)  # gets cached model
    feat = rec_model.get_feat(crop112).flatten().astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-10)
    return feat
# This function is added to verify the haar with Dnn detection

def dnn_filter_boxes(frame_bgr, boxes, margin, conf_thr):
    confirmed = []

    for det in boxes:
        box_arr = det[0]      # (4,2)

        # bbox from points
        xs = box_arr[:, 0]
        ys = box_arr[:, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        mx = int(w * margin)
        my = int(h * margin)

        x1c = max(0, x1 - mx)
        y1c = max(0, y1 - my)
        x2c = min(frame_bgr.shape[1], x2 + mx)
        y2c = min(frame_bgr.shape[0], y2 + my)

        crop = frame_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue
        crop = resize_to_112(crop)

        faces, confs = dnn_detector.detect_faces(crop)


        # keep this Haar box only if DNN sees a face in the crop confidently
        if len(faces) > 0 and any(c >= conf_thr for c in confs):
            confirmed.append(det)
        if any(c <= conf_thr for c in confs):
            print("function works")

    return confirmed

def get_face_app(ctx_id=0, det_size=(320, 320), name="buffalo_l"):
    """
    Returns a cached InsightFace FaceAnalysis instance.
    Creates it once, then reuses it.
    """
    global _app
    if _app is None:
        _app = FaceAnalysis(name=name)
        _app.prepare(ctx_id=ctx_id, det_size=det_size)
    return _app

def load_known_faces(base_path="dataset"):
    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            app = get_face_app(ctx_id=0, det_size=(320, 320))
            faces = app.get(img)
            if len(faces) > 0:
                known_embeddings.append(faces[0].embedding.astype(np.float32))
                known_names.append(person)

    print(f"Loaded {len(known_embeddings)} known faces")