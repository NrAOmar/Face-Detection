import cv2

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

def rotate_image(img, angle = 90, scale = 1.0):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    # rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated

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

    rotated = cv2.warpAffine(img, rot_mat, (new_w, new_h))
    return rotated
