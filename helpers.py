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