import cv2
import numpy as np
import haar_detector
import dnn_detector
import helpers

imagePath = 'input_image.jpg'

img = cv2.imread(imagePath)

rotated_img, rot_mat = helpers.rotate_image(img.copy(), 20)

faces, img_rgb_haar = haar_detector.detect_faces(rotated_img.copy())

all_boxes = []

for (x, y, w, h) in faces:
    # corners in rotated frame
    corners = np.array([
        [x,         y        ],
        [x+w,   y        ],
        [x,         y+h  ],
        [x+w,   y+h  ],
    ], dtype=np.float32)

    # rotate corners back
    unrot_corners = helpers.rotate_points_back(corners, rot_mat)

    # fit axis-aligned box
    xmin = int(unrot_corners[:,0].min())
    ymin = int(unrot_corners[:,1].min())
    xmax = int(unrot_corners[:,0].max())
    ymax = int(unrot_corners[:,1].max())

    all_boxes.append((xmin, ymin, xmax, ymax))

# draw all boxes on original frame
for (xmin, ymin, xmax, ymax) in all_boxes:
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

cv2.putText(img, f"Faces: {len(all_boxes)}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.imshow("Rotated detection", img)



# # Display number of faces
# cv2.putText(
#     frame,                          # Image
#     f"Faces: {len(faces)}",         # Text
#     (10, 30),                       # Position (x, y)
#     cv2.FONT_HERSHEY_SIMPLEX,       # Font
#     1,                              # Font scale
#     (0, 0, 255),                    # Color (B,G,R) → red
#     2                               # Thickness
# )

# # Write the frame into the file 'output.mp4'
# if out != "":
#     out.write(frame)









# img_rgb_dnn = dnn_detector.detect_faces(rotated_img.copy())

# rot_mat3d = [[rot_mat[0][0], rot_mat[0][1], 0],
#              [rot_mat[1][0], rot_mat[1][1], 0],
#              [0, 0, 1],
# ]

# rot_mat3d = [rot_mat[0],
#              rot_mat[1],
#              [0, 0, 0],
# ]


# print(rot_mat)
# # rot_inv = np.linalg.inv(rot_mat3d)
# rot_inv = cv2.invertAffineTransform(rot_mat)

# for (x, y, w, h) in faces:
#     center_point_old = (x+w/2, y+h/2, 0)

#     Minv_full = np.vstack([rot_inv, [0, 0, 1]])

#     center_point_new = np.dot(Minv_full, center_point_old)
    
#     starting_point = (int(center_point_new[0]-w/2), int(center_point_new[1]-h/2))
#     ending_point = (int(center_point_new[0]+w/2), int(center_point_new[1]+h/2))
#     cv2.rectangle(img, starting_point, ending_point, (0, 255, 0), 4)
    
#     # x_detected = int(center_point_new[0]-w/2)
#     # y_detected = int(center_point_new[1]-h/2)
#     # cv2.rectangle(img, (x_detected, y_detected), (x_detected+w, y_detected+h), (0, 255, 0), 4)
    
#     # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)


# cv2.imshow('HAAR', img_rgb_haar)
# cv2.imshow('DNN', img_rgb_dnn)
# cv2.imshow('rotated_img', rotated_img)

cv2.imshow('Original Image', img)
cv2.imshow('HAAR', img_rgb_haar)
# cv2.imshow('DNN', img_rgb_dnn)
cv2.imshow('Rotated Image', rotated_img)
cv2.waitKey(0)           # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Close the window properly
# time.sleep(10)
# plt.figure(figsize=(10,5))
# plt.imshow(rotated_img)
# plt.axis('off')

# plt.figure(figsize=(10,5))
# plt.imshow(img_rgb_haar)
# plt.axis('off')

# plt.figure(figsize=(10,5))
# plt.imshow(img_rgb_dnn)
# plt.axis('off')

# plt.pause(10)