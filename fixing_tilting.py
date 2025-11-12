import cv2
import matplotlib.pyplot as plt
import haar_detector
import dnn_detector
import helpers

imagePath = 'input_image.jpg'

img = cv2.imread(imagePath)

rotated = helpers.rotate_image_without_cropping(img, 20)

img_rgb_haar = haar_detector.detect_faces(rotated.copy())
img_rgb_dnn = dnn_detector.detect_faces(rotated.copy())

# cv2.imshow('HAAR', img_rgb_haar)
# cv2.imshow('DNN', img_rgb_dnn)
# cv2.imshow('rotated', rotated)

cv2.imshow('HAAR', img_rgb_haar)
cv2.imshow('DNN', img_rgb_dnn)
cv2.imshow('rotated', rotated)
cv2.waitKey(0)           # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Close the window properly
# time.sleep(10)
# plt.figure(figsize=(10,5))
# plt.imshow(rotated)
# plt.axis('off')

# plt.figure(figsize=(10,5))
# plt.imshow(img_rgb_haar)
# plt.axis('off')

# plt.figure(figsize=(10,5))
# plt.imshow(img_rgb_dnn)
# plt.axis('off')

# plt.pause(10)