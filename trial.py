import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("bad_photos/WIN_20251225_20_40_16_Pro.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----- CLAHE on Y -----
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(ycrcb)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
y_clahe = clahe.apply(y)

ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
img_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)

# ----- Gamma correction -----
def gamma_correction(image, gamma=1.2):
    inv = 1.0 / gamma
    table = np.array([(i/255.0)**inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def unsharp(image, amount=0.5):
    blur = cv2.GaussianBlur(image, (0,0), 1.0)
    return cv2.addWeighted(image, 1 + amount, blur, -amount, 0)

for a in [0.2, 0.4, 0.6, 0.8]:
    img_s = unsharp(img_rgb, amount=a)
    plt.figure()
    plt.title(f"Sharpen amount={a}")
    plt.imshow(img_s)
    plt.axis("off")


img_gamma = gamma_correction(img_rgb, gamma=1.1)

# ----- Sharpening -----
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
img_sharp = cv2.filter2D(img_rgb, -1, kernel)

# ----- Display -----
titles = ["Original", "CLAHE (Y)", "Gamma", "Sharpened"]
images = [img_rgb, img_clahe, img_gamma, img_sharp]

plt.figure(figsize=(12,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")
plt.show()
