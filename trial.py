import cv2
import matplotlib.pyplot as plt

def clahe_luma_param(bgr, clip, grid):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    y_clahe = clahe.apply(y)

    out = cv2.merge([y_clahe, cr, cb])
    
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

# Load image
img = cv2.imread("lena.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Parameter sets to compare
configs = [
    (1.0, 8),
    (2.0, 8),
    (4.0, 8),
    (2.0, 4),
    (2.0, 16),
]

plt.figure(figsize=(15, 8))

# Original
plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

for i, (clip, grid) in enumerate(configs, start=2):
    out = clahe_luma_param(img, clip, grid)
    plt.subplot(2, 3, i)
    plt.title(f"clip={clip}, grid={grid}x{grid}")
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis("off")

plt.tight_layout()
plt.show()
