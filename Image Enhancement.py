import cv2
import numpy as np
from matplotlib import pyplot as plt
color_image = cv2.imread('image.jpg')
if color_image is None:
    raise FileNotFoundError("Image not found. Make sure 'image.jpeg' is in the working directory.")
color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
hist_eq = cv2.equalizeHist(gray_image)
_, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(binary, kernel, iterations=1)
titles = ['Input Image', 'Grayscale Image', 'Histogram Equalized', 'Binary Image',
          'Erosion']
images = [color_image_rgb, gray_image, hist_eq, binary, erosion]
plt.figure(figsize=(14, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
