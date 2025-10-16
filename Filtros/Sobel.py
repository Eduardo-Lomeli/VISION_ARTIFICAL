import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pangolin.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar el filtro Sobel en la dirección x
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# Aplicar el filtro Sobel en la dirección y
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#Combinacion
abs_sobelx = np.uint8(np.absolute(sobelx))
abs_sobely = np.uint8(np.absolute(sobely))
sobel_combined = cv2.bitwise_or(abs_sobelx, abs_sobely)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(abs_sobelx, cmap='gray')
plt.title('Sobel X')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(abs_sobely, cmap='gray')
plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Combinado')
plt.xticks([]), plt.yticks([])

plt.show()