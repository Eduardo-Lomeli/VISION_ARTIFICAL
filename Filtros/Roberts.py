import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pangolin.jpg', cv2.IMREAD_GRAYSCALE)

kernel_roberts_x = np.array([[1, 0],[0, -1]])

kernel_roberts_y = np.array([[0, 1],[-1, 0]])

# Aplicar el filtro Roberts en la dirección x
roberts_x = cv2.filter2D(img, -1, kernel_roberts_x)

# Aplicar el filtro Roberts en la dirección y
roberts_y = cv2.filter2D(img, -1, kernel_roberts_y)
#Combinado
roberts_combined = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(roberts_x, cmap='gray')
plt.title('Roberts X')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(roberts_y, cmap='gray')
plt.title('Roberts Y')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(roberts_combined, cmap='gray')
plt.title('Roberts Combinado')
plt.xticks([]), plt.yticks([])

plt.show()