import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pangolin.jpg', cv2.IMREAD_GRAYSCALE)


# Definir los kernels de Prewitt
kernel_prewitt_x = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
kernel_prewitt_y = np.array([[-1, -1, -1],[ 0,  0,  0],[ 1,  1,  1]])

# Aplicar el filtro Prewitt en la dirección x
prewitt_x = cv2.filter2D(img, -1, kernel_prewitt_x)
# Aplicar el filtro Prewitt en la dirección y
prewitt_y = cv2.filter2D(img, -1, kernel_prewitt_y)

#Combinacion
prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(prewitt_x, cmap='gray')
plt.title('Prewitt X')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(prewitt_y, cmap='gray')
plt.title('Prewitt Y')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(prewitt_combined, cmap='gray')
plt.title('Prewitt Combinado')
plt.xticks([]), plt.yticks([])

plt.show()