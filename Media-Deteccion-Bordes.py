import cv2
import numpy as np
import random




def apply_filter(image, kernel_size, filter_type):
    rows, cols = image.shape
    #Imagen vacia para guardar el resultado
    output_image = np.zeros((rows, cols), dtype=np.uint8)
    
    #Padding
    pad_width = kernel_size // 2
    
    #Se crea la imagen con los bordes extra
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    #Bucle de recorrido por cada píxel
    for i in range(rows):
        for j in range(cols):
            window = padded_image[i : i + kernel_size, j : j + kernel_size]
            
            if filter_type == 'media':
                pixel_value = np.mean(window)

            output_image[i, j] = pixel_value
            
    return output_image

try:
    origin = cv2.imread("./pangolin.jpg", cv2.IMREAD_GRAYSCALE)
except:
    print("Error: No se encontró la imagen")
    exit()

KERNEL_SIZE = 5

media = apply_filter(origin, KERNEL_SIZE, 'media')


m,n =media.shape
kernel_dx=np.array([[1,1,1],[-1,-1,-1],[0,0,0]])
kernel_dy=np.array([[0,-1,1],[0,-1,1],[0,-1,1]])
bordes_dx=np.zeros_like(media)
bordes_dy=np.zeros_like(media)

for i in range(m-2):
    for j in range(n-2):
        res_dx=np.sum(media[i:i+3,j:j+3]*kernel_dx)
        res_dy=np.sum(media[i:i+3,j:j+3]*kernel_dy)
        if res_dx>15:
            bordes_dx[i,j]=255
        if res_dy>15:
            bordes_dy[i,j]=255
bordes = np.abs(bordes_dx) + np.abs(bordes_dy)
cv2.imshow("media",media)
cv2.imshow("bordes",bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("end")