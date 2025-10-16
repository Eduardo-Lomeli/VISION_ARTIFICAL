import cv2
import numpy as np

def apply_filter(image, kernel_size, filter_type):
    rows, cols = image.shape
    output_image = np.zeros((rows, cols), dtype=np.uint8)
    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    for i in range(rows):
        for j in range(cols):
            window = padded_image[i : i + kernel_size, j : j + kernel_size]
            if filter_type == 'media':
                pixel_value = np.mean(window)
            output_image[i, j] = pixel_value
    return output_image

def detectar_bordes_manual(fotograma):
    img_gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
    
    media = apply_filter(img_gris, 5, 'media')
    
    m, n = media.shape
    kernel_dx = np.array([[1, 1, 1], [-1, -1, -1], [0, 0, 0]])
    kernel_dy = np.array([[0, -1, 1], [0, -1, 1], [0, -1, 1]])
    bordes_dx = np.zeros_like(media)
    bordes_dy = np.zeros_like(media)

    for i in range(m - 2):
        for j in range(n - 2):
            res_dx = np.sum(media[i:i+3, j:j+3] * kernel_dx)
            res_dy = np.sum(media[i:i+3, j:j+3] * kernel_dy)
            if res_dx > 15:
                bordes_dx[i, j] = 255
            if res_dy > 15:
                bordes_dy[i, j] = 255
                
    bordes = cv2.add(bordes_dx, bordes_dy)
    return bordes, media

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    imagen_bordes, imagen_media = detectar_bordes_manual(frame)
    cv2.imshow("Bordes Manuales", imagen_bordes)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()