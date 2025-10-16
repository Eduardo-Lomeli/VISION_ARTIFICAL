import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

#Funciones para agregar ruido a las imágenes
def add_gaussian_noise(image, mean=0, std_dev=25):
    # Ruido aleatorio con el mismo tamaño que la imagen
    noise = np.random.normal(mean, std_dev, image.shape)
    # Suma el ruido a la imagen
    noisy_image = image + noise
    # Ajusta el rango a 255
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_salt_and_pepper_noise(image, amount=0.05):
    output_image = np.copy(image)
    num_salt = int(np.ceil(amount * image.size * 0.5))
    num_pepper = int(np.ceil(amount * image.size * 0.5))
    # Genera coordenadas aleatorias para poner los píxeles blancos
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    output_image[tuple(coords)] = 255
    # Genera coordenadas aleatorias para poner los píxeles negros
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    output_image[tuple(coords)] = 0
    
    return output_image

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
            elif filter_type == 'mediana':
                pixel_value = np.median(window)
            elif filter_type == 'min':
                pixel_value = np.min(window)
            elif filter_type == 'max':
                pixel_value = np.max(window)
            
            output_image[i, j] = pixel_value
            
    return output_image

try:
    origin = cv2.imread("./pangolin.jpg", cv2.IMREAD_GRAYSCALE)
except:
    print("Error: No se encontró la imagen")
    exit()

KERNEL_SIZE = 5
#Generar imágenes con ruido
gaussiano_ruido = add_gaussian_noise(origin, std_dev=40)
sal_pimienta_ruido = add_salt_and_pepper_noise(origin, amount=0.1)

#filtros a la imagen con ruido gaussiano
media_gaussiana = apply_filter(gaussiano_ruido, KERNEL_SIZE, 'media')
mediana_gaussiana = apply_filter(gaussiano_ruido, KERNEL_SIZE, 'mediana')
min_gaussiano = apply_filter(gaussiano_ruido, KERNEL_SIZE, 'min')
max_gaussiano = apply_filter(gaussiano_ruido, KERNEL_SIZE, 'max')

#filtros a la imagen con ruido sal y pimienta
sp_mean = apply_filter(sal_pimienta_ruido, KERNEL_SIZE, 'media')
sp_median = apply_filter(sal_pimienta_ruido, KERNEL_SIZE, 'mediana')
sp_min = apply_filter(sal_pimienta_ruido, KERNEL_SIZE, 'min')
sp_max = apply_filter(sal_pimienta_ruido, KERNEL_SIZE, 'max')

fig, axes = plt.subplots(4, 3, figsize=(18, 7))
fig.suptitle('Implementación de Filtrado Espacial', fontsize=16)
#Ruido Gaussiano
axes[0,0].imshow(origin, cmap='gray')
axes[0,0].set_title('Original')
axes[0,1].imshow(gaussiano_ruido, cmap='gray')
axes[0,1].set_title('Ruido Gaussiano')
axes[0,2].imshow(media_gaussiana, cmap='gray')
axes[0,2].set_title('Filtro de Media')
axes[1,0].imshow(mediana_gaussiana, cmap='gray')
axes[1,0].set_title('Filtro de Mediana')
axes[1,1].imshow(min_gaussiano, cmap='gray')
axes[1,1].set_title('Filtro Minimo')
axes[1,2].imshow(max_gaussiano, cmap='gray')
axes[1,2].set_title('Filtro Maximo')

#Ruido Sal y Pimienta
axes[2,0].imshow(origin, cmap='gray')
axes[2,0].set_title('Original')
axes[2,1].imshow(sal_pimienta_ruido, cmap='gray')
axes[2,1].set_title('Ruido S&P')
axes[2,2].imshow(sp_mean, cmap='gray')
axes[2,2].set_title('Filtro de Media')
axes[3,0].imshow(sp_median, cmap='gray')
axes[3,0].set_title('Filtro de Mediana')
axes[3,1].imshow(sp_min, cmap='gray')
axes[3,1].set_title('Filtro Minimo')
axes[3,2].imshow(sp_max, cmap='gray')
axes[3,2].set_title('Filtro Maximo')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()