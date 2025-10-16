import matplotlib.pyplot as plt
import numpy as np 

# Generar datos de ejempl 8002888222

x = np.array([-3, -2, -1, 0, 1, 2, 3])
y = np.array([-6, -4, -2, 0, 2, 4, 6])

w = np.array([-1, 0, 1, 2, 3, 4, 5])

errores = []
for wi in w:
	y_prima = wi * x 
	error = np.sum((y - y_prima) ** 2) / (2 * len(x))
	errores.append(error)

# Graficar W vs error
plt.plot(w, errores, marker='o')
plt.xlabel('W')
plt.ylabel('Error')
plt.title('Error vs W en Regresión Lineal')
plt.grid(True)
plt.show()

