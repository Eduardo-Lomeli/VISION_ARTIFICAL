import matplotlib.pyplot as plt
import numpy as np 

# Generar datos aleatorios
np.random.seed(42)
x = np.random.uniform(0, 10, 100)
y = 3 * x + np.random.normal(0, 2, 100)

theta0 = 0  
theta1 = 0  

alpha = 0.01  
epocas = 500
m = len(x)

for i in range(epocas):
 
    h = theta0 + theta1 * x
    
    # Calcular gradientes
    grad_theta0 = np.sum(h - y) / m
    grad_theta1 = np.sum((h - y) * x) / m
    
    # Actualizar parámetros
    theta0 = theta0 - alpha * grad_theta0
    theta1 = theta1 - alpha * grad_theta1
    

print(f"theta0 (intercepto) = {theta0:.4f}")
print(f"theta1 (pendiente) = {theta1:.4f}")
print(f"La ecuacion de la recta es: y = {theta0:.4f} + {theta1:.4f}*x")
print(f"(Valor esperado y = 0 + 3*x)")