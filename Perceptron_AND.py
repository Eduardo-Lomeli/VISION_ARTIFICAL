import numpy as np
import matplotlib.pyplot as plt

entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
salidas = np.array([0, 0, 0, 1])

# W1*X1 + W2*X2 + ... + Wn*Xn

def activacion(pesos, x, bias):
    z = pesos * x
    if z.sum() + bias > 0:
        return 1
    else:
        return 0

np.random.seed(42)
pesos = np.random.uniform(-1, 1, size=2)
bias = np.random.uniform(-1, 1)
delta = 0.01
epocas = 100

for epoca in range(epocas):
    for i in range(len(entradas)):
        prediccion = activacion(pesos, entradas[i], bias)
        error = salidas[i] - prediccion

        pesos[0] += delta * error * entradas[i][0]
        pesos[1] += delta * error * entradas[i][1]
        bias += delta * error
print(f"Epoca {epoca+1}: Pesos = {pesos}, Bias = {bias}")

print(f"0 AND 0 = {activacion(pesos, [0, 0], bias)}")
print(f"0 AND 1 = {activacion(pesos, [0, 1], bias)}")
print(f"1 AND 0 = {activacion(pesos, [1, 0], bias)}")
print(f"1 AND 1 = {activacion(pesos, [1, 1], bias)}")