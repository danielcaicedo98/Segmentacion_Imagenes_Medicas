import numpy as np
import math




img_data= np.array(
    [[ 0,  1.,  2. , 3.],
    [ 4.,  200. , 30,  7.],
    [ 200. , 9., 10., 21.]])

F = [(0, 0), (0, 1)]
B = [(2, 1), (2, 0), (2, 2)]
height, width = img_data.shape

B = list(set(B))  
F = list(set(F))  
V = [(d, h) for d in range(height) for h in range(width)]
E = []
W_e = []
N = []
D = []
# Crear una matriz para la nueva imagen con la misma forma que la imagen original
nueva_img_data = np.zeros_like(img_data)
shape = img_data.shape
# print(shape)
def calculate_sigma(edges):
    sigma = 0
    for i in edges:
        diff = np.abs(i[0] - i[1])
        max_diff = np.max(diff)
        if max_diff > sigma:
            sigma = max_diff
    return sigma

def calculate_wijk(edges,sigma):    
    w = []
    β = 0.1
    for i in edges:
        diff = np.abs(i[0] - i[1])      
        max_diff = β * np.max(diff)**2        
        div = -1 * (max_diff / sigma)
        exp = math.exp(div)  
        w.append(exp)
    # print(w)   
    return w
cont = 0

for i in range(height):    
    for j in range(width):
        # Obtener el valor del píxel actual
        voxel_actual = img_data[i, j]
        
        # Obtener los valores de los 8 vecinos
        vecinos = [
            img_data[max(i-1, 0), max(j-1, 0)], img_data[max(i-1, 0), j],
            img_data[max(i-1, 0), min(j+1, width-1)], img_data[i, max(j-1, 0)],
            img_data[i, min(j+1, width-1)], img_data[min(i+1, height-1), max(j-1, 0)],
            img_data[min(i+1, height-1), j], img_data[min(i+1, height-1), min(j+1, width-1)]
        ]
        
        # Crear las aristas y los índices de los vecinos
        e = [
            (voxel_actual, vecinos[0]), (voxel_actual, vecinos[1]),
            (voxel_actual, vecinos[2]), (voxel_actual, vecinos[3]),
            (voxel_actual, vecinos[4]), (voxel_actual, vecinos[5]),
            (voxel_actual, vecinos[6]), (voxel_actual, vecinos[7])
        ]
        print(e)

        n = [
            V.index((max(i-1, 0), max(j-1, 0))), V.index((max(i-1, 0), j)),
            V.index((max(i-1, 0), min(j+1, width-1))), V.index((i, max(j-1, 0))),
            V.index((i, min(j+1, width-1))), V.index((min(i+1, height-1), max(j-1, 0))),
            V.index((min(i+1, height-1), j)), V.index((min(i+1, height-1), min(j+1, width-1)))
        ]

        indice_actual = i * width + j
        
        # Obtener los índices de los 8 vecinos
        nbh = [
            max(i-1, 0) * width + max(j-1, 0), max(i-1, 0) * width + j,
            max(i-1, 0) * width + min(j+1, width-1), i * width + max(j-1, 0),
            i * width + min(j+1, width-1), min(i+1, height-1) * width + max(j-1, 0),
            min(i+1, height-1) * width + j, min(i+1, height-1) * width + min(j+1, width-1)
        ]
        
        # Crear las aristas
        nb = [(indice_actual, nb) for nb in nbh]
        
        # Agregar las aristas al arreglo E
        # E.append(nb)

        N.append(n)
        E.append(nb)
        sigma = (10**-6) + calculate_sigma(e)
        w = calculate_wijk(e,sigma)
        d = sum(w)
        D.append(d)
        W_e.append(w)



D_m = np.diag(np.sum(W_e, axis=1))


W = np.zeros((height * width, height * width))
for i in range(height * width):    
    for j in range(height * width):
        # Calcular la matriz de adyacencia ponderada W
        index = (i, j)
        for e in E:
            if index in e: 
                W[i, j] = W_e[i][e.index(index)]

# Imprimir la matriz de adyacencia ponderada W
print("Matriz de adyacencia ponderada W:")

S = B + F  # Concatenar las listas B y F
S_indices = [V.index(pixel) for pixel in S if pixel in V]  # Obtener los índices de los elementos en S que están en V

I_s = np.zeros((height * width, height * width))  # Inicializar matriz diagonal de ceros
np.fill_diagonal(I_s, 1, list(S_indices))  # Establecer los elementos correspondientes a 1

xB = 1
xF = 0.1

L = D_m - W

b = np.zeros((height * width, 1))
for pixel in V:
    if pixel in B:        
        b[V.index(pixel)] = xB
    if pixel in F:        
        b[V.index(pixel)] = xF    



print("Vector b:")
# print(V)
# print(b)

x = np.linalg.solve(I_s + L**2, b)
y = np.zeros_like(x)
for i in range(len(x)):
    if x[i] >= (xB + xF) / 2:
        y[i] = xB
    else:
        y[i] = xF


print("Solución x:")

nueva_img_data = y.reshape(img_data.shape)
print(nueva_img_data)
print(x)
print(y)
# print(L)
