# # def encontrar_aristas_imagen(m, n):
# #     """
# #     Encuentra el conjunto de aristas para una imagen bidimensional de tamaño m x n.

# #     Parámetros:
# #         m: int
# #             Número de filas en la imagen.
# #         n: int
# #             Número de columnas en la imagen.

# #     Devuelve:
# #         E: list of tuples
# #             Lista de aristas del grafo.
# #     """
# #     E = []

# #     for i in range(m):
# #         for j in range(n):
# #             # Arriba a la izquierda
# #             if i > 0 and j > 0:
# #                 E.append(((i, j), (i-1, j-1)))
# #             # Arriba
# #             if i > 0:
# #                 E.append(((i, j), (i-1, j)))
# #             # Arriba a la derecha
# #             if i > 0 and j < n-1:
# #                 E.append(((i, j), (i-1, j+1)))
# #             # Izquierda
# #             if j > 0:
# #                 E.append(((i, j), (i, j-1)))
# #             # Derecha
# #             if j < n-1:
# #                 E.append(((i, j), (i, j+1)))
# #             # Abajo a la izquierda
# #             if i < m-1 and j > 0:
# #                 E.append(((i, j), (i+1, j-1)))
# #             # Abajo
# #             if i < m-1:
# #                 E.append(((i, j), (i+1, j)))
# #             # Abajo a la derecha
# #             if i < m-1 and j < n-1:
# #                 E.append(((i, j), (i+1, j+1)))

# #     return E

# # # Ejemplo de uso
# # m = 3  # Número de filas en la imagen
# # n = 3  # Número de columnas en la imagen

# # E = encontrar_aristas_imagen(m, n)
# # print("Conjunto de aristas E:")
# # print(E)
# import numpy as np

# # Definir las dimensiones de la matriz
# filas = 3
# columnas = 4

# # Crear una matriz 2D llena de ceros
# matriz = np.zeros((filas, columnas))

# # Llenar la matriz con valores de ejemplo
# for i in range(filas):
#     for j in range(columnas):
#         matriz[i, j] = i * columnas + j

# # Imprimir la matriz
# print("Matriz 2D:")
# print(matriz)
import numpy as np

# Definir los arreglos B y F
B = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)]
F = [(2, 3, 3), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 1, 0), (3, 1, 1), (3, 1, 2),
     (3, 1, 3), (3, 2, 0), (3, 2, 1), (3, 2, 2)]

# Eliminar el último elemento de cada subarreglo
B_2D = np.array([list(pixel[:-1]) for pixel in B])
F_2D = np.array([list(pixel[:-1]) for pixel in F])

print("Arreglo B en 2D:")
print(B_2D)
print("\nArreglo F en 2D:")
print(F_2D)

import numpy as np
import math
from scipy.optimize import minimize
import nibabel as nib
from tkinter import filedialog

# img = nib.load('imagen.nii')
def laplacian_coordinates_segmentation(img,B,F):
    img_data = img.get_fdata()    
    height, width , p = img_data.shape

    # B = list(set(B))  
    # F = list(set(F))  
    B = np.array([list(pixel[:-1]) for pixel in B])
    F = np.array([list(pixel[:-1]) for pixel in F])
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
            # print(e)

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


    # print("Solución x:")

    nueva_img_data = y.reshape(img_data.shape)
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(nueva_img_data, img.affine)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")
