import numpy as np
import math

img_data= np.array([[[10, 10, 10, 10],
                      [10, 20, 20, 10],
                      [10, 20, 20, 10],
                      [10, 10, 10, 10]],
                     
                     [[335, 5, 5, 5],
                      [5, 15, 15, 5],
                      [5, 15, 15, 5],
                      [5, 5, 5, 5]],

                     [[10, 10, 10, 10],
                      [10, 20, 20, 10],
                      [10, 20, 20, 10],
                      [10, 10, 10, 10]],

                    [[5, 5, 5, 5],
                      [5, 15, 15, 5],
                      [5, 15, 15, 5],
                      [5, 5, 5, 5]],
                      ])

# Crear una matriz para la nueva imagen con la misma forma que la imagen original
nueva_img_data = np.zeros_like(img_data)
shape = img_data.shape
print(shape)
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
    for i in edges:
        diff = np.abs(i[0] - i[1])      
        max_diff = np.max(diff)**2
        betha = 0.1 * max_diff
        div = betha / sigma
        exp = math.exp(-1 * div)  
        w.append(exp)
    print(w)   
    return w

for i in range(1, shape[0] - 1):    
    for j in range(1, shape[1] - 1):
        for k in range(1, shape[2] - 1):
            # Obtener el valor del voxel actual            
            voxel_actual = img_data[i, j, k]
            # Obtener los valores de los 4 vecinos
            vecinos = [
                        img_data[i+1, j, k], img_data[i-1, j, k],
                        img_data[i, j+1, k], img_data[i, j-1, k],
                        img_data[i, j, k+1], img_data[i, j, k-1]
                    ]
            e = [
                (voxel_actual,vecinos[0]) , (voxel_actual,vecinos[1]),
                (voxel_actual,vecinos[2]) , (voxel_actual,vecinos[3]),
                (voxel_actual,vecinos[4]) , (voxel_actual,vecinos[5])
            ]
            sigma = (10**-6) + calculate_sigma(e)
            calculate_wijk(e,sigma)
depth, height, width = img_data.shape
V = [(d, h, w) for d in range(depth) for h in range(height) for w in range(width)]
print(V)