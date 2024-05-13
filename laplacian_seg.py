import numpy as np
import math
from scipy.optimize import minimize
import nibabel as nib
from tkinter import filedialog

img = nib.load('imagen_recortada_2.nii')
# img_data = img.get_fdata()
 
# B = [
#     (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 5), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 0, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 4), (4, 1, 3), (4, 1, 3), (4, 1, 3), (4, 1, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 2, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 3, 3), (4, 4, 3), (4, 4, 3), (4, 4, 3), (4, 4, 3), (4, 4, 3), (4, 4, 3), (4, 4, 3), (4, 4, 3), (4, 4, 3), (4, 4, 4), (4, 4, 4), (4, 4, 4), (4, 4, 4), (4, 4, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 4), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 5), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 5, 6), (4, 6, 6), (4, 6, 6), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 6, 7), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6), (4, 0, 6)]

# F = [
#     (4, 0, 3), (4, 0, 3), (4, 0, 3), (4, 0, 3), (4, 0, 3), (4, 0, 3), (4, 0, 3), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 2), (4, 0, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 1, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 3, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 4, 2), (4, 5, 2), (4, 5, 2), (4, 5, 2), (4, 5, 2), (4, 5, 2), (4, 5, 2), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 5, 3), (4, 6, 3), (4, 6, 3), (4, 6, 3), (4, 6, 3), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 4), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 6, 5), (4, 7, 5), (4, 7, 5), (4, 7, 5), (4, 7, 5), (4, 7, 5), (4, 7, 5), (4, 7, 5), (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6), (4, 7, 6)]

# B = list(set(B))
# F = list(set(F))
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

F = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)]
B = [(2, 3, 3), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 1, 0), (3, 1, 1), (3, 1, 2),
    (3, 1, 3), (3, 2, 0), (3, 2, 1), (3, 2, 2)]
height, width, depth, = img_data.shape

B = list(set(B))  
F = list(set(F))  
V = [(d, h, w) for d in range(height) for h in range(width) for w in range(depth)]
E = []
W = []
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
        for k in range(depth):
            # Obtener el valor del voxel actual     
            cont += 1       
            voxel_actual = img_data[i, j, k]
            # Obtener los valores de los 6 vecinos
            vecinos = [
                        img_data[min(i+1, height-1), j, k], img_data[max(i-1, 0), j, k],
                        img_data[i, min(j+1, width-1), k], img_data[i, max(j-1, 0), k],
                        img_data[i, j, min(k+1, depth-1)], img_data[i, j, max(k-1, 0)]
                    ]
            e = [
                (voxel_actual,vecinos[0]) , (voxel_actual,vecinos[1]),
                (voxel_actual,vecinos[2]) , (voxel_actual,vecinos[3]),
                (voxel_actual,vecinos[4]) , (voxel_actual,vecinos[5])
            ]

            n = [
                V.index((min(i+1, height-1), j, k)), V.index((max(i-1, 0), j, k)),
                V.index((i, min(j+1, width-1), k)), V.index((i, max(j-1, 0), k)),
                V.index((i, j, min(k+1, depth-1))), V.index((i, j, max(k-1, 0)))
            ]
            N.append(n)
            E.append(e)
            sigma = (10**-6) + calculate_sigma(e)
            w = calculate_wijk(e,sigma)
            d = sum(w)
            D.append(d)
            W.append(w)

k1 = 1
k2 = 1
k3 = 1

xB = 1
xF = 0.1



def loss_function(x):
    suma_b = 0
    suma_f = 0
    suma_v = 0
    Ex = 0
    for i in range(len(V)):   
        if  V[i] in B:
            suma_b += k1 * np.linalg.norm(x[i] - xB)**2 
        if  V[i] in F:    
            suma_f = k2 * np.linalg.norm(x[i] - xF)**2
        if V[i] in V:    
            suma_v = k3 * np.linalg.norm(x[i]*D[i] - sum(W[i][j]*x[j] for j in range(0,len(N[i]))) )**2    
            # suma_v = k3 * np.linalg.norm(x[i]*D[i] - sum(W[i][N[i][j]]*x[j] for j in range(0,len(N[i]))) )**2            
                    
    Ex += suma_b + suma_f + suma_v
    return Ex   

x_initial = np.zeros(len(V))

# Minimización de la función de pérdida
# result = minimize(loss_function, x_initial, method='CG')

result = minimize(loss_function, x_initial,method='BFGS', tol=1e-3)

# Los valores óptimos de x
x_optimal = result.x
# print(len(x_optimal))
# print("Valores óptimos de x que minimizan En:")
# print(x_optimal)
y = np.zeros(len(V))


for i in range(len(V)):
    if (x_optimal[i] > ((xB + xF) / 2)):
        
        y[i] = xB
    else:
        
        y[i] = xF    

# file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
# if file_path:
#     # Crear un objeto Nibabel con los datos de la segmentación
#     img_nifti = nib.Nifti1Image(nueva_img_data, img.affine)
#     nib.save(img_nifti, file_path)
#     print(f"Segmentación guardada como '{file_path}'")




# new_img = y.reshape(img_data.shape)
# imagen_nueva = nib.Nifti1Image(new_img, np.eye(4))
# nib.save(imagen_nueva, 'imagen_deg_lap.nii')

# print(E[0])
print(W[11])