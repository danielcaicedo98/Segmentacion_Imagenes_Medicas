import numpy as np
import math
from scipy.optimize import minimize
import nibabel as nib
from tkinter import filedialog

def laplacian_coordinates_segmentation(img, B, F):
    img_data = img.get_fdata()    
    height, width , p = img_data.shape

    B = np.array([list(pixel[:-1]) for pixel in B])
    F = np.array([list(pixel[:-1]) for pixel in F])
    V = [(d, h) for d in range(height) for h in range(width)]
    E = []
    W_e = []
    N = []
    D = []
    
    nueva_img_data = np.zeros_like(img_data)

    def calculate_sigma(edges):
        sigma = 0
        for i in edges:
            diff = np.abs(i[0] - i[1])
            max_diff = np.max(diff)
            if max_diff > sigma:
                sigma = max_diff
        return sigma

    def calculate_wijk(edges, sigma):    
        w = []
        β = 0.1
        for i in edges:
            diff = np.abs(i[0] - i[1])      
            max_diff = β * np.max(diff)**2        
            div = -1 * (max_diff / sigma)
            exp = math.exp(div)  
            w.append(exp)
        return w
    
    for i in range(height):    
        for j in range(width):
            voxel_actual = img_data[i, j]
            
            vecinos = [
                img_data[max(i-1, 0), max(j-1, 0)], img_data[max(i-1, 0), j],
                img_data[max(i-1, 0), min(j+1, width-1)], img_data[i, max(j-1, 0)],
                img_data[i, min(j+1, width-1)], img_data[min(i+1, height-1), max(j-1, 0)],
                img_data[min(i+1, height-1), j], img_data[min(i+1, height-1), min(j+1, width-1)]
            ]
            
            e = [
                (voxel_actual, vecinos[0]), (voxel_actual, vecinos[1]),
                (voxel_actual, vecinos[2]), (voxel_actual, vecinos[3]),
                (voxel_actual, vecinos[4]), (voxel_actual, vecinos[5]),
                (voxel_actual, vecinos[6]), (voxel_actual, vecinos[7])
            ]

            n = [
                V.index((max(i-1, 0), max(j-1, 0))), V.index((max(i-1, 0), j)),
                V.index((max(i-1, 0), min(j+1, width-1))), V.index((i, max(j-1, 0))),
                V.index((i, min(j+1, width-1))), V.index((min(i+1, height-1), max(j-1, 0))),
                V.index((min(i+1, height-1), j)), V.index((min(i+1, height-1), min(j+1, width-1)))
            ]

            indice_actual = i * width + j
            
            nbh = [
                max(i-1, 0) * width + max(j-1, 0), max(i-1, 0) * width + j,
                max(i-1, 0) * width + min(j+1, width-1), i * width + max(j-1, 0),
                i * width + min(j+1, width-1), min(i+1, height-1) * width + max(j-1, 0),
                min(i+1, height-1) * width + j, min(i+1, height-1) * width + min(j+1, width-1)
            ]
            
            nb = [(indice_actual, nb) for nb in nbh]
            
            N.append(n)
            E.append(nb)
            sigma = (10**-6) + calculate_sigma(e)
            w = calculate_wijk(e, sigma)
            d = sum(w)
            D.append(d)
            W_e.append(w)

    D_m = np.diag(np.sum(W_e, axis=1))

    S = np.concatenate((B,F))
    # S_indices = [V.index(pixel) for pixel in S if pixel in V]  
    S_indices = np.where(np.isin(V, S).all(axis=1))[0]

    I_s = np.zeros((height * width, height * width))  
    np.fill_diagonal(I_s, 1, list(S_indices))  

    xB = 1
    xF = 0.1

    L = D_m - np.array(W_e)

    b = np.zeros((height * width, 1))
    for pixel in V:
        if pixel in B:        
            b[V.index(pixel)] = xB
        if pixel in F:        
            b[V.index(pixel)] = xF    

    x = np.linalg.solve(I_s + L**2, b)
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] >= (xB + xF) / 2:
            y[i] = xB
        else:
            y[i] = xF

    nueva_img_data = y.reshape(img_data.shape)
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        img_nifti = nib.Nifti1Image(nueva_img_data, img.affine)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")
       