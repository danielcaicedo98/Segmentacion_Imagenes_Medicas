import numpy as np
import nibabel as nib
from tkinter import filedialog

def Median_Filter(img):
    def mediana_voxel_y_vecinos(voxel, vecinos):
        valores = [voxel] + vecinos
        mediana = np.median(valores)
        return mediana

    # Cargar la imagen
    # img = nib.load('imagen.nii')
    img_data = img.get_fdata()

    # Crear una matriz para la nueva imagen con la misma forma que la imagen original
    nueva_img_data = np.zeros_like(img_data)

    # Obtener la forma de la imagen
    shape = img_data.shape

    # Iterar sobre cada voxel en la imagen
    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            for k in range(1, shape[2] - 1):
                # Obtener el valor del voxel actual
                voxel_actual = img_data[i, j, k]
                # Obtener los valores de los 4 vecinos
                vecinos = [img_data[i+1, j, k], img_data[i-1, j, k],
                        img_data[i, j+1, k], img_data[i, j-1, k],
                        img_data[i, j, k+1], img_data[i, j, k-1]]
                # Calcular el promedio del voxel y sus vecinos
                valor_promedio = mediana_voxel_y_vecinos(voxel_actual, vecinos)
                # Guardar el valor del promedio en la nueva imagen
                nueva_img_data[i, j, k] = valor_promedio

    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un omeanbjeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(nueva_img_data, img.affine)
        nib.save(img_nifti, file_path)
        # img_nifti = nib.Nifti1Image(nueva_img_data.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        # nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")            

# Crear una nueva imagen a partir de la matriz de datos de la nueva imagen
# nueva_img = nib.Nifti1Image(nueva_img_data, img.affine)

# # Guardar la nueva imagen en un archivo
# nib.save(nueva_img, 'k_means.nii')
