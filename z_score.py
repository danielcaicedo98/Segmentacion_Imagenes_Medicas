import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tkinter import filedialog
# img = nib.load('imagen.nii')
def Z_Score(img):
    img_data = img.get_fdata()

    mean_value = img_data[img_data > 10].mean()
    std_value = img_data[img_data > 10].std()
    img_zscore = (img_data - mean_value) / std_value

    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(img_zscore, img.affine)  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")
