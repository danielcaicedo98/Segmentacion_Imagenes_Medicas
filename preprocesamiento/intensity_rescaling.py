import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tkinter import filedialog

def Intensity_Rescaling(img):
    # min = np.min(img[img > 10])
    # max = np.max(img[img > 10])
    
    img_data = img.get_fdata()
    rescaling = np.zeros(img.shape)
    min = np.min(img_data[img_data > 10])
    max = np.max(img_data)
    rescaling = (img_data - min) / (max - min)   
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    # print(img_data[96,88,96])    
    # print(rescaling[96,88,96])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        # img_nifti = nib.Nifti1Image(rescaling, img.affine) 
        # nib.save(img_nifti, file_path)
        img_nifti  = nib.Nifti1Image(rescaling, img.affine, img.header) # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")


