import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tkinter import filedialog

def Intensity_Rescaling(img):
    # min = np.min(img[img > 10])
    # max = np.max(img[img > 10])
    img_data = img.get_fdata()
    min = np.min(img_data[img_data > 10])
    max = np.max(img_data[img_data > 10])
    rescaling = (img_data - min) / (max - min)   
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])

    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(rescaling, img.affine) 
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")


