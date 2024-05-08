from tkinter import filedialog
import tkinter as tk
import numpy as np
import nibabel as nib

def save_segmentation_isodata(data):
    # Algoritmo Isodata
    img = data.get_fdata()
    delta = 0.1
    tau_init = 100
    t = 0
    tau_t = tau_init
    while True:
        img_th = img > tau_t
        m_foreground = img[img_th == 1].mean()
        m_background = img[img_th == 0].mean()
        tau_new = 0.5 * (m_background + m_foreground)

        if abs(tau_new - tau_t) < delta:
            break
        tau_t = tau_new

    # Guardado del archivo
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(img_th.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")
