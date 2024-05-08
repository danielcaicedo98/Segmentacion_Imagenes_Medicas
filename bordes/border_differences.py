import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tkinter import filedialog
import scipy

# Cargar la imagen
# img = nib.load('../imagen.nii')
def Border_Differences(img):
    img_data = img.get_fdata()

    # Definir el kernel para la dirección X
    kernel = np.array([
        [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]
    ]) / 9
    border = scipy.ndimage.convolve(img_data, kernel)

    # Calcular la magnitud del gradiente de los bordes
    magnitude = (border - img_data) > 25

    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(magnitude.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")

# Aplicar un umbral en la magnitud del gradiente para resaltar los bordes
# plt.imshow((img_data[:, :, 117] - border[:, :, 117]) > 15)  # Mostrar la sección en Z=117
# plt.show()
