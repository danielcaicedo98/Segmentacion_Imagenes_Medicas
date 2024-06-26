import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tkinter import filedialog
import scipy

# Cargar la imagen
# img = nib.load('../imagen.nii')
def Border_Second(img):
    img_data = img.get_fdata()

    # Definir el kernel para la dirección X
    kernel_x = np.array([
        [[0, 0, 0],
        [-1, 0, 1],
        [0, 0, 0]]
    ]) / 2

    # Definir el kernel para la dirección Y
    kernel_y = np.array([
        [[0, -1, 0],
        [0, 0, 0],
        [0, 1, 0]]
    ]) / 2


    kernel_z = np.array([
        [[0, 0, 0],
        [0, -1, 0],
        [0, 0, 0]]
    ]) / 2


    # Aplicar el filtro de convolución en las direcciones X e Y para toda la imagen
    border_x = scipy.ndimage.convolve(img_data, kernel_x)
    border_y = scipy.ndimage.convolve(img_data, kernel_y)
    border_z = scipy.ndimage.convolve(img_data, kernel_z)

    border_D_x = scipy.ndimage.convolve(border_x, kernel_x)
    border_D_y = scipy.ndimage.convolve(border_y, kernel_y)
    border_D_z = scipy.ndimage.convolve(border_z, kernel_z)

    border_D_xy = scipy.ndimage.convolve(border_x, kernel_y)
    border_D_xz = scipy.ndimage.convolve(border_x, kernel_z)
    # Calcular la magnitud del gradiente de los bordes
    # magnitude = np.sqrt(border_D_x ** 2 + border_D_y ** 2 + border_D_z ** 2 + 2 * (border_D_xy ** 2 + border_D_xz ** 2)) 
    magnitude = np.sqrt(border_D_x ** 2 + border_D_y ** 2 +  border_D_xy) 


    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(magnitude.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")

# Aplicar un umbral en la magnitud del gradiente para resaltar los bordes
# plt.imshow(magnitude[:, :, 117] > 70)  # Mostrar la sección en Z=117
# plt.show()
