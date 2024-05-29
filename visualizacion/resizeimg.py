import nibabel as nib
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

# Cargar la imagen NIfTI
img = nib.load('../imagen.nii')
img_d = img.get_fdata()

# Obtener el índice de la rebanada a la mitad del volumen
slice_index = img_d.shape[0] // 2
img_data = img_d[:,slice_index,:]

# Reducir la imagen
factor_reduccion = 0.15 # Cambiar el factor de reducción según sea necesario
nueva_altura = int(img_data.shape[0] * factor_reduccion)
nueva_anchura = int(img_data.shape[1] * factor_reduccion)
img_data_resized = resize(img_data, (nueva_altura, nueva_anchura))


# Guardar la imagen como PNG con la resolución ajustada
plt.imsave('imagen_reducida.jpeg', img_data_resized, cmap='gray')

print("Imagen guardada como imagen_reducida.png")

