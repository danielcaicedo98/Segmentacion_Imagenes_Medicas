import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

img = nib.load('imagen.nii')
img_data = img.get_fdata()

mean_value = img_data[img_data > 10].mean()
std_value = img_data[img_data > 10].std()
img_zscore = (img_data - mean_value) / std_value

nueva_img = nib.Nifti1Image(img_zscore.astype(np.uint8), np.eye(4))

# Guardar la nueva imagen en un archivo
nib.save(nueva_img, 'img.nii')
