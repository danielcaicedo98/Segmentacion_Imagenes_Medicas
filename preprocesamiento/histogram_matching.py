import numpy as np
import nibabel as nib
import scipy.stats
from tkinter import filedialog

def hist_match(img, k):
    train = img.get_fdata()
    f, landmarks = training(train, k)
    hm_img = testing(landmarks, train, f)
    # img_nifti = nib.Nifti1Image(hm_img, img.affine)
    # nib.save(img_nifti, 'hs.nii')
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(hm_img, img.affine)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")

def training(train, x):
    percentiles = np.linspace(5, 95, x)
    landmarks = np.percentile(train.flatten(), percentiles)
    functions = []
    for i in range(1, len(landmarks)):
        m = (landmarks[i] - landmarks[i-1]) / (percentiles[i] - percentiles[i-1])
        b = landmarks[i-1] - m * percentiles[i-1]
        f = lambda x, m=m, b=b: m * x + b
        functions.append(f)
    return functions, landmarks

def testing(landmarks, data, functions):
    hm_img = np.zeros(data.shape)
    samples = np.random.choice(data.flatten(), size=1000, replace=False)  # Tomar 1000 muestras aleatorias
    percentiles = scipy.stats.percentileofscore(samples, data)
    for i in range(len(landmarks) - 1):
        hm_img[(percentiles > landmarks[i]) & (percentiles < landmarks[i+1])] = functions[i](percentiles[(percentiles > landmarks[i]) & (percentiles < landmarks[i+1])])
    return hm_img

# Cargar la imagen
# img = nib.load('imagen.nii')


# hist_match(img, 5)
