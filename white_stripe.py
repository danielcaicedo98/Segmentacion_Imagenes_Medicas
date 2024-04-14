import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from tkinter import filedialog
# Cargar la imagen
img = nib.load('imagen.nii')
img_data = img.get_fdata()
img_data = img_data[img_data > 10]
img_original = img.get_fdata()

# Calcular el histograma de intensidades
hist, bins = np.histogram(img_data.flatten(), bins=50)

# Encontrar los picos en el histograma
peaks, _ = find_peaks(hist)

# Calcular las prominencias de los picos
prominences = peak_prominences(hist, peaks)[0]

# Calcular la combinación de prominencia y altura para cada pico
combined_score = prominences + hist[peaks]

# Ordenar los picos según su combinación de prominencia y altura
peaks_sorted_by_score = sorted(zip(peaks, combined_score), key=lambda x: x[1], reverse=True)

# Número de picos representativos que deseas encontrar
num_picos_representativos = 5  # Puedes ajustar este valor según tu necesidad

# Seleccionar los primeros N picos representativos
picos_representativos = [p[0] for p in peaks_sorted_by_score[:num_picos_representativos]]

ws_i = bins[picos_representativos[-1]]
ws = img_original / ws_i

file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
if file_path:
    # Crear un objeto Nibabel con los datos de la segmentación
    img_nifti = nib.Nifti1Image(ws.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
    nib.save(img_nifti, file_path)
    print(f"Segmentación guardada como '{file_path}'")

# # Visualizar el histograma y marcar los picos representativos
# plt.figure()
# plt.plot(bins[:-1], hist, color='b')
# plt.plot(bins[picos_representativos], hist[picos_representativos], "rx", label='Picos representativos')
# plt.legend()
# plt.xlabel('Intensidad')
# plt.ylabel('Frecuencia')
# plt.title('Histograma de Intensidades')
# plt.show()

# # Imprimir los valores de intensidad de los picos representativos
# print("Los picos representativos tienen intensidades de:", bins[picos_representativos])