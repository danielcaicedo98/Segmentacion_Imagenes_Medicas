from tkinter import filedialog
import tkinter as tk
import numpy as np
import nibabel as nib

def save_region_growing(data):
    img = data.get_fdata()

    # Crear una máscara binaria para almacenar la segmentación
    img_th = np.zeros_like(img)

    # Punto inicial
    # seed_point = (110, 20, 100)
    image_shape = img.shape
    seed_point = tuple(dim // 2 for dim in image_shape)

    # Valor de tolerancia
    tolerancia = 80

    # Cola para almacenar puntos por visitar
    cola = [seed_point]

    # Movimientos
    def movimiento(point, direction):
        x, y, z = point
        if direction == 'derecha':
            return x + 1, y, z
        elif direction == 'izquierda':
            return x - 1, y, z
        elif direction == 'arriba':
            return x, y + 1, z
        elif direction == 'abajo':
            return x, y - 1, z
        elif direction == 'adelante':
            return x, y, z + 1
        elif direction == 'atras':
            return x, y, z - 1

    # Función para realizar desplazamientos
    arr_intensity = []
    arr_intensity.append(img[seed_point])
    def desplazamiento(point):
        mean_intensity = np.mean(arr_intensity)
        # Definir los vecinos
        vecinos = ['arriba', 'abajo', 'derecha', 'izquierda', 'atras','adelante']

        for vecino in vecinos:
            new_point = movimiento(point, vecino)
            if new_point not in recorrido:
                if abs(img[new_point] - mean_intensity) <= tolerancia:
                    img_th[new_point] = 1
                    arr_intensity.append(img[new_point])
                    recorrido.append(new_point)
                    cola.append(new_point)
                else:
                    img_th[new_point] = 0

    # Inicializar lista de puntos recorridos
    recorrido = [seed_point]

    # Iterar hasta que la cola esté vacía o hasta alcanzar un número máximo de iteraciones
    max_iterations = 1000
    for _ in range(max_iterations):
        if not cola:
            break
        point = cola.pop(0)
        desplazamiento(point)

    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(img_th.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")
