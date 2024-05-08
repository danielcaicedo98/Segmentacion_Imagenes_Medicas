
import nibabel as nib
import numpy as np

def recortar_imagen(imagen, tamaño_recorte):
    # Obtener los datos de la imagen
    img = nib.load(imagen)
    img_data = img.get_fdata()

    # Obtener dimensiones de la imagen
    dim_x, dim_y, dim_z = img.shape

    # Calcular coordenadas del centro
    centro_x = dim_x // 2
    centro_y = dim_y // 2
    centro_z = dim_z // 2

    # Calcular límites del recorte
    inicio_x = centro_x - tamaño_recorte // 2
    fin_x = inicio_x + tamaño_recorte
    inicio_y = centro_y - tamaño_recorte // 2
    fin_y = inicio_y + tamaño_recorte
    inicio_z = centro_z - tamaño_recorte // 2
    fin_z = inicio_z + tamaño_recorte

    # Recortar la imagen
    imagen_recortada = img_data[inicio_x:fin_x, inicio_y:fin_y, inicio_z:fin_y]

    return imagen_recortada

# Ruta de la imagen
ruta_imagen = 'imagen.nii'
# Tamaño del recorte
tamaño_recorte = 8

# Obtener imagen recortada
imagen_recortada = recortar_imagen(ruta_imagen, tamaño_recorte)

# Guardar la imagen recortada
imagen_recortada_nueva = nib.Nifti1Image(imagen_recortada, np.eye(4))
nib.save(imagen_recortada_nueva, 'imagen_recortada_2.nii')






# import nibabel as nib
# import numpy as np

# def recortar_imagen(imagen, tamaño_recorte):
#     # Obtener los datos de la imagen
#     img = nib.load(imagen)
#     img_data = img.get_fdata()

#     # Obtener dimensiones de la imagen
#     dim_x, dim_y, dim_z = img.shape

#     # Calcular coordenadas del centro en cada dimensión
#     centro_x = dim_x // 2
#     centro_y = dim_y // 2
#     centro_z = dim_z // 2

#     # Calcular límites del recorte
#     inicio_x = centro_x - tamaño_recorte // 2
#     fin_x = inicio_x + tamaño_recorte
#     inicio_y = centro_y - tamaño_recorte // 2
#     fin_y = inicio_y + tamaño_recorte
#     inicio_z = centro_z - tamaño_recorte // 2
#     fin_z = inicio_z + tamaño_recorte

#     # Recortar la imagen
#     imagen_recortada = img_data[inicio_x:fin_x, inicio_y:fin_y, inicio_z:fin_z]

#     return imagen_recortada

# # Ruta de la imagen
# ruta_imagen = 'imagen.nii'
# # Tamaño del recorte
# tamaño_recorte = 64

# # Obtener imagen recortada
# imagen_recortada = recortar_imagen(ruta_imagen, tamaño_recorte)

# # Guardar la imagen recortada
# imagen_recortada_nueva = nib.Nifti1Image(imagen_recortada, np.eye(4))
# nib.save(imagen_recortada_nueva, 'imagen_recortada.nii')



# import nibabel as nib
# import numpy as np

# def recortar_imagen(imagen, tamaño_recorte):
#     # Obtener los datos de la imagen
#     img = nib.load(imagen)
#     img_data = img.get_fdata()

#     # Obtener dimensiones de la imagen
#     dim_x, dim_y, dim_z = img.shape

#     # Calcular coordenadas del centro
#     centro_x = dim_x // 2
#     centro_y = dim_y // 2

#     # Calcular límites del recorte
#     inicio_x = centro_x - tamaño_recorte // 2
#     fin_x = inicio_x + tamaño_recorte
#     inicio_y = centro_y - tamaño_recorte // 2
#     fin_y = inicio_y + tamaño_recorte

#     # Recortar la imagen
#     imagen_recortada = img_data[inicio_x:fin_x, inicio_y:fin_y, :]

#     return imagen_recortada

# # Ruta de la imagen
# ruta_imagen = 'imagen.nii'
# # Tamaño del recorte
# tamaño_recorte = 64

# # Obtener imagen recortada
# imagen_recortada = recortar_imagen(ruta_imagen, tamaño_recorte)

# # Guardar la imagen recortada
# imagen_recortada_nueva = nib.Nifti1Image(imagen_recortada, np.eye(4))
# nib.save(imagen_recortada_nueva, 'imagen_recortada.nii')
