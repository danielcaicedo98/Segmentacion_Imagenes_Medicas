import SimpleITK as sitk
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import os

def registration_euler(img):    

    # root = tk.Tk()
    filename = filedialog.askopenfilename(filetypes=[("NIFTI files", "*.nii")])
    #(initialdir=os.getcwd(), title="Seleccionar imagen móvil", filetypes=(("NIfTI files", "*.nii"), ("all files", "*.*")))
    # root.withdraw()

    # Cargamos las imágenes usando SimpleITK en lugar de nibabel
    fixed_image = sitk.ReadImage(img)
    moving_image = sitk.ReadImage(filename)



    # moving_image = sitk.ReadImage('segunda_imagen.nii')

    # Inicializamos el método de registro
    registration_method = sitk.ImageRegistrationMethod()

    # Configuración de la métrica de similitud
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Configuración del optimizador
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )

    # No optimizamos in-place para poder ejecutar esta celda varias veces.
    registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    ), inPlace=False)


    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Siempre verificamos la razón por la que terminó la optimización.
    print("Valor final de la métrica: {0}".format(registration_method.GetMetricValue()))
    print(
        "Condición de parada del optimizador: {0}".format(
            registration_method.GetOptimizerStopConditionDescription()
        )
    )

    # Aplicamos la transformación final a la imagen móvil
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Guardamos la imagen móvil registrada
    sitk.WriteImage(moving_resampled, 'segunda_imagen_registrada.nii')

    # Convertimos las imágenes a arrays numpy para poder visualizarlas con matplotlib
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_resampled_array = sitk.GetArrayFromImage(moving_resampled)

    # Visualizamos las imágenes
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_array[64,:,:], cmap='gray')
    plt.title('Imagen fija')
    plt.subplot(1, 2, 2)
    plt.imshow(moving_resampled_array[64,:,:], cmap='gray')
    plt.title('Imagen móvil registrada')
    plt.show()






# root = tk.Tk()
# root.filename = filedialog.askopenfilename(filetypes=[("NIFTI files", "*.nii")])
# #(initialdir=os.getcwd(), title="Seleccionar imagen móvil", filetypes=(("NIfTI files", "*.nii"), ("all files", "*.*")))
# root.withdraw()

# # Cargamos las imágenes usando SimpleITK en lugar de nibabel
# fixed_image = sitk.ReadImage('../imagen.nii')
# moving_image = sitk.ReadImage(root.filename)



# # moving_image = sitk.ReadImage('segunda_imagen.nii')

# # Inicializamos el método de registro
# registration_method = sitk.ImageRegistrationMethod()

# # Configuración de la métrica de similitud
# registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
# registration_method.SetMetricSamplingPercentage(0.01)

# registration_method.SetInterpolator(sitk.sitkLinear)

# # Configuración del optimizador
# registration_method.SetOptimizerAsGradientDescent(
#     learningRate=1.0,
#     numberOfIterations=100,
#     convergenceMinimumValue=1e-6,
#     convergenceWindowSize=10,
# )

# # No optimizamos in-place para poder ejecutar esta celda varias veces.
# registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
#     fixed_image,
#     moving_image,
#     sitk.Euler3DTransform(),
#     sitk.CenteredTransformInitializerFilter.GEOMETRY,
# ), inPlace=False)


# final_transform = registration_method.Execute(fixed_image, moving_image)

# # Siempre verificamos la razón por la que terminó la optimización.
# print("Valor final de la métrica: {0}".format(registration_method.GetMetricValue()))
# print(
#     "Condición de parada del optimizador: {0}".format(
#         registration_method.GetOptimizerStopConditionDescription()
#     )
# )

# # Aplicamos la transformación final a la imagen móvil
# moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

# # Guardamos la imagen móvil registrada
# sitk.WriteImage(moving_resampled, 'segunda_imagen_registrada.nii')

# # Convertimos las imágenes a arrays numpy para poder visualizarlas con matplotlib
# fixed_array = sitk.GetArrayFromImage(fixed_image)
# moving_resampled_array = sitk.GetArrayFromImage(moving_resampled)

# # Visualizamos las imágenes
# plt.figure(figsize=(10,5))
# plt.subplot(1, 2, 1)
# plt.imshow(fixed_array[64,:,:], cmap='gray')
# plt.title('Imagen fija')
# plt.subplot(1, 2, 2)
# plt.imshow(moving_resampled_array[64,:,:], cmap='gray')
# plt.title('Imagen móvil registrada')
# plt.show()
