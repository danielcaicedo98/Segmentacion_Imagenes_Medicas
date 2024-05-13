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
    moving_image = sitk.ReadImage(img)
    fixed_image = sitk.ReadImage(filename)

    # Se inicializa la transformación
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),                            # Transformación de Euler en 3D
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Se inicializa el método de registro de imágenes
    registration_method = sitk.ImageRegistrationMethod()

    # Configuración de la métrica de similitud.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)    # Se usa la métrica de información mutua de Mattes
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)            # Se utiliza muestreo aleatorio para la métrica
    registration_method.SetMetricSamplingPercentage(0.01)                                # Se utiliza el 1% de la imagen para el cálculo de la métrica

    registration_method.SetInterpolator(sitk.sitkLinear)                                # Se utiliza interpolación lineal

    # Configuración del optimizador.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,                                                                # Tasa de aprendizaje del optimizador
        numberOfIterations=100,                                                          # Número máximo de iteraciones
        convergenceMinimumValue=1e-6,                                                    # Valor mínimo de convergencia
        convergenceWindowSize=10,                                                        # Tamaño de la ventana de convergencia
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()                            # Se escalan los parámetros del optimizador según los desplazamientos físicos

    # Configuración del framework de multiresolución.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])                # Factores de reducción por nivel
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])            # Sigmas de suavizado por nivel
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()                   # Se especifican las sigmas de suavizado en unidades físicas

    # Se establecen las transformaciones inicial y optimizada.
    optimized_transform = sitk.AffineTransform(3)
    registration_method.SetMovingInitialTransform(initial_transform)                    # Transformación inicial de la imagen móvil
    registration_method.SetInitialTransform(optimized_transform, inPlace=True)           # Transformación inicial del optimizador

    # Se ejecuta el registro de imágenes.
    final_transform = registration_method.Execute(fixed_image, moving_image)          # Se ejecuta el registro y se obtiene la transformación final

    # Se verifica la razón por la que terminó la optimización.
    print("Valor métrico final: {0}".format(registration_method.GetMetricValue()))       # Se imprime el valor de la métrica final
    print(
        "Condición de parada del optimizador: {0}".format(
            registration_method.GetOptimizerStopConditionDescription()
        )
    )

    # Se aplica la transformación final a la imagen móvil.
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0)

    # Se guarda la imagen móvil registrada.
    sitk.WriteImage(moving_resampled, 'segunda_imagen_registrada.nii')

    # Se convierten las imágenes a arrays numpy para visualizarlas con matplotlib.
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_resampled_array = sitk.GetArrayFromImage(moving_resampled)

    # Se visualizan las imágenes.
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_array[64,:,:], cmap='gray')
    plt.title('Imagen fija')
    plt.subplot(1, 2, 2)
    plt.imshow(moving_resampled_array[64,:,:], cmap='gray')
    plt.title('Imagen móvil registrada')
    plt.show()



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
    # initial_transform = registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
    #     fixed_image,
    #     moving_image,
    #     sitk.Euler3DTransform(),
    #     sitk.CenteredTransformInitializerFilter.GEOMETRY,
    # ), inPlace=False)

    # optimized_transform = sitk.AffineTransform(3)
    # registration_method.SetMovingInitialTransform(initial_transform)
    # registration_method.SetInitialTransform(optimized_transform, inPlace=True)

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











