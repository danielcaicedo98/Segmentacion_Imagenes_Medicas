# import SimpleITK as sitk
# import numpy as np
# import matplotlib.pyplot as plt

# # Cargamos las imágenes usando SimpleITK
# fixed_image = sitk.ReadImage('imagen.nii')
# moving_image = sitk.ReadImage('segunda_imagen.nii')

# initial_transform = sitk.CenteredTransformInitializer(
#     fixed_image,
#     moving_image,
#     sitk.Euler3DTransform(),
#     sitk.CenteredTransformInitializerFilter.GEOMETRY,
# )


# registration_method = sitk.ImageRegistrationMethod()

# # Similarity metric settings.
# registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
# registration_method.SetMetricSamplingPercentage(0.01)

# registration_method.SetInterpolator(sitk.sitkLinear)

# # Optimizer settings.
# registration_method.SetOptimizerAsGradientDescent(
#     learningRate=1.0,
#     numberOfIterations=100,
#     convergenceMinimumValue=1e-6,
#     convergenceWindowSize=10,
# )
# registration_method.SetOptimizerScalesFromPhysicalShift()

# # Setup for the multi-resolution framework.
# registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
# registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
# registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# # Set the initial moving and optimized transforms.
# optimized_transform = sitk.Euler3DTransform()
# registration_method.SetMovingInitialTransform(initial_transform)
# registration_method.SetInitialTransform(optimized_transform, inPlace=False)

# # Connect all of the observers so that we can perform plotting during registration.
# # registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
# # registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
# # registration_method.AddCommand(
# #     sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations
# # )
# # registration_method.AddCommand(
# #     sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method)
# # )

# # Need to compose the transformations after registration.
# final_transform_v4 = sitk.CompositeTransform(
#     [registration_method.Execute(fixed_image, moving_image), initial_transform]
# )

# # Always check the reason optimization terminated.
# print("Final metric value: {0}".format(registration_method.GetMetricValue()))
# print(
#     "Optimizer's stopping condition, {0}".format(
#         registration_method.GetOptimizerStopConditionDescription()
#     )
# )
# moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform_v4, sitk.sitkLinear, 0.0)
# # Aplicamos la transformación final a la imagen móvil

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
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Cargamos las imágenes usando SimpleITK
fixed_image = sitk.ReadImage('imagen.nii')
moving_image = sitk.ReadImage('segunda_imagen.nii')

initial_transform = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
)


registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
registration_method.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10,
)
registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Set the initial moving and optimized transforms.
optimized_transform = sitk.Euler3DTransform()
registration_method.SetMovingInitialTransform(initial_transform)
registration_method.SetInitialTransform(optimized_transform, inPlace=True)

# Connect all of the observers so that we can perform plotting during registration.
# registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
# registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
# registration_method.AddCommand(
#     sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations
# )
# registration_method.AddCommand(
#     sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method)
# )

# Need to compose the transformations after registration.
final_transform_v4 = registration_method.Execute(fixed_image, moving_image)

# Always check the reason optimization terminated.
print("Final metric value: {0}".format(registration_method.GetMetricValue()))
print(
    "Optimizer's stopping condition, {0}".format(
        registration_method.GetOptimizerStopConditionDescription()
    )
)

# Aplicamos la transformación final a la imagen móvil
moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform_v4, sitk.sitkLinear, 0.0)

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
