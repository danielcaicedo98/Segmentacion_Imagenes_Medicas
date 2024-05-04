import SimpleITK as sitk

# Cargar las imágenes que se desean registrar
fixed_image = sitk.ReadImage("imagen.nii")
moving_image = sitk.ReadImage("segunda_imagen.nii")

# Crear un transformador para alinear la imagen móvil con la imagen fija
initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

# Parámetros del registro
registration_method = sitk.ImageRegistrationMethod()

# Función de optimización
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

# Transformador a optimizar
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                  numberOfIterations=100, 
                                                  convergenceMinimumValue=1e-6, 
                                                  convergenceWindowSize=10)

# Tipo de transformación
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Realizar el registro
final_transform = registration_method.Execute(fixed_image, moving_image)

# Aplicar la transformación a la imagen movil
registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

# Guardar la imagen registrada
sitk.WriteImage(registered_image, "imagen_movil_registrada.nii")
