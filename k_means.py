import numpy as np
import nibabel as nib
# import matplotlib.pyplot as plt
from tkinter import filedialog
def k_means(imagen):

    def kmeans_segmentation(image_data, num_clusters, max_iterations=100):
        # Flatten the image data to 1D array
        flattened_data = image_data.flatten()
        
        # Centroides iniciales aleatorios
        centroids_indices = np.random.choice(flattened_data.size, size=num_clusters, replace=False) 
        centroids = flattened_data[centroids_indices] + 1
        print(centroids)
        
        # Iterate until convergence or max iterations reached
        for _ in range(max_iterations):
            # Assign each data point to the nearest centroid
            distances = np.abs(flattened_data[:, np.newaxis] - centroids)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([flattened_data[labels == k].mean() for k in range(num_clusters)])
            
            # Check for convergence
            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids
        
        # Assign labels back to original shape
        segmented_image = labels.reshape(image_data.shape)
        
        return segmented_image

    img = imagen
    image_data = img.get_fdata()

    num_clusters = 3  
    segmented_image = kmeans_segmentation(image_data, num_clusters)

    # Guardar imagen como archivo nifti
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        # Crear un objeto Nibabel con los datos de la segmentación
        img_nifti = nib.Nifti1Image(segmented_image.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        nib.save(img_nifti, file_path)
        print(f"Segmentación guardada como '{file_path}'")