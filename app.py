import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from visualizacion.visualizacion_3d import save_segmentation_3d
from segmentacion.k_means import k_means
from segmentacion.region_growing import save_region_growing
from segmentacion.isodata import save_segmentation_isodata
from filtros.median_filter import Median_Filter
from filtros.mean_filter import Mean_Filter
from preprocesamiento.white_stripe import White_Stripe
from preprocesamiento.intensity_rescaling import Intensity_Rescaling
from preprocesamiento.z_score import Z_Score
from preprocesamiento.histogram_matching import hist_match
from bordes.border_differences import Border_Differences
from bordes.border_first import Border_First
from bordes.border_second import Border_Second
from registration.registration import registration_euler
from segmentacion.laplacian_segmentation import laplacian_coordinates_segmentation



class NiftiViewer:
    def __init__(self, master, nifti_file):
        self.B = []
        self.F = []
        self.ruta = nifti_file
        self.master = master
        self.img = nib.load(nifti_file)
        self.data = self.img.get_fdata()
        self.shape = self.data.shape
        self.current_index_axial = self.shape[2] // 2
        self.current_index_sagittal = self.shape[0] // 2
        self.current_index_coronal = self.shape[1] // 2

        # Variables para las anotaciones
        self.annotations = []
        self.segmentation = []

        # Frame para la imagen
        self.image_frame = tk.Frame(master)
        self.image_frame.grid(row=0, column=1)
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.imshow(self.data[:, :, self.current_index_axial], cmap='gray')
        self.ax.set_title("Axial")
        self.ax.set_aspect('equal')
        self.ax.set_autoscale_on(False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        # Frame para el Slider
        self.slider_frame = tk.Frame(master)
        self.slider_frame.grid(row=1, column=0, padx=1, pady=1)

        # Label para el slider
        self.sagittal_label = tk.Label(self.slider_frame, text="Sagital")
        self.sagittal_label.grid(row=0, column=0)

        self.axial_label = tk.Label(self.slider_frame, text="Axial")
        self.axial_label.grid(row=1, column=0)

        self.coronal_label = tk.Label(self.slider_frame, text="Coronal")
        self.coronal_label.grid(row=2, column=0)

        # Slider Sagital
        self.sagittal_slider = tk.Scale(self.slider_frame, from_=0, to=self.shape[0]-1, orient=tk.HORIZONTAL,command=self.update_sagittal, width=10)
        self.sagittal_slider.set(self.current_index_sagittal)
        self.sagittal_slider.grid(row=0, column=1, sticky="ew")

        self.axial_slider = tk.Scale(self.slider_frame, from_=0, to=self.shape[2]-1, orient=tk.HORIZONTAL, command=self.update_axial, width=10)
        self.axial_slider.set(self.current_index_axial)
        self.axial_slider.grid(row=1, column=1, sticky="ew")

        self.coronal_slider = tk.Scale(self.slider_frame, from_=0, to=self.shape[1]-1, orient=tk.HORIZONTAL, command=self.update_coronal, width=10)
        self.coronal_slider.set(self.current_index_coronal)
        self.coronal_slider.grid(row=2, column=1, sticky="ew")

        # Botón para activar/desactivar modo de anotación
        self.annotation_mode = False
        self.annotation_buffer = []
        self.annotation_button = ttk.Button(master, text="  Modo Anotación   ", command=self.toggle_annotation_mode)
        self.annotation_button.grid(row=3, column=0, padx=10, pady=10)

        # Botón para activar/desactivar modo de segmentación
        self.segmentation_mode = False
        self.segmentation_buffer = []
        self.segmentation_button = ttk.Button(master, text="Modo Segmentación", command=self.toggle_segmentation_mode)
        self.segmentation_button.grid(row=4, column=0)

        # Botón para guardar las anotaciones
        # self.save_annotation_button = ttk.Button(master, text="Guardar Anotaciones", command=self.save_annotations)
        # self.save_annotation_button.grid(row=4, column=0, padx=10, pady=10)

        # # Botón para cargar las anotaciones
        # self.load_annotation_button = ttk.Button(master, text="Cargar Anotaciones", command=self.load_annotations)
        # self.load_annotation_button.grid(row=5, column=0, padx=10, pady=10)

        # Botón para guardar la segmentación
        # self.save_segmentation_button = ttk.Button(master, text="Guardar Segmentación", command=self.save_segmentation)
        # self.save_segmentation_button.grid(row=6, column=0, padx=10, pady=10)

        # # Botón para cargar la segmentación
        # self.load_segmentation_button = ttk.Button(master, text="Cargar Segmentación", command=self.load_segmentation)
        # self.load_segmentation_button.grid(row=7, column=0, padx=10, pady=10)

        #Botón para imprimir los voxeles con anotaciones
        # self.print_voxels_button = ttk.Button(master, text="Imprimir Voxeles con Anotaciones", command=self.print_annotations)
        # self.print_voxels_button.grid(row=1, column=3, padx=10, pady=10)

        # Botón para imprimir los voxeles con segmentación
        # self.print_segmented_voxels_button = ttk.Button(master, text="Imprimir Voxeles Segmentados", command=self.print_segmentation)
        # self.print_segmented_voxels_button.grid(row=1, column=3, padx=10, pady=10)

        # self.save_button = ttk.Button(master, text="Guardar crecimiento de regiones", command=self.save_region_growing,style='Custom.TButton', width=30)
        # self.save_button.grid(row=2, column=1, padx=10, pady=10)

        # Menú desplegable anotación
        self.anotacion_frame = tk.Frame(master)
        self.anotacion_frame.grid(row=5, column=1, padx=10, pady=10)
        self.selected_option_anotacion = tk.StringVar()
        self.selected_option_anotacion.set("Anotación")
        self.options_menu_anotacion = ttk.OptionMenu(self.anotacion_frame, self.selected_option_anotacion, "Anotación", "Guardar Anotacion", "Cargar Anotacion", "Visualizar 3D", command=self.call_selected_anotation)
        self.options_menu_anotacion.pack()

        # Menú desplegable segementacion
        self.menu_frame_seg = tk.Frame(master)
        self.menu_frame_seg.grid(row=2, column=1, padx=10, pady=10)
        self.selected_option_seg = tk.StringVar()
        self.selected_option_seg.set("Seleccione Segmentación")
        self.options_menu_seg = ttk.OptionMenu(self.menu_frame_seg, self.selected_option_seg, "Seleccione Segmentación", "Isodata", "Region Growing", "K-Means", "Laplaciana", command=self.call_selected_segmentation)
        self.options_menu_seg.pack()

        # Menú desplegable filtros
        self.menu_frame_filt = tk.Frame(master)
        self.menu_frame_filt.grid(row=3, column=1)
        self.selected_option_filt = tk.StringVar()
        self.selected_option_filt.set("Seleccione Filtro")
        self.options_menu_filt = ttk.OptionMenu(self.menu_frame_filt, self.selected_option_filt, "Seleccione Filtro", "Mean", "Median", command=self.call_selected_filter)
        self.options_menu_filt.pack()

        # Menú desplegable preprocesamiento
        self.menu_frame_prep = tk.Frame(master)
        self.menu_frame_prep.grid(row=4, column=1, padx=10, pady=10)
        self.selected_option_prep = tk.StringVar()
        self.selected_option_prep.set("Preprocesamiento")
        self.options_menu_prep = ttk.OptionMenu(self.menu_frame_prep, self.selected_option_prep,  "Preprocesamiento","Histogram Matching", "Intensity Rescaling", "Z-Score", "White Stripe",command=self.call_selected_preprocesing)
        self.options_menu_prep.pack()

        # Menú desplegable bordes
        self.menu_frame_bord = tk.Frame(master)
        self.menu_frame_bord.grid(row=2, column=2, padx=10, pady=10)
        self.selected_option_bord = tk.StringVar()
        self.selected_option_bord.set("Seleccione Borde")
        self.options_menu_bord = ttk.OptionMenu(self.menu_frame_bord, self.selected_option_bord,  "Bordes","Primera Derivada", "Segunda Derivada", "Diferencia",command=self.call_selected_border)
        self.options_menu_bord.pack()

        # Menú desplegable registro
        self.menu_frame_reg = tk.Frame(master)
        self.menu_frame_reg.grid(row=3, column=2, padx=10, pady=10)
        self.selected_option_reg = tk.StringVar()
        self.selected_option_reg.set("Seleccione Borde")
        self.options_menu_reg = ttk.OptionMenu(self.menu_frame_reg, self.selected_option_reg,  "Registro","Registro Euler",command=self.call_selected_registration)
        self.options_menu_reg.pack()






        # Configuración de eventos del ratón
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        if event.button == MouseButton.LEFT and self.annotation_mode:
            self.annotation_buffer = [(int(event.xdata), int(event.ydata))]

        if event.button == MouseButton.LEFT and self.segmentation_mode:
            self.segmentation_buffer = [(int(event.xdata), int(event.ydata))]

    def on_motion(self, event):
        if event.button == MouseButton.LEFT and self.annotation_mode:
            if self.annotation_buffer:
                x, y = int(event.xdata), int(event.ydata)
                self.annotation_buffer.append((x, y))
                self.ax.plot([self.annotation_buffer[-2][0], x], [self.annotation_buffer[-2][1], y], color='red')
                self.canvas.draw()

        if event.button == MouseButton.LEFT and self.segmentation_mode:
            if self.segmentation_buffer:
                x, y = int(event.xdata), int(event.ydata)
                self.segmentation_buffer.append((x, y))
                self.ax.plot([self.segmentation_buffer[-2][0], x], [self.segmentation_buffer[-2][1], y], color='green')
                self.canvas.draw()

    def on_release(self, event):
        if event.button == MouseButton.LEFT and self.annotation_mode:
            current_slice = self.get_current_slice()
            self.annotations += [(self.current_view, current_slice, self.annotation_buffer)]
            self.annotation_buffer = []

        if event.button == MouseButton.LEFT and self.segmentation_mode:
            current_slice = self.get_current_slice()
            self.segmentation += [(self.current_view, current_slice, self.segmentation_buffer)]
            self.segmentation_buffer = []

    def toggle_annotation_mode(self):
        self.annotation_mode = not self.annotation_mode
        if not self.annotation_mode:
            self.annotation_buffer = []

    def toggle_segmentation_mode(self):
        self.segmentation_mode = not self.segmentation_mode
        if not self.segmentation_mode:
            self.segmentation_buffer = []

    def save_annotations(self):
        np.save("annotations.npy", self.annotations)

    def load_annotations(self):
        try:
            self.annotations = np.load("annotations.npy", allow_pickle=True)
            self.draw_saved_annotations()
        except FileNotFoundError:
            print("No se encontraron anotaciones guardadas.")

    def voxels_annotation(self):
        print(self.annotations)

    # def draw_saved_annotations(self):
    #     self.ax.clear()
    #     self.ax.imshow(self.data[:, :, self.get_current_slice()], cmap='gray')
    #     self.ax.set_title(self.current_view.capitalize())
    #     self.ax.set_aspect('equal')
    #     self.ax.set_autoscale_on(False)
    #     for annotation in self.annotations:
    #         view, slice_index, voxel_list = annotation
    #         if view == self.current_view and slice_index == self.get_current_slice():
    #             self.ax.plot(*zip(*voxel_list), marker=None, color='red')
    #     self.canvas.draw()

    #     for segmentation in self.segmentation:
    #         view, slice_index, voxel_list = segmentation
    #         if view == self.current_view and slice_index == self.get_current_slice():
    #             self.ax.plot(*zip(*voxel_list), marker=None, color='green')
    #     self.canvas.draw()

    def draw_saved_segmentation(self):
        # print(self.current_view)
        axe = self.current_view
        self.ax.clear()
        if axe == "sagittal":
            self.ax.imshow(self.data[self.get_current_slice(), :, :], cmap='gray')
        if axe == "coronal":
            self.ax.imshow(self.data[:, self.get_current_slice(), :], cmap='gray')   
        if axe == "axial":
            self.ax.imshow(self.data[:, :, self.get_current_slice()], cmap='gray')    
        self.ax.set_title(self.current_view.capitalize())
        self.ax.set_aspect('equal')
        self.ax.set_autoscale_on(False)
        for segmentation in self.segmentation:
            view, slice_index, voxel_list = segmentation
            if view == self.current_view and slice_index == self.get_current_slice():
                self.ax.plot(*zip(*voxel_list), marker=None, color='green')
        self.canvas.draw()    

        for annotation in self.annotations:
            view, slice_index, voxel_list = annotation
            if view == self.current_view and slice_index == self.get_current_slice():
                self.ax.plot(*zip(*voxel_list), marker=None, color='red')
        self.canvas.draw()

    def save_segmentation(self):
        np.save("segmentation.npy", self.segmentation)

    def load_segmentation(self):
        try:
            self.segmentation = np.load("segmentation.npy", allow_pickle=True)
            self.draw_saved_segmentation()
        except FileNotFoundError:
            print("No se encontraron segmentaciones guardadas.")

    

    def update_sagittal(self, val):
        self.current_index_sagittal = int(val)
        self.current_view = "sagittal"        
        self.draw_saved_segmentation()
        # self.draw_saved_annotations()
        

    def update_axial(self, val):
        self.current_index_axial = int(val)
        self.current_view = "axial"
        # self.draw_saved_annotations()
        self.draw_saved_segmentation()

    def update_coronal(self, val):
        self.current_index_coronal = int(val)
        self.current_view = "coronal"
        # self.draw_saved_annotations()
        self.draw_saved_segmentation()

    def get_current_slice(self):
        if self.current_view == "axial":
            return self.current_index_axial
        elif self.current_view == "sagittal":
            return self.current_index_sagittal
        elif self.current_view == "coronal":
            return self.current_index_coronal

    def call_selected_segmentation(self, option):
        if option == "Laplaciana":
            self.return_annotations()
            self.return_segmentation()
            # print(self.B)
            # laplacian_coordinates_segmentation(self.img,self.B,self.F)
        elif option == "Region Growing":
            save_region_growing(self)
        elif option == "K-Means":
            k_means(self.img)
        elif option == "Isodata":
            save_segmentation_isodata(self.img)
            #"Isodata", "Region Growing", "K-Means"

    def call_selected_anotation(self, option):
        if option == "Guardar Anotacion":
            self.save_segmentation()
        elif option == "Cargar Anotacion":
            self.load_segmentation()
        elif option == "Visualizar 3D":
            save_segmentation_3d() 
        
    def call_selected_filter(self, option):
        if option == "Mean":
            Mean_Filter(self.img)
        elif option == "Median":
            Median_Filter(self.img)
             
    def call_selected_preprocesing(self, option):        
        if option == "Histogram Matching":
            hist_match(self.img,5)
        elif option == "Intensity Rescaling":
            Intensity_Rescaling(self.img)
        elif option == "Z-Score":
            Z_Score(self.img)
        elif option == "White Stripe":
            White_Stripe(self.img)

    def call_selected_border(self, option):
        #"Primera Derivada", "Segunda Derivada", "Diferencia"
        if option == "Primera Derivada":
            Border_First(self.img)            
        elif option == "Segunda Derivada":
            Border_Second(self.img)
        elif option == "Diferencia":
            Border_Differences(self.img)

    def call_selected_registration(self, option):
        if option == "Registro Euler":
            registration_euler(self.ruta)
        elif option == "Opción 2":
            self.option2_function()
        elif option == "Opción 3":
            self.option3_function()   

    def call_selected_option(self, option):
        if option == "Opción 1":
            self.option1_function()
        elif option == "Opción 2":
            self.option2_function()
        elif option == "Opción 3":
            self.option3_function()       


    def option1_function(self):
        print("Seleccionó la Opción 1")

    def option2_function(self):
        print("Seleccionó la Opción 2")

    def option3_function(self):
        print("Seleccionó la Opción 3")

    def return_segmentation(self):
        print("Voxels con anotaciones:")
        # print(self.annotations[0][0])
        for segmentation in self.segmentation:
            view, slice_index, voxel_list = segmentation
            # if view == self.current_view and slice_index == self.get_current_slice():            
            for voxel in voxel_list:
                if view == "sagittal":
                    self.F.append((slice_index, voxel[0], voxel[1]))
                elif view == "coronal":
                    self.F.append((voxel[0], slice_index, voxel[1]))    
                elif view == "axial":
                    self.F.append((voxel[0], voxel[1],slice_index))
        print(self.F) 

    def return_annotations(self):
        print("Voxels con anotaciones:")
        # print(self.annotations[0][0])
        for annotations in self.annotations:
            view, slice_index, voxel_list = annotations
            # if view == self.current_view and slice_index == self.get_current_slice():            
            for voxel in voxel_list:
                if view == "sagittal":
                    self.B.append((slice_index, voxel[0], voxel[1]))
                elif view == "coronal":
                    self.B.append((voxel[0], slice_index, voxel[1]))    
                elif view == "axial":
                    self.B.append((voxel[0], voxel[1],slice_index))                
        print(self.B)                
        # for annotation in self.annotations:
        #     view, slice_index, voxel_list = annotation
        #     print(f"Vista: {view.capitalize()}, Slice: {slice_index}, Voxels: {voxel_list}")

    def print_segmentation(self):
        print("Voxels segmentados:")
        for segmentation in self.segmentation:
            view, slice_index, voxel_list = segmentation
            print(f"Vista: {view.capitalize()}, Slice: {slice_index}, Voxels: {voxel_list}")

    # def save_region_growing(self):
    #     img = self.data

    #     # Crear una máscara binaria para almacenar la segmentación
    #     img_th = np.zeros_like(img)

    #     # Iterar sobre las anotaciones para definir los seed_points
    #     seed_points = []
    #     for segmentation in self.segmentation:
    #         view, slice_index, voxel_list = segmentation
    #         if view == self.current_view and slice_index == self.get_current_slice():
    #             for voxel in voxel_list:
    #                 seed_points.append((voxel[1], voxel[0], slice_index))  # Cambio de coordenadas (x, y) a (y, x)

    #     # Valor de tolerancia
    #     tolerancia = 80

    #     # Cola para almacenar puntos por visitar
    #     for seed_point in seed_points:
    #         # Movimientos
    #         def movimiento(point, direction):
    #             x, y, z = point
    #             if direction == 'derecha':
    #                 return x + 1, y, z
    #             elif direction == 'izquierda':
    #                 return x - 1, y, z
    #             elif direction == 'arriba':
    #                 return x, y + 1, z
    #             elif direction == 'abajo':
    #                 return x, y - 1, z
    #             elif direction == 'adelante':
    #                 return x, y, z + 1
    #             elif direction == 'atras':
    #                 return x, y, z - 1

    #         # Función para realizar desplazamientos
    #         arr_intensity = []
    #         arr_intensity.append(img[seed_point])
    #         recorrido = [seed_point]  # Inicializar lista de puntos recorridos

    #         # Iterar hasta que la cola esté vacía o hasta alcanzar un número máximo de iteraciones
    #         max_iterations = 1000
    #         for _ in range(max_iterations):
    #             if not recorrido:
    #                 break
    #             point = recorrido.pop(0)
    #             mean_intensity = np.mean(arr_intensity)
    #             # Definir los vecinos
    #             vecinos = ['arriba', 'abajo', 'derecha', 'izquierda', 'atras', 'adelante']

    #             for vecino in vecinos:
    #                 new_point = movimiento(point, vecino)
    #                 if new_point not in recorrido:
    #                     if all(0 <= coord < limit for coord, limit in zip(new_point, img.shape)) and abs(img[new_point] - mean_intensity) <= tolerancia:
    #                         img_th[new_point] = 1
    #                         arr_intensity.append(img[new_point])
    #                         recorrido.append(new_point)

    #     file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
    #     if file_path:
    #         # Crear un objeto Nibabel con los datos de la segmentación
    #         img_nifti = nib.Nifti1Image(img_th.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
    #         nib.save(img_nifti, file_path)
    #         print(f"Segmentación guardada como '{file_path}'")

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        global nifti_viewer
        nifti_viewer = NiftiViewer(root, file_path)


# Crear la ventana principal
root = tk.Tk()
root.title("Seleccionar imagen Nifti")
root.geometry("800x800")

# Crear el botón para seleccionar el archivo
select_button = ttk.Button(root, text="Seleccionar imagen", command=select_file)
select_button.grid(row=2, column=0, padx=10, pady=10)

# Ejecutar el bucle principal de la ventana
root.mainloop()
