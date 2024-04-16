import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from tkinter import filedialog
import tkinter as tk
from matplotlib.path import Path
from k_means import k_means
from tkinter import ttk
from vedo import load, volume, show
from mean_filter import Mean_Filter
from median_filter import Median_Filter
from white_stripe import White_Stripe
from intensity_rescaling import Intensity_Rescaling
from z_score import Z_Score

class NiftiViewer:
    # global toggle_button

    
    def __init__(self, nifti_file):
        
        self.img = nib.load(nifti_file)
        self.data = self.img.get_fdata()
        self.shape = self.data.shape
        self.current_index_axial = self.shape[2] // 2
        self.current_index_sagittal = self.shape[0] // 2
        self.current_index_coronal = self.shape[1] // 2

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.35)

        self.slider_axial = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider_axial = Slider(self.slider_axial, 'Axial', 0, self.shape[2]-1, valinit=self.current_index_axial, valstep=1)
        self.slider_axial.on_changed(self.update_axial)

        self.slider_sagittal = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider_sagittal = Slider(self.slider_sagittal, 'Sagittal', 0, self.shape[0]-1, valinit=self.current_index_sagittal, valstep=1)
        self.slider_sagittal.on_changed(self.update_sagittal)

        self.slider_coronal = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.slider_coronal = Slider(self.slider_coronal, 'Coronal', 0, self.shape[1]-1, valinit=self.current_index_coronal, valstep=1)
        self.slider_coronal.on_changed(self.update_coronal)

        self.ax.imshow(self.data[:, :, self.current_index_axial], cmap='gray')
        self.traces = []
        self.traces_enabled = False  # Variable para activar/desactivar trazados

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=1, padx=1, pady=1)

        # Botón para activar/desactivar trazados
        style = ttk.Style()        
        style.configure('Custom.TButton', foreground="black", font=('Arial', 10, 'bold'), background="green")
        # Se configura también el fondo del botón cuando está en estado normal y en estado activo (pressed)
        style.map('Custom.TButton', background=[('active', 'blue'), ('pressed', '!disabled', 'blue')])
        self.toggle_button = ttk.Button(root, text="Activar Trazado", command=self.toggle_traces,style='Custom.TButton', width=20)
        self.toggle_button.grid(row=2, column=0, padx=10, pady=10)

        self.ax.set_title("Axial")
        self.ax.set_aspect('equal')
        self.ax.set_autoscale_on(False)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # Botón para guardar segmentación
        self.save_button = ttk.Button(root, text="Guardar trazado", command=self.save_segmentation,style='Custom.TButton', width=30)
        self.save_button.grid(row=1, column=1, padx=10, pady=10)

        # Botón para guardar isodata
        self.isodata_button = ttk.Button(root, text="Guardar Isodata", command=self.save_segmentation_isodata,style='Custom.TButton', width=30)
        self.isodata_button.grid(row=2, column=1, padx=10, pady=10)
        

        self.save_button = ttk.Button(root, text="Guardar crecimiento de regiones", command=self.save_region_growing,style='Custom.TButton', width=30)
        self.save_button.grid(row=3, column=1, padx=10, pady=10)

        self.save_button = ttk.Button(root, text="Guardar K-Means", command=lambda:k_means(self.img),style='Custom.TButton', width=30)
        self.save_button.grid(row=4, column=1, padx=10, pady=10)

        self.save_button = ttk.Button(root, text="Guardar 3D", command=self.save_segmentation_3d,style='Custom.TButton', width=20)
        self.save_button.grid(row=3, column=0, padx=10, pady=10)

        self.mean_filter = ttk.Button(root, text="Mean Filter", command=lambda:Mean_Filter(self.img),style='Custom.TButton', width=20)
        self.mean_filter.grid(row=0, column=0, padx=1, pady=1)

        self.median_filter = ttk.Button(root, text="Median Filter", command=lambda:Median_Filter(self.img),style='Custom.TButton', width=20)
        self.median_filter.grid(row=0, column=2, padx=0, pady=0)

        self.ws_button = ttk.Button(root, text="White Stripe", command=lambda:White_Stripe(self.img),style='Custom.TButton', width=20)
        self.ws_button.grid(row=1, column=3, padx=0, pady=0)

        self.zs_button = ttk.Button(root, text="Z Score", command=lambda:Z_Score(self.img),style='Custom.TButton', width=20)
        self.zs_button.grid(row=2, column=3, padx=0, pady=0)

        self.ir_button = ttk.Button(root, text="Intensity Rescaling", command=lambda:Intensity_Rescaling(self.img),style='Custom.TButton', width=20)
        self.ir_button.grid(row=3, column=3, padx=0, pady=0)

        

    def update_axial(self, index):
        self.current_index_axial = int(index)
        self.ax.clear()
        self.ax.imshow(self.data[:, :, self.current_index_axial], cmap='gray')
        self.draw_traces()
        self.fig.canvas.draw_idle()

    def update_sagittal(self, index):
        self.current_index_sagittal = int(index)
        self.show_sagittal()

    def update_coronal(self, index):
        self.current_index_coronal = int(index)
        self.show_coronal()

    def show_sagittal(self):
        self.ax.clear()
        self.ax.imshow(self.data[self.current_index_sagittal, :, :], cmap='gray')
        self.draw_traces()
        self.fig.canvas.draw_idle()

    def show_coronal(self):
        self.ax.clear()
        self.ax.imshow(self.data[:, self.current_index_coronal, :], cmap='gray')
        self.draw_traces()
        self.fig.canvas.draw_idle()

    def show_axial(self):
        self.ax.clear()
        self.ax.imshow(self.data[:, :, self.current_index_axial], cmap='gray')
        self.draw_traces()
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.button == 1 and self.traces_enabled:
            self.traces.append([event.xdata, event.ydata])

    def on_motion(self, event):
        if event.button == 1 and self.traces_enabled and self.traces:
            self.traces[-1].extend([event.xdata, event.ydata])
            self.draw_traces()
            self.fig.canvas.draw()

    def on_release(self, event):
        if event.button == 1 and self.traces_enabled and self.traces:
            self.traces[-1].extend([event.xdata, event.ydata])
            self.draw_traces()
            self.fig.canvas.draw()

    def draw_traces(self):
        self.ax.plot([], [])  # Limpiar trazados existentes
        if self.traces_enabled:
            for trace in self.traces:
                xs = trace[::2]
                ys = trace[1::2]
                self.ax.plot(xs, ys, 'r-', linewidth=5)# Ajustar el grosor del trazo
            self.fig.canvas.draw()

    def toggle_traces(self):
        
        if self.toggle_button.cget("text") == "Activar Trazado":
            self.toggle_button.config(text="Desactivar Trazado")  
        else:
            self.toggle_button.config(text="Activar Trazado") 

        self.traces_enabled = not self.traces_enabled
        # if not self.traces_enabled:
        #     self.traces = []  # Limpiar trazados al desactivar

    def update_thickness(self, value):
        self.trace_thickness = int(value)
        self.draw_traces()

    def save_segmentation(self):
        if self.traces:
            # Convertir trazos a una máscara binaria
            trace_mask = np.zeros(self.data.shape[:2], dtype=bool)
            for trace in self.traces:
                xs = trace[::2]
                ys = trace[1::2]
                path = Path(np.column_stack((xs, ys)))
                x, y = np.meshgrid(np.arange(self.data.shape[1]), np.arange(self.data.shape[0]))
                points = np.column_stack((x.ravel(), y.ravel()))
                mask = path.contains_points(points)
                trace_mask |= mask.reshape(self.data.shape[:2])

            # Aplicar la máscara a los datos originales
            trace_data = np.where(trace_mask[..., np.newaxis], self.data, 0)

            # Guardar la segmentación como un archivo NIFTI
            trace_img = nib.Nifti1Image(trace_data, affine=self.img.affine)
            file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
            if file_path:
                nib.save(trace_img, file_path)
                print(f"Segmentación guardada como '{file_path}'")
        else:
            print("No hay trazos para guardar.")

    def save_segmentation_isodata(self):
        # Algoritmo Isodata
        img = self.data
        delta = 0.1
        tau_init = 100
        t = 0
        tau_t = tau_init
        while True:
            img_th = img > tau_t
            m_foreground = img[img_th == 1].mean()
            m_background = img[img_th == 0].mean()
            tau_new = 0.5 * (m_background + m_foreground)

            if abs(tau_new - tau_t) < delta:
                break
            tau_t = tau_new

        # Guardado del archivo
        file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
        if file_path:
            # Crear un objeto Nibabel con los datos de la segmentación
            img_nifti = nib.Nifti1Image(img_th.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
            nib.save(img_nifti, file_path)
            print(f"Segmentación guardada como '{file_path}'")

    def save_segmentation_3d(self):
        # Algoritmo Isodata
        
        path_sl = "./k_means.nii"
        mesh = load(path_sl)

        show(mesh)

        # Guardado del archivo
        # file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
        # if file_path:
        #     # Crear un objeto Nibabel con los datos de la segmentación
        #     img_nifti = nib.Nifti1Image(img_th.astype(np.uint8), np.eye(4))  # Utilizamos np.eye(4) para la matriz de transformación (espacio físico)
        #     nib.save(img_nifti, file_path)
        #     print(f"Segmentación guardada como '{file_path}'")        

    def save_region_growing(self):
        img = self.data

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
        max_iterations = 100000
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


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("NIFTI files", "*.nii")])
    if file_path:
        nifti_viewer = NiftiViewer(file_path)
        nifti_viewer.canvas_widget.grid(row=0, column=1, padx=10, pady=10)

# Crear la ventana principal
root = tk.Tk()
root.title("Seleccionar imagen Nifti")
root.geometry("1200x720")
style = ttk.Style()        
style.configure('Custom.TButton', foreground="black", font=('Arial', 10, 'bold'), background="green")
        # Se configura también el fondo del botón cuando está en estado normal y en estado activo (pressed)
style.map('Custom.TButton', background=[('active', 'blue'), ('pressed', '!disabled', 'blue')])
# Crear el botón para seleccionar el archivo
select_button = ttk.Button(root,style='Custom.TButton',text="Seleccionar imagen", command=select_file, width=20)
select_button.grid(row=1, column=0, padx=10, pady=10)

# Ejecutar el bucle principal de la ventana
root.mainloop()