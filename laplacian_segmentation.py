import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import math
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class ImageCanvas(tk.Canvas):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.image_path = image_path
        self.original_image = Image.open(image_path)
        self.image = ImageOps.grayscale(self.original_image)
        self.photo = ImageTk.PhotoImage(self.image)
        
        self.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.config(width=self.photo.width(), height=self.photo.height())
        
        self.bind("<B1-Motion>", self.paint_red)
        self.bind("<B3-Motion>", self.paint_blue)
        self.drawn_pixels = {'red': [], 'blue': []}

    def paint_red(self, event):
        x, y = event.x, event.y
        self.create_oval(x-2, y-2, x+2, y+2, outline='red', fill='red', width=2)
        self.drawn_pixels['red'].append((x, y))
        print(f"Drew at: {x}, {y} in red")

    def paint_blue(self, event):
        x, y = event.x, event.y
        self.create_oval(x-2, y-2, x+2, y+2, outline='blue', fill='blue', width=2)
        self.drawn_pixels['blue'].append((x, y))
        print(f"Drew at: {x}, {y} in blue")

    def print_positions_and_matrix(self):
        laplacian(self.get_intensity_matrix(),self.drawn_pixels['red'],self.drawn_pixels['blue'])

    def print_intensity_matrix(self):
        print("Intensity Matrix:")

    def get_intensity_matrix(self):
        original_image = Image.open(self.image_path)
        original_image = ImageOps.grayscale(original_image)
        return [[original_image.getpixel((x, y)) for x in range(original_image.width)] for y in range(original_image.height)]
    

def laplacian(img,F,B):
    img_data = np.array(img)
    
    height, width = img_data.shape

    V = [(d, h) for d in range(height) for h in range(width)]
    E = []
    W_e = []
    N = []
    D = []
    # Crear una matriz para la nueva imagen con la misma forma que la imagen original

    shape = img_data.shape

    def calculate_sigma(image):
        height, width = image.shape
        max_difference = 0

        # Recorremos la imagen
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Vecindad de 8 píxeles
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j-1], image[i, j+1],
                    image[i+1, j-1], image[i+1, j], image[i+1, j+1]
                ]
                # Calculamos la diferencia máxima
                diff = np.max(neighbors) - np.min(neighbors)
                if diff > max_difference:
                    max_difference = diff

        return max_difference

    def calculate_wij(edges,sigma):
        w = []
        β = 1
        for i in edges:
            diff =  np.abs(i[0] - i[1])
            max_diff = β * diff**2
            div = -1 * (max_diff / sigma)
            exp = math.exp(div)
            w.append(exp)
        # print(w)
        return w

    for i in range(height):
        for j in range(width):
            indice_actual = i * width + j

            # Obtener los índices de los 8 vecinos
            nbh = [
                max(i-1, 0) * width + max(j-1, 0), max(i-1, 0) * width + j,
                max(i-1, 0) * width + min(j+1, width-1), i * width + max(j-1, 0),
                i * width + min(j+1, width-1), min(i+1, height-1) * width + max(j-1, 0),
                min(i+1, height-1) * width + j, min(i+1, height-1) * width + min(j+1, width-1)
            ]

            # Crear las aristas
            nb = [(indice_actual, nb) for nb in nbh]
            E.append(nb)


    sigma = (10**-6) + calculate_sigma(img_data)

    for i in range(height):
        for j in range(width):
            # Obtener el valor del píxel actual
            voxel_actual = img_data[i, j]

            # Obtener los valores de los 8 vecinos
            vecinos = [
                img_data[max(i-1, 0), max(j-1, 0)], img_data[max(i-1, 0), j],
                img_data[max(i-1, 0), min(j+1, width-1)], img_data[i, max(j-1, 0)],
                img_data[i, min(j+1, width-1)], img_data[min(i+1, height-1), max(j-1, 0)],
                img_data[min(i+1, height-1), j], img_data[min(i+1, height-1), min(j+1, width-1)]
            ]

            # Crear las aristas y los índices de los vecinos
            e = [
                (voxel_actual, vecinos[0]), (voxel_actual, vecinos[1]),
                (voxel_actual, vecinos[2]), (voxel_actual, vecinos[3]),
                (voxel_actual, vecinos[4]), (voxel_actual, vecinos[5]),
                (voxel_actual, vecinos[6]), (voxel_actual, vecinos[7])
            ]

            n = [
                V.index((max(i-1, 0), max(j-1, 0))), V.index((max(i-1, 0), j)),
                V.index((max(i-1, 0), min(j+1, width-1))), V.index((i, max(j-1, 0))),
                V.index((i, min(j+1, width-1))), V.index((min(i+1, height-1), max(j-1, 0))),
                V.index((min(i+1, height-1), j)), V.index((min(i+1, height-1), min(j+1, width-1)))
            ]

            N.append(n)
            w = calculate_wij(e,sigma)
            d = sum(w)
            D.append(d)
            W_e.append(w)

    D_m = coo_matrix((height * width, height * width), dtype=np.float64)

    # Lista para almacenar los valores no cero, filas y columnas
    data = []
    rows = []
    cols = []

    # Llenar la matriz dispersa COO
    for i in range(height * width):
        for j in range(height * width):
            if i == j:
                index = (i, j)            
                data.append(D[i])
                rows.append(i)
                cols.append(j)
                

    # Crear la matriz dispersa COO
    D_m = diags(D, 0, format='csr')

    W = coo_matrix((height * width, height * width), dtype=np.float64)


    # Lista para almacenar los valores no cero, filas y columnas
    data = []
    rows = []
    cols = []

    # Llenar la matriz dispersa COO
    for i in range(height * width):
        for j in range(height * width):
            index = (i, j)
            for e in E:
                if index in e:
                    data.append(W_e[i][e.index(index)])
                    rows.append(i)
                    cols.append(j)

    # Crear la matriz dispersa COO
    W = coo_matrix((data, (rows, cols)), shape=(height * width, height * width), dtype=np.float64)

    xB = 1
    xF = 0

    L = D_m - W

    b = np.zeros((height * width, 1))

    for pixel in V:
        # print(pixel)
        if pixel in B:
            b[V.index(pixel)] = xB
        elif pixel in F:
            b[V.index(pixel)] = xF
        
    S = B + F  # Concatenar las listas B y F

    I_s = coo_matrix((height * width, height * width), dtype=np.float64)


    # Lista para almacenar los valores no cero, filas y columnas
    data = []
    rows = []
    cols = []

    # Llenar la matriz dispersa COO
    for i, elem in enumerate(S):
        data.append(1)
        rows.append(i)
        cols.append(i)
                

    # Crear la matriz dispersa COO
    I_s = coo_matrix((data, (rows, cols)), shape=(height * width, height * width), dtype=np.float64)

    M = I_s + np.dot(L,L)
    x = spsolve(M, b)

    y = np.zeros_like(x)
    t = (xB + xF) / 2
    for i in range(len(x)):    
        if x[i] >= t:
            y[i] = xB
        else:
            y[i] = xF
    nueva_img_data = np.zeros_like(img_data)
    nueva_img_data = y.reshape(img_data.shape)

    # print(nueva_img_data)

    plt.imshow(nueva_img_data, cmap='gray')
    plt.axis('off')
    plt.show()


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        app.canvas = ImageCanvas(app, file_path)
        app.canvas.pack()
        app.print_button.config(state=tk.NORMAL)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Drawer")
        self.canvas = None
        
        self.open_button = tk.Button(self, text="Open Image", command=open_image)
        self.open_button.pack()

        self.print_button = tk.Button(self, text="Segmentation", command=self.print_positions_and_matrix, state=tk.DISABLED)
        self.print_button.pack()

    def print_positions_and_matrix(self):
        self.canvas.print_positions_and_matrix()

if __name__ == "__main__":
    app = App()
    app.mainloop()
