from vedo import load, show
from tkinter import filedialog

def save_segmentation_3d():
        
    # Algoritmo Isodata
    filename = filedialog.askopenfilename(filetypes=[("NIFTI files", "*.nii")])
    path_sl = filename
    mesh = load(path_sl)
    show(mesh)