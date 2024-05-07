import numpy as np
import math
from scipy.optimize import minimize
import nibabel as nib
from tkinter import filedialog

img = nib.load('imagen.nii')
img_data= np.array([[[10, 10, 10, 10],
                          [10, 20, 20, 10],
                          [10, 20, 20, 10],
                          [10, 10, 10, 10]],
                        
                         [[335, 5, 5, 5],
                          [5, 15, 15, 5],
                          [5, 15, 15, 5],
                          [5, 5, 5, 5]],

                         [[10, 10, 10, 10],
                          [10, 20, 20, 10],
                          [10, 20, 20, 10],
                          [10, 10, 10, 10]],

                        [[5, 5, 5, 5],
                          [5, 15, 15, 5],
                          [5, 15, 15, 5],
                          [5, 5, 5, 5]],
                          ])

# B = [(88, 4, 32), (88, 5, 32), (88, 5, 32), (88, 6, 32), (88, 8, 32), (88, 11, 31), (88, 15, 31), (88, 21, 30), (88, 28, 28), (88, 32, 27), (88, 37, 26), (88, 41, 25), (88, 44, 25), (88, 50, 25), (88, 54, 25), (88, 58, 25), (88, 64, 25), (88, 69, 24), (88, 73, 23), (88, 78, 23), (88, 80, 22), (88, 84, 22), (88, 88, 21), (88, 91, 21), (88, 95, 20), (88, 100, 20), (88, 104, 20), (88, 109, 20), (88, 111, 20), (88, 114, 20), (88, 116, 20), (88, 119, 20), (88, 125, 21), (88, 129, 22), (88, 132, 23), (88, 136, 23), (88, 137, 24), (88, 141, 25), (88, 144, 26), (88, 145, 26), (88, 150, 28), (88, 153, 28), (88, 156, 30), (88, 161, 31), (88, 164, 33), (88, 167, 34), (88, 172, 38), (88, 175, 40), (88, 179, 43), (88, 181, 45), (88, 183, 47), (88, 185, 51), (88, 187, 52), (88, 187, 54), (88, 189, 56), (88, 189, 57), (88, 190, 59), (88, 191, 61), (88, 191, 116), (88, 190, 118), (88, 189, 119), (88, 187, 120), (88, 186, 121), (88, 185, 122), (88, 183, 123), (88, 180, 125), (88, 179, 126), (88, 177, 127), (88, 177, 129), (88, 175, 131), (88, 174, 132), (88, 173, 133), (88, 172, 134), (88, 171, 136), (88, 170, 136), (88, 170, 137), (88, 169, 139), (88, 168, 139), (88, 167, 141), (88, 165, 142), (88, 164, 143), (88, 162, 144), (88, 161, 144), (88, 159, 145), (88, 158, 146), (88, 156, 147), (88, 155, 149), (88, 152, 150), (88, 150, 150), (88, 147, 152), (88, 144, 152), (88, 140, 154), (88, 136, 155), (88, 130, 157), (88, 127, 158), (88, 121, 159), (88, 117, 159), (88, 111, 160), (88, 107, 160), (88, 101, 162), (88, 96, 162), (88, 93, 162), (88, 88, 162), (88, 85, 162), (88, 81, 163), (88, 77, 163), (88, 74, 163), (88, 70, 163), (88, 66, 163), (88, 63, 163), (88, 60, 163), (88, 56, 163), (88, 51, 163), (88, 46, 163), (88, 43, 163), (88, 39, 163), (88, 36, 163), (88, 31, 163), (88, 29, 163), (88, 25, 163), (88, 23, 162), (88, 20, 162), (88, 18, 161), (88, 17, 161), (88, 15, 160), (88, 13, 160), (88, 11, 160), (88, 10, 159), (88, 10, 159), (88, 9, 159), (88, 8, 159), (88, 8, 159), (88, 7, 159), (88, 6, 159), (88, 6, 158), (88, 5, 158), (88, 5, 158), (88, 4, 158), (88, 3, 158), (88, 3, 158), (88, 2, 158), (88, 3, 158), (88, 3, 158), (88, 4, 158), (88, 6, 158), (88, 8, 158), (88, 11, 158), (88, 13, 158), (88, 16, 158), (88, 19, 158), (88, 21, 158), (88, 23, 158), (88, 23, 158), (88, 25, 158), (88, 26, 158), (88, 27, 158), (88, 30, 158), (88, 33, 158), (88, 35, 159), (88, 39, 159), (88, 42, 159), (88, 45, 159), (88, 47, 159), (88, 50, 159), (88, 53, 159), (88, 57, 159), (88, 62, 159), (88, 66, 159), (88, 69, 159), (88, 74, 159), (88, 89, 159), (88, 94, 159), (88, 100, 158), (88, 106, 157), (88, 111, 156), (88, 115, 154), (88, 120, 152), (88, 125, 150), (88, 128, 149), (88, 131, 147), (88, 135, 145), (88, 138, 144), (88, 141, 142), (88, 143, 141), (88, 145, 139), (88, 149, 137), (88, 151, 135), (88, 154, 133), (88, 155, 131), (88, 157, 128), (88, 158, 127), (88, 159, 125), (88, 160, 124), (88, 162, 122), (88, 162, 121), (88, 164, 119), (88, 165, 116), (88, 165, 114), (88, 167, 112), (88, 167, 111), (88, 168, 109), (88, 169, 107), (88, 170, 104), (88, 170, 102), (88, 171, 99), (88, 172, 95), (88, 173, 93), (88, 173, 89), (88, 174, 87), (88, 175, 84), (88, 175, 81), (88, 175, 79), (88, 175, 77), (88, 175, 75), (88, 175, 73), (88, 175, 71), (88, 175, 69), (88, 174, 67), (88, 174, 65), (88, 172, 63), (88, 170, 60), (88, 169, 58), (88, 167, 56), (88, 165, 55), (88, 164, 53), (88, 162, 51), (88, 160, 50), (88, 157, 48), (88, 155, 47), (88, 152, 45), (88, 150, 43), (88, 146, 41), (88, 143, 40), (88, 137, 38), (88, 134, 36), (88, 132, 36), (88, 129, 35), (88, 125, 34), (88, 122, 34), (88, 120, 34), (88, 116, 34), (88, 111, 34), (88, 105, 34), (88, 98, 34), (88, 91, 34), (88, 88, 34), (88, 82, 34), (88, 76, 34), (88, 72, 34), (88, 68, 34), (88, 66, 34), (88, 62, 34), (88, 59, 34), (88, 56, 35), (88, 54, 36), (88, 51, 36), (88, 47, 37), (88, 43, 38), (88, 40, 39), (88, 38, 39), (88, 36, 39), (88, 34, 40), (88, 32, 40), (88, 30, 40), (88, 28, 41), (88, 26, 41), (88, 25, 41), (88, 24, 41), (88, 21, 41), (88, 20, 43), (88, 18, 43), (88, 16, 43), (88, 14, 43), (88, 12, 45), (88, 11, 45), (88, 10, 45), (88, 10, 45), (88, 9, 45), (88, 8, 45), (88, 5, 46), (88, 4, 46), (88, 16, 26), (88, 17, 26), (88, 17, 25), (88, 18, 21), (88, 20, 16), (88, 20, 13), (88, 21, 13), (88, 21, 13), (88, 21, 16), (88, 18, 26), (88, 18, 31), (88, 17, 31), (88, 18, 31), (88, 20, 28), (88, 25, 20), (88, 28, 16), (88, 28, 15), (88, 28, 16), (88, 26, 21), (88, 22, 34), (88, 20, 42), (88, 19, 44), (88, 20, 43), (88, 21, 40), (88, 29, 27), (88, 35, 19), (88, 36, 16), (88, 37, 16), (88, 37, 19), (88, 37, 24), (88, 36, 27), (88, 36, 28), (88, 37, 27), (88, 42, 18), (88, 46, 11), (88, 48, 10), (88, 48, 11), (88, 48, 16), (88, 47, 25), (88, 46, 28), (88, 46, 29), (88, 48, 28), (88, 53, 16), (88, 58, 8), (88, 59, 7), (88, 59, 9), (88, 59, 18), (88, 59, 23), (88, 59, 26), (88, 59, 26), (88, 60, 25), (88, 65, 18), (88, 69, 12), (88, 70, 11), (88, 70, 14), (88, 70, 21), (88, 69, 29), (88, 69, 31), (88, 71, 31), (88, 75, 26), (88, 86, 12), (88, 91, 9), (88, 91, 10), (88, 92, 16), (88, 92, 25), (88, 92, 30), (88, 93, 31), (88, 94, 30), (88, 101, 20), (88, 111, 9), (88, 112, 8), (88, 113, 10), (88, 113, 14), (88, 114, 17), (88, 114, 18), (88, 116, 16), (88, 124, 11), (88, 125, 10), (88, 125, 13), (88, 125, 19), (88, 125, 22), (88, 125, 24), (88, 126, 23), (88, 129, 21), (88, 134, 18), (88, 136, 17), (88, 136, 18), (88, 136, 23), (88, 136, 28), (88, 135, 30), (88, 135, 31), (88, 137, 30), (88, 143, 26), (88, 153, 22), (88, 159, 21), (88, 159, 21), (88, 159, 23), (88, 157, 26), (88, 157, 28), (88, 157, 28), (88, 160, 27), (88, 170, 23), (88, 175, 21), (88, 176, 21), (88, 176, 21), (88, 174, 25), (88, 170, 29), (88, 169, 33), (88, 168, 33), (88, 169, 33), (88, 173, 28), (88, 181, 20), (88, 182, 20), (88, 182, 20), (88, 181, 22), (88, 179, 26), (88, 175, 30), (88, 172, 33), (88, 172, 33), (88, 172, 33), (88, 172, 32), (88, 178, 122), (88, 179, 124), (88, 180, 132), (88, 181, 141), (88, 181, 146), (88, 181, 148), (88, 181, 147), (88, 180, 144), (88, 175, 135), (88, 174, 131), (88, 173, 131), (88, 171, 136), (88, 168, 152), (88, 165, 167), (88, 165, 168), (88, 165, 167), (88, 165, 162), (88, 162, 146), (88, 159, 135), (88, 159, 135), (88, 157, 141), (88, 154, 157), (88, 152, 169), (88, 152, 172), (88, 152, 171), (88, 152, 165), (88, 152, 154), (88, 150, 150), (88, 150, 150), (88, 147, 159), (88, 144, 169), (88, 143, 172), (88, 143, 172), (88, 142, 165), (88, 142, 160), (88, 141, 160), (88, 141, 162), (88, 138, 172), (88, 136, 175), (88, 136, 177), (88, 136, 175), (88, 134, 168), (88, 133, 162), (88, 132, 161), (88, 131, 162), (88, 129, 167), (88, 124, 172), (88, 123, 173), (88, 123, 172), (88, 121, 164), (88, 121, 162), (88, 120, 162), (88, 118, 166), (88, 110, 178), (88, 106, 182), (88, 106, 182), (88, 103, 179), (88, 100, 172), (88, 97, 169), (88, 97, 168), (88, 95, 170), (88, 93, 175), (88, 91, 177), (88, 91, 176), (88, 91, 173), (88, 90, 170), (88, 89, 170), (88, 87, 175), (88, 83, 180), (88, 82, 181), (88, 81, 181), (88, 81, 177), (88, 79, 169), (88, 78, 168), (88, 77, 168), (88, 76, 170), (88, 71, 177), (88, 69, 180), (88, 69, 180), (88, 69, 178), (88, 68, 172), (88, 66, 167), (88, 66, 167), (88, 64, 169), (88, 58, 179), (88, 56, 182), (88, 54, 182), (88, 54, 178), (88, 53, 170), (88, 52, 164), (88, 51, 164), (88, 50, 165), (88, 44, 172), (88, 41, 177), (88, 40, 177), (88, 40, 175), (88, 38, 168), (88, 36, 163), (88, 35, 164), (88, 31, 170), (88, 30, 174), (88, 29, 174), (88, 29, 173), (88, 28, 169), (88, 26, 166), (88, 25, 166), (88, 25, 168), (88, 21, 172), (88, 21, 172), (88, 20, 172), (88, 20, 172), (88, 19, 172), (88, 15, 177), (88, 13, 181), (88, 13, 182), (88, 12, 180), (88, 12, 178), (88, 11, 176), (88, 11, 177), (88, 10, 180), (88, 9, 185), (88, 8, 187), (88, 8, 187), (88, 8, 184), (88, 8, 181), (88, 8, 180), (88, 8, 181), (88, 9, 181), (88, 10, 181)]
# F = [(88, 11, 65), (88, 12, 65), (88, 13, 64), (88, 15, 64), (88, 18, 63), (88, 20, 63), (88, 21, 62), (88, 25, 61), (88, 27, 61), (88, 29, 60), (88, 33, 59), (88, 36, 59), (88, 40, 59), (88, 42, 58), (88, 46, 58), (88, 48, 57), (88, 51, 56), (88, 55, 56), (88, 58, 56), (88, 62, 56), (88, 63, 56), (88, 65, 56), (88, 68, 56), (88, 70, 56), (88, 72, 56), (88, 74, 56), (88, 76, 56), (88, 78, 56), (88, 80, 56), (88, 83, 57), (88, 84, 57), (88, 86, 58), (88, 88, 59), (88, 89, 59), (88, 91, 59), (88, 92, 60), (88, 93, 61), (88, 93, 61), (88, 94, 61), (88, 95, 62), (88, 96, 63), (88, 97, 63), (88, 97, 63), (88, 98, 63), (88, 99, 64), (88, 99, 64), (88, 101, 65), (88, 101, 66), (88, 102, 66), (88, 104, 66), (88, 105, 67), (88, 107, 68), (88, 107, 68), (88, 109, 69), (88, 109, 69), (88, 111, 70), (88, 112, 71), (88, 113, 72), (88, 114, 73), (88, 116, 74), (88, 116, 74), (88, 117, 76), (88, 119, 78), (88, 120, 79), (88, 120, 80), (88, 121, 81), (88, 121, 82), (88, 122, 83), (88, 122, 84), (88, 122, 84), (88, 124, 86), (88, 124, 87), (88, 124, 88), (88, 125, 89), (88, 125, 89), (88, 126, 91), (88, 126, 91), (88, 126, 93), (88, 126, 94), (88, 126, 96), (88, 126, 97), (88, 124, 99), (88, 124, 100), (88, 122, 102), (88, 121, 103), (88, 121, 104), (88, 119, 106), (88, 118, 106), (88, 116, 107), (88, 113, 109), (88, 111, 110), (88, 108, 111), (88, 106, 112), (88, 105, 113), (88, 103, 114), (88, 101, 114), (88, 99, 116), (88, 97, 116), (88, 94, 117), (88, 92, 117), (88, 89, 117), (88, 86, 118), (88, 83, 119), (88, 81, 119), (88, 78, 119), (88, 75, 119), (88, 74, 119), (88, 71, 119), (88, 70, 119), (88, 69, 119), (88, 68, 119), (88, 66, 119), (88, 64, 119), (88, 63, 119), (88, 61, 119), (88, 59, 119), (88, 57, 119), (88, 55, 119), (88, 53, 119), (88, 50, 119), (88, 48, 118), (88, 46, 117), (88, 44, 117), (88, 42, 117), (88, 41, 116), (88, 39, 116), (88, 38, 116), (88, 36, 116), (88, 36, 116), (88, 34, 115), (88, 33, 114), (88, 31, 114), (88, 30, 114), (88, 30, 114), (88, 30, 113), (88, 29, 113), (88, 28, 112), (88, 26, 111), (88, 26, 111), (88, 25, 110), (88, 25, 109), (88, 25, 109), (88, 25, 107), (88, 25, 107), (88, 25, 106), (88, 24, 105), (88, 24, 104), (88, 24, 102), (88, 24, 102), (88, 24, 101), (88, 24, 99), (88, 24, 97), (88, 23, 96), (88, 23, 95), (88, 23, 93), (88, 23, 92), (88, 22, 90), (88, 22, 88), (88, 22, 86), (88, 22, 83), (88, 22, 81), (88, 21, 79), (88, 21, 78), (88, 21, 76), (88, 20, 74), (88, 20, 73), (88, 20, 71), (88, 20, 69), (88, 19, 68), (88, 19, 67), (88, 19, 66), (88, 19, 66), (88, 19, 65), (88, 19, 64), (88, 20, 64), (88, 20, 65), (88, 21, 69), (88, 22, 76), (88, 23, 81), (88, 24, 84), (88, 24, 85), (88, 24, 84), (88, 24, 83), (88, 24, 75), (88, 24, 69), (88, 25, 69), (88, 25, 72), (88, 28, 86), (88, 30, 101), (88, 31, 109), (88, 31, 111), (88, 32, 111), (88, 32, 108), (88, 33, 93), (88, 34, 81), (88, 35, 78), (88, 35, 78), (88, 35, 80), (88, 35, 91), (88, 35, 101), (88, 35, 103), (88, 35, 102), (88, 35, 96), (88, 36, 78), (88, 36, 66), (88, 37, 64), (88, 38, 67), (88, 39, 83), (88, 40, 95), (88, 40, 99), (88, 40, 98), (88, 42, 91), (88, 45, 75), (88, 46, 64), (88, 46, 63), (88, 47, 66), (88, 48, 81), (88, 49, 91), (88, 50, 99), (88, 51, 100), (88, 53, 96), (88, 55, 78), (88, 59, 59), (88, 60, 55), (88, 60, 58), (88, 60, 68), (88, 60, 86), (88, 61, 97), (88, 61, 98), (88, 61, 96), (88, 64, 83), (88, 68, 64), (88, 69, 59), (88, 69, 61), (88, 69, 77), (88, 69, 96), (88, 70, 103), (88, 71, 103), (88, 71, 101), (88, 73, 83), (88, 77, 63), (88, 78, 59), (88, 78, 63), (88, 78, 76), (88, 78, 92), (88, 78, 95), (88, 78, 95), (88, 79, 89), (88, 83, 69), (88, 85, 56), (88, 85, 55), (88, 85, 60), (88, 85, 78), (88, 85, 91), (88, 85, 94), (88, 86, 93), (88, 86, 86), (88, 86, 68), (88, 86, 61), (88, 86, 64), (88, 86, 79), (88, 86, 95), (88, 86, 100), (88, 87, 99), (88, 88, 93), (88, 91, 74), (88, 93, 66), (88, 93, 66), (88, 94, 76), (88, 94, 92), (88, 95, 98), (88, 96, 97), (88, 97, 84), (88, 99, 73), (88, 99, 73), (88, 99, 74), (88, 99, 89), (88, 99, 96), (88, 99, 95), (88, 100, 92), (88, 101, 81), (88, 102, 78), (88, 102, 79), (88, 102, 88), (88, 102, 97), (88, 103, 99), (88, 104, 98), (88, 105, 89), (88, 107, 78), (88, 107, 78), (88, 107, 83), (88, 107, 93), (88, 108, 98), (88, 109, 98), (88, 110, 95), (88, 110, 91), (88, 110, 91), (88, 110, 95), (88, 111, 101), (88, 112, 101), (88, 113, 96), (88, 116, 90), (88, 116, 89), (88, 116, 89), (88, 117, 93), (88, 117, 94), (88, 117, 93), (88, 119, 90), (88, 119, 85), (88, 120, 84), (88, 120, 86), (88, 120, 91), (88, 120, 93), (88, 120, 92), (88, 121, 91), (88, 121, 88), (88, 121, 87), (88, 122, 87)]



# img_data = img.get_fdata()
height, width, depth, = img_data.shape


V = [(d, h, w) for d in range(height) for h in range(width) for w in range(depth)]
E = []
W = []
N = []
D = []
# Crear una matriz para la nueva imagen con la misma forma que la imagen original
nueva_img_data = np.zeros_like(img_data)
shape = img_data.shape
# print(shape)
def calculate_sigma(edges):
    sigma = 0
    for i in edges:
        diff = np.abs(i[0] - i[1])
        max_diff = np.max(diff)
        if max_diff > sigma:
            sigma = max_diff
    return sigma

def calculate_wijk(edges,sigma):    
    w = []
    for i in edges:
        diff = np.abs(i[0] - i[1])      
        max_diff = np.max(diff)**2
        betha = 0.1 * max_diff
        div = -1 * (betha / sigma)
        exp = math.exp(div)  
        w.append(exp)
    # print(w)   
    return w
cont = 0
for i in range(height):    
    for j in range(width):
        for k in range(depth):
            # Obtener el valor del voxel actual     
            cont += 1       
            voxel_actual = img_data[i, j, k]
            # Obtener los valores de los 6 vecinos
            vecinos = [
                        img_data[min(i+1, height-1), j, k], img_data[max(i-1, 0), j, k],
                        img_data[i, min(j+1, width-1), k], img_data[i, max(j-1, 0), k],
                        img_data[i, j, min(k+1, depth-1)], img_data[i, j, max(k-1, 0)]
                    ]
            e = [
                (voxel_actual,vecinos[0]) , (voxel_actual,vecinos[1]),
                (voxel_actual,vecinos[2]) , (voxel_actual,vecinos[3]),
                (voxel_actual,vecinos[4]) , (voxel_actual,vecinos[5])
            ]

            n = [
                V.index((min(i+1, height-1), j, k)), V.index((max(i-1, 0), j, k)),
                V.index((i, min(j+1, width-1), k)), V.index((i, max(j-1, 0), k)),
                V.index((i, j, min(k+1, depth-1))), V.index((i, j, max(k-1, 0)))
            ]
            N.append(n)
            E.append(e)
            sigma = (10**-6) + calculate_sigma(e)
            w = calculate_wijk(e,sigma)
            d = sum(w)
            D.append(d)
            W.append(w)
# print(N)
# print(V)
# print(cont)
# print(len(W))
k1 = 1
k2 = 1
k3 = 1

xB = 1
xF = 0.1
F = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)]
B = [(2, 3, 3), (3, 0, 0), (3, 0, 1), (3, 0, 2), (3, 0, 3), (3, 1, 0), (3, 1, 1), (3, 1, 2),
    (3, 1, 3), (3, 2, 0), (3, 2, 1), (3, 2, 2)]


def loss_function(x):
    suma_b = 0
    suma_f = 0
    suma_v = 0
    Ex = 0
    for i in range(len(V)):   
        if  V[i] in B:
            suma_b += k1 * np.linalg.norm(x[i] - xB)**2 
        if  V[i] in F:    
            suma_f = k2 * np.linalg.norm(x[i] - xF)**2
        if V[i] in V:    
            suma_v = k3 * np.linalg.norm(x[i]*D[i] - sum(W[i][j]*x[j] for j in range(0,len(N[i]))) )**2    
            # suma_v = k3 * np.linalg.norm(x[i]*D[i] - sum(W[i][N[i][j]]*x[j] for j in range(0,len(N[i]))) )**2            
                    
    Ex += suma_b + suma_f + suma_v
    return Ex   

x_initial = np.zeros(len(V))

# Minimización de la función de pérdida
# result = minimize(loss_function, x_initial, method='CG')

result = minimize(loss_function, x_initial,method='BFGS', tol=1e-3)

# Los valores óptimos de x
x_optimal = result.x
# print(len(x_optimal))
# print("Valores óptimos de x que minimizan En:")
# print(x_optimal)
y = np.zeros(len(V))


for i in range(len(V)):
    if (x_optimal[i] > ((xB + xF) / 2)):
        nueva_img_data[V[i]] = xB
        y[i] = xB
    else:
        nueva_img_data[V[i]] = xF
        y[i] = xF    

# file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIFTI files", "*.nii")])
# if file_path:
#     # Crear un objeto Nibabel con los datos de la segmentación
#     img_nifti = nib.Nifti1Image(nueva_img_data, img.affine)
#     nib.save(img_nifti, file_path)
#     print(f"Segmentación guardada como '{file_path}'")
print(x_optimal)
print(nueva_img_data)
# print(N)