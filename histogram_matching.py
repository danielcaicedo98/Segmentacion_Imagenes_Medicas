import numpy as np
import numpy as np
import nibabel as nib
import scipy.stats

# Cargar la imagen
img = nib.load('imagen.nii')
img_data = img.get_fdata()



def hist_match(train, test, k):

    pass

def trainning(train,x):
    x = np.linspace(5,95,x)
    y = np.percentile(train.flatten(),x)
    functions = []
    for i in range(1, len(x)):
        m = (y[i] - y[i-1]) / (x[i] - x[i-1])
        b = y[i-1] - m*x[i-1]
        f = lambda x : m*x + b
        functions.append(f) 
    # m = (y[1] - y[0]) / (x[1] - x[0])
    # b = y[0] - m*x[0]
    # f = lambda x : m*x + b

    return functions

print(trainning(img_data,5))



def testing(landmarks, data,functions):
    hm_img = np.zeros(data.shape)    
    percentil = scipy.stats.percentileofscore(sorted(list(set(data.flatten())) , reverse=False),data)
    hm_img[percentil > landmarks[0] and percentil < landmarks[1]] = functions[0](percentil > landmarks[0] and percentil < landmarks[1])
    
    return hm_img





