#%%
#author Lasse Totland @ Ullevål OUS
#analysis of MR combodatasets
#(streamlining: all of this can be implemented as a more user friendly 
#function/class eventually (methods for V and SR plot?))


#imports

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns
#import sklearn

import scipy.io as sio
import scipy.signal as sig

from scipy.ndimage import gaussian_filter, center_of_mass
import imageio

#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
dict = sio.loadmat(r'R:\Lasse\ComboData.mat')
data = dict["ComboData_thisonly"]

print(f'Keys in dictionary: {dict.keys()}') #dict_keys(['StudyData', 'StudyParam'])
print(f'Combodata shape: {np.shape(data)}')

#%%

#velocity V field, and magnitudes

# shape of V is (1, 1), indexing necessary to 'unpack' the correct format (?)

# last indexing is to zoom in; reducing amount of data to iterate through
# same minimum and maximum image position values
f_max = 80
f_min = 30

V = data['V'][0,0][f_min:f_max, f_min:f_max, :, :, :] #velocity field
M = data['Magn'][0,0][f_min:f_max, f_min:f_max, :, :] #magnitudes
mask = data['Mask'][0,0][f_min:f_max, f_min:f_max, :, :] #mask for non-heart tissue

T = len(V[0,0,0,:,0]) #Total amount of time steps

print(f'Velocity field shape: {np.shape(V)}')
print(f'Magnitudes field shape: {np.shape(M)}')
print(f'Mask shape: {np.shape(mask)}')

#%%
#generate images (implement line that clears gifdump?)

#plot every n'th vector
n = 2

#axis lengths
ax = len(mask[:,0,0,0])
ay = len(mask[0,:,0,0])
X, Y = np.meshgrid(np.arange(0, ax, 1), np.arange(0, ay, 1))

#find center of mask at t=0
cy, cx = center_of_mass(mask[:f, :f, 0, 0])

for t in range(T):
    #whats the third component?
    frame1 = M[:, :, 0, t] #photon density at time t
    mask_t = mask[:, :, 0, t]
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(frame1/np.max(frame1), cmap='gray', vmin = 0, vmax = 1)
    plt.colorbar()
    plt.title(f'Velocity plot over proton density at timepoint t = {t}', fontsize = 15)
    
    
    #2D vector visualization 128x128 res image in xy plane
    
    #wiener noise reduction filter (?)
    vx = gaussian_filter(V[:, :, 0, t, 1], sigma = 1)*mask_t #x components of velocity w mask
    vy = gaussian_filter(-V[:, :, 0, t, 0], sigma = 1)*mask_t #y components (negative?)
    
    q = ax.quiver(X[::n, ::n], Y[::n, ::n], vx[::n, ::n], vy[::n, ::n], 
                  color = 'w', scale = 100, minshaft = 1, minlength=0, width = 0.01)
    plt.savefig(f'R:\Lasse\plots\Vdump\V(t={t}).PNG')
    plt.show()
    
#%%
#generate GIF from image sequence (https://github.com/adenarayana/Python-Matplotlib/blob/main/009_gifAnimation.ipynb)

filenames = [f'R:\Lasse\plots\Vdump\V(t={t}).PNG' for t in range(T)]

with imageio.get_writer('R:\Lasse\plots\MP4\Velocity.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
    