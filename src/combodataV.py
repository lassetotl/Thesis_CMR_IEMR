#%%
#author Lasse Totland @ Ullevål OUS
#analysis of MR combodatasets
#(streamlining: all of this can be implemented as a more user friendly 
#function/class eventually (methods for V and SR plot?))


#imports

import numpy as np
import matplotlib.pyplot as plt
from lasse_functions import divergence
#import pandas as pd
#import seaborn as sns
#import sklearn

import scipy.io as sio
import scipy.signal as sig
import scipy.ndimage as ndi 

from scipy.ndimage import gaussian_filter
import imageio

#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
file = 'sham_D4-4_1d'
dict = sio.loadmat(f'R:\Lasse\combodata_shax\{file}.mat')
data = dict["ComboData_thisonly"]

print(f'Keys in dictionary: {dict.keys()}') #dict_keys(['StudyData', 'StudyParam'])
print(f'Combodata shape: {np.shape(data)}')

#%%

#velocity V field, and magnitudes

# shape of V is (1, 1), indexing necessary to 'unpack' the correct format (?)
V = data['V'][0,0] #velocity field
M = data['Magn'][0,0] #magnitudes
mask = data['Mask'][0,0] #mask for non-heart tissue

T = len(V[0,0,0,:,0]) #Total amount of time steps
T_es = data['TimePointEndSystole'][0,0][0][0]
T_ed = data['TimePointEndDiastole'][0,0][0][0]
res = data['Resolution'][0,0][0][0]  # voxel length in mm?

print(f'{file} overview:')
print(f'Velocity field shape: {np.shape(V)}')
print(f'Magnitudes field shape: {np.shape(M)}')
print(f'Mask shape: {np.shape(mask)}')

print(f'End systole at t={T_es}, end diastole at t={T_ed}')

#%%
#generate images (implement line that clears gifdump?)

#plot every n'th vector
f = 80
n = 2

#axis lengths
ax = len(mask[:,0,0,0])
ay = len(mask[0,:,0,0])
X, Y = np.meshgrid(np.arange(0, ax, 1), np.arange(0, ay, 1))

# center of mass at t=0
cy_0, cx_0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, 0]))

for t in range(T):
    
    frame1 = M[:, :, 0, t] #photon density at time t
    mask_t = mask[:, :, 0, t]
    
    fig = plt.subplots(figsize=(10,10))
    ax = plt.gca()
    
    
    plt.imshow(frame1/np.max(frame1), origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
    #plt.imshow(mask_t, origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
    plt.colorbar()
    
    #find center of mass of filled mask (middle of the heart)
    cy, cx = ndi.center_of_mass(ndi.binary_fill_holes(mask_t))
    plt.scatter(cx, cy, marker = 'x', c = 'w')
    
    plt.title(f'Velocity plot over proton density at timepoint t = {t} ({file})', fontsize = 15)
    
    
    #loops over y first
    
    '''
    # just for troubleshooting
    for x in range(0, f, n):
        for y in range(0, f, n):
            if mask_t[y, x] == 1: 
                plt.scatter(x, y, color = 'green')
    '''
    
    
    
    #2D vector visualization 128x128 res image in xy plane
    
    #certainty matrix
    C = frame1/np.max(frame1)
    
    #wiener noise reduction filter (?)
    vx = gaussian_filter(V[:, :, 0, t, 1]*C, sigma = 2)*mask_t #x components of velocity w mask
    vy = gaussian_filter(V[:, :, 0, t, 0]*C, sigma = 2)*mask_t #y components (negative?)
    
    # remove '#' to display divergence of V field; 
    # tells us about unit volume growth per unit volume per unit time (basically strain rate?)
    #vx = divergence(vx)
    #vy = divergence(vy)
    
    q = ax.quiver(X[::n, ::n], Y[::n, ::n], vx[::n, ::n], vy[::n, ::n], 
                  color = 'w', scale = 10, minshaft = 1, minlength=0, width = 0.005)
    z = 25
    plt.xlim(cx_0-z, cx_0+z); plt.ylim(cy_0-z, cy_0+z)
    #plt.xlim(0, f-1); plt.ylim(0, f-1)
    
    plt.savefig(f'R:\Lasse\plots\Vdump\V(t={t}).PNG')
    
    
    plt.show()
    
#%%
#generate GIF from image sequence (https://github.com/adenarayana/Python-Matplotlib/blob/main/009_gifAnimation.ipynb)

filenames = [f'R:\Lasse\plots\Vdump\V(t={t}).PNG' for t in range(T)]

with imageio.get_writer(f'R:\Lasse\plots\MP4\{file}\Velocity.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
    