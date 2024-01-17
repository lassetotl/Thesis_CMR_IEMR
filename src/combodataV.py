#%%
#author Lasse Totland @ Ullevål OUS
#analysis of MR combodatasets
#(streamlining: all of this can be implemented as a more user friendly 
#function/class eventually (methods for V and SR plot?))


#imports

import numpy as np
import matplotlib.pyplot as plt
from util import divergence, theta_rad, clockwise_angle
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
file = 'sham_D11-1_1d'
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
f = 100
n = 2

#axis lengths
ax = len(mask[:,0,0,0])
ay = len(mask[0,:,0,0])

# global rad and circ velocity
gr = np.zeros(T)
gc = np.zeros(T)

# center of mass at t=0
cx_0, cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, 0]))

range_ = range(T)
for t in range_:
    
    frame1 = M[:, :, 0, t] #photon density at time t
    mask_t = mask[:, :, 0, t]
    #mask_t = ndi.binary_erosion(mask_t).astype(mask_t.dtype)
    
    plt.subplots(figsize=(10,10))
    #ax = plt.gca()
    
    
    plt.imshow(frame1.T/np.max(frame1), origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
    #plt.imshow(mask_t.T, origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
    #plt.colorbar()
    
    #find center of mass of filled mask (middle of the heart)
    cx, cy = ndi.center_of_mass(ndi.binary_fill_holes(mask_t))
    
    plt.title(f'Velocity plot over proton density at timepoint t = {t} ({file})', fontsize = 15)
    
    
    #certainty matrix
    C = frame1/np.max(frame1)
    
    #wiener noise reduction filter (?)
    vx = ndi.gaussian_filter(V[:, :, 0, t, 0]*C, sigma = 2)*mask_t #x components of velocity w mask
    vy = ndi.gaussian_filter(V[:, :, 0, t, 1]*C, sigma = 2)*mask_t #y components (negative?)
    #vz = ndi.gaussian_filter(V[:, :, 0, t, 2]*C, sigma = 2)*mask_t #y components (negative?)
    
    #vx = sig.convolve2d(V[:, :, 0, t, 1]*C, g) / sig.convolve2d(C, g)
    #vy = sig.convolve2d(V[:, :, 0, t, 0]*C, g) / sig.convolve2d(C, g)
    
    X = np.arange(128); Y = np.arange(128)
    X, Y = np.meshgrid(X, Y)
    
    plt.quiver(X[::n, ::n], Y[::n, ::n], vx[::n, ::n].T, vy[::n, ::n].T, 
                  color = 'w', scale = 10, minshaft = 1, minlength=0, width = 0.004)
    '''
    # just for troubleshooting
    for x in range(0, f, n):
        for y in range(0, f, n):
            
            if mask_t[x, y] == 1:
                
                r = np.array([x - cx, y - cy])
                #plt.quiver(cx, cy, r[0], r[1], scale = 50, width = 0.001)
                
                v_ = np.array([vx[x, y], vy[x, y]])
                plt.quiver(x, y, v_[0], v_[1], color = 'w', scale = 10, minshaft = 1, minlength = 0, width = 0.004)
                theta = clockwise_angle(r, v_) + np.pi
                
                gr[t] += np.linalg.norm(v_)*np.cos(theta) 
                gc[t] += np.linalg.norm(v_)*np.sin(theta) 
    '''
    
    #plt.scatter(cx, cy, marker = 'x', c = 'w', s = 210, linewidths = 2)
    
    
    #2D vector visualization 128x128 res image in xy plane
    
    # remove '#' to display divergence of V field; 
    # tells us about unit volume growth per unit volume per unit time (basically strain rate?)
    #vx = divergence(vx)
    #vy = divergence(vy)
    
    #q = ax.quiver(X[::n, ::n], Y[::n, ::n], vx[::n, ::n], vy[::n, ::n], 
    #              color = 'w', scale = 10, minshaft = 1, minlength=0, width = 0.005)
    z = 25
    plt.xlim(cx_0-z, cx_0+z); plt.ylim(cy_0-z, cy_0+z)
    #plt.xlim(0, f-1); plt.ylim(0, f-1)
    
    plt.savefig(f'R:\Lasse\plots\Vdump\V(t={t}).PNG')
    
    
    plt.show()
    
#%%
plt.figure(figsize=(10, 8))
plt.title(f'Global velocity over time ({file})', fontsize = 15)
plt.axvline(T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.axhline(0, c = 'k', lw = 1)

plt.plot(range_, gr, lw = 2, label = 'radial')

#is this actually 'circumferential velocity'?
#plt.plot(range_, gc, label = 'circ')
plt.legend()
#%%
#generate GIF from image sequence (https://github.com/adenarayana/Python-Matplotlib/blob/main/009_gifAnimation.ipynb)

filenames = [f'R:\Lasse\plots\Vdump\V(t={t}).PNG' for t in range_]

with imageio.get_writer(f'R:\Lasse\plots\MP4\{file}\Velocity.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
    