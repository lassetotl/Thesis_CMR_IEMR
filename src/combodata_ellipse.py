"""
Created on Tue Aug 29 10:46:34 2023

@author: lassetot
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
from lasse_functions import D_ij, D_ij_2D, theta_rad, running_average, clockwise_angle
from lasse_functions import gaussian_2d
#import pandas as pd
#import seaborn as sns
#import sklearn

import scipy.io as sio
import scipy.ndimage as ndi 
from scipy.signal import wiener, convolve2d
import scipy.interpolate as scint
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
file = 'sham_D4-4_6w'
dict = sio.loadmat(f'R:\Lasse\combodata_shax\{file}.mat')
data = dict["ComboData_thisonly"]

#print(f'Keys in dictionary: {dict.keys()}') #dict_keys(['StudyData', 'StudyParam'])
#print(f'Combodata shape: {np.shape(data)}')

#%%

#velocity V field, and magnitudes

# shape of V is (1, 1), indexing necessary to 'unpack' the correct format (?)
V = data['V'][0,0] #velocity field
M = data['Magn'][0,0] #magnitudes
mask = data['Mask'][0,0] #mask for non-heart tissue

T = len(V[0,0,0,:,0]) #Total amount of time steps
T_es = data['TimePointEndSystole'][0,0][0][0]
T_ed = data['TimePointEndDiastole'][0,0][0][0]

print(f'{file} overview:')
print(f'Velocity field shape: {np.shape(V)}')
print(f'Magnitudes field shape: {np.shape(M)}')
print(f'Mask shape: {np.shape(mask)}')

print(f'End systole at t={T_es}, end diastole at t={T_ed}')

#%%
# visualizing Strain Rate
# 'fov'
f = 100

# plot every n'th ellipse
n = 2

# cyclic colormap, normalize to radians 
c_cmap = plt.get_cmap('plasma')

# you can get color palettes from seaborn like this
#c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 8).as_hex())
norm_ = mpl.colors.Normalize(vmin = 0, vmax = 90)

#%%
frame_ = np.zeros((128, 128))
g1 = np.zeros(T); g1[:] = np.nan #Graph values
g2 = np.zeros(T); g2[:] = np.nan #Graph values

sub = 1 # Graph subplot on (1) or off (0)
for t in range(T):
    fig = plt.figure(figsize=(18, 8))
    ax = plt.subplot(1, 2, 1) #SR colormap
    ax = plt.gca()
    #ax.set_facecolor('b')
    
    # combodata mask 
    mask_t = mask[:, :, 0, t] #mask at this timepoint
    
    # erode mask
    mask_t = ndi.binary_erosion(mask_t).astype(mask_t.dtype)
    
    #find center of mass of filled mask (middle of the heart)
    cx, cy = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, t]))
    
    # generate strain rate tensor
    #D = D_ij(V=V, M=M[:f, :f, 0, t], t=t, f=f, mask_ = 1)
    
    #plot magnitude M plot, normalize for certainty values
    M_norm = M[:, :, 0, t]/np.max(M[:, :, 0, t])
    plt.imshow(M_norm.T, origin = 'lower', cmap = 'gray', alpha = 1)
    
    # generate strain rate tensor (old method)
    #D = D_ij(V=V, M=M_norm, t=t, f=f, mask_ = 1)
    
    # arrays for this timepoint t / this image
    eigvals = np.zeros((f,f), dtype = object)
    eigvals_m = np.zeros((f,f), dtype = object) #highest values
    eigvecs = np.zeros((f,f), dtype = object)
    
    # reset radial and circumferential contributions from last frame / initialize
    rad_e = 0 #radial components of eigenvectors, sum will be saved every t
    circ_e = 0 #circumferential ...
    
    # generate 2d gaussian kernel for data smoothing
    g = gaussian_2d(sigma = 2)
    
    #calculate eigenvalues and vectors
    for x in range(0, f, n):
        for y in range(0, f, n):
            # mask choice
            if mask_t[x, y] == 1: #why does this work? (Switching x and y)
                #SR tensor for specific point xy
                #D_ = np.array([[D[0,0][x,y], D[1,0][x,y]], [D[0,1][x,y], D[1,1][x,y]]])
                D_ = D_ij_2D(x, y, V, M_norm, t, g)
            
                val, vec = np.linalg.eig(D_)
                #print(np.sign(val[0]) != np.sign(val[1]))
                 
                eigvals[x, y] = val 
                eigvecs[x, y] = vec
                eigvals_m[x, y] = val[np.argmax(abs(val))]
                #print(eigvals_m[x,y] > np.pi)
                
                #the highest eigenvalue is saved
                #frame_[x,y] = val[np.argmax(abs(val))] #stretch/shortening
            
    #create ellipses, normalized eigenvals at mean
    #eigvals = eigvals/abs(np.max((eigvals_m))) #scale with max eigval this frame
    
    for x in range(0, f, n):
        for y in range(0, f, n): 
            if mask_t[x, y] == 1: #why does this work? (Switching x and y)
                    
                # vector between center of mass and point (x, y) 
                r = np.array([x - cx, y - cy])
                #plt.quiver(cx, cy, r[0], r[1], scale = 50, width = 0.001)
                
                vec = eigvecs[x, y]
                val = eigvals[x, y]
                
                # index of highest/lowest (absolute) eigenvalue
                val_max_i = np.argmax(abs(val))
                val_min_i = np.argmin(abs(val))
                
                # color code in hex notation, from c value
                # I = val[0]**2 + val[1]**2  # invariant
                
                theta = clockwise_angle(r, vec[val_max_i])  # angle between highest eigenvector and r
                theta_ = clockwise_angle(r, vec[val_min_i]) # angle between lowest eigenvector and r
                
                # for color coding; smallest angle between long-axis eigenvector (tends to correspond to positive eigenvalue) and r
                # max value is now 90 degrees (pi/2) relative to radial unit vector
                theta_r = theta_rad(r, vec[np.argmax(val)])
                if theta_r > np.pi/2:
                    theta_r = np.pi - theta_r
                
                # radial/circumferential contributions from each eigenvector
                # (abs of cos and sin factors)
                rad_e += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                circ_e += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                
                #print(r, vec[val_i], c)
                
                #hex code, inputs in range (0, 1) so theta is scaled
                hx = mpl.colors.rgb2hex(c_cmap(theta_r/(np.pi/2)))
                #print(c_cmap(c), hx)
                
                # angle between eigenvector and x-axis, converted to degrees anti-clockwise
                e_angle = -clockwise_angle([1,0], vec[val_max_i])*180/np.pi
                
                # draw ellipses that are spanned by eigenvectors
                # eigenvalues are transformed (1 + tanh(val)) to have a circular unit ellipse
                ellipse = patches.Ellipse((x, y), (1 + np.tanh(val[val_max_i])), (1 + np.tanh(val[val_min_i])), 
                                          angle = e_angle, color = hx)
                
                #unit ellipse
                #unit_ellipse = patches.Ellipse((x, y), 1, 1, color = 'k'); ax.add_artist(unit_ellipse)
                
                ax.add_artist(ellipse)
                #plt.quiver(x, y, vec[val_max_i][0], vec[val_max_i][1], color = hx, scale = 10/np.sqrt(val[val_max_i]))
                #plt.quiver(x, y, vec[val_min_i][0], vec[val_min_i][1], color = 'k', scale = 10/np.sqrt(val[val_min_i]))
    #I[t] = Invariant(I_[0], I_[1], D)
    
    #frame_e = frame_*mask_e
    
    # graph subplot values
    g1[t] = rad_e #global radial strain rate this frame
    g2[t] = circ_e #global circumferential strain rate

    plt.scatter(cx, cy, marker = 'x', c = 'w')
 
    plt.title(f'Strain Rate at t = {t} ({file})', fontsize = 15)
    plt.xlim(0, f-1); plt.ylim(0, f-1)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    sm = plt.cm.ScalarMappable(cmap = c_cmap, norm = norm_)
    cbar = plt.colorbar(sm, cax = cax)
    cbar.set_label('$\Theta$ (degrees)', fontsize = 15)
    
    if sub == 1:  # subplot graph
        plt.subplot(1, 2, 2) #TSR
        #plt.title('Invariant $\lambda_1^2 + \lambda_2^2$ in marked position', fontsize = 15)
        plt.title('Global Strain Rate over time')
        
        plt.axhline(0, c = 'k', lw = 1)
        plt.plot(np.arange(0, T, 1), g1, 'darkblue', label = 'Radial')
        plt.plot(np.arange(0, T, 1), g2, 'chocolate', label = 'Circumferential')
        
        plt.axvline(T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
        plt.axvline(T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
        
        
        plt.xlim(0, T)#; plt.ylim(0, 50)
        plt.xlabel('Timepoints', fontsize = 15)
        plt.ylabel('$s^{-1}$', fontsize = 20)
        
        plt.legend()
    
    plt.savefig(f'R:\Lasse\plots\SRdump\SR(t={t}).PNG')
    plt.show()

#%%
#last frame with running average

N = 4 #window
I_g1 = running_average(g1, 4)
I_g2 = running_average(g2, 4)

plt.figure(figsize=(10, 8))

plt.title(f'Global Strain rate over time ({file})', fontsize = 15)
plt.axvline(T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.axhline(0, c = 'k', lw = 1)

plt.xlim(0, T)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('$s^{-1}$', fontsize = 20)
plt.legend()

plt.plot(np.arange(0, T, 1), g1, 'lightgrey')
plt.plot(np.arange(0, T, 1), g2, 'lightgrey')

plt.plot(np.arange(0, T, 1), I_g1, 'darkblue', lw=2, label = 'Radial (Walking Average)') #walking average
plt.plot(np.arange(0, T, 1), I_g2, 'chocolate', lw=2, label = 'Circumferential (Walking Average)') #walking average

plt.legend()

plt.subplots_adjust(wspace=0.25)
plt.savefig(f'R:\Lasse\plots\{file}_GSR.PNG')
plt.show()

#%%
#Generate mp4

filenames = [f'R:\Lasse\plots\SRdump\SR(t={t}).PNG' for t in range(T)]

with imageio.get_writer(f'R:\Lasse\plots\MP4\{file}\Ellipses.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
