"""
Created on Tue Aug 29 10:46:34 2023

@author: lassetot
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
from lasse_functions import D_ij, theta, running_average
#import pandas as pd
#import seaborn as sns; sns.set()
#import sklearn

import scipy.io as sio
import scipy.ndimage as ndi 
from scipy.signal import wiener
import scipy.interpolate as scint
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
file = 'Combodata'
dict = sio.loadmat(f'R:\Lasse\combodata\{file}.mat')
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
#visualizing Strain Rate

f = 100

#plot every n'th ellipse
n = 2

#cyclic colormap

c_cmap = plt.get_cmap('plasma')
norm_ = mpl.colors.Normalize(vmin = 0, vmax = 2*np.pi)

#%%
frame_ = np.zeros((f,f))
g1 = np.zeros(T); g1[:] = np.nan #Graph values
g2 = np.zeros(T); g2[:] = np.nan #Graph values

sub = 1  # Graph subplot on (1) or off (0)
for t in range(T):
    fig = plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1) #SR colormap
    ax = plt.gca()
    #ax.set_facecolor('b')
    
    #diluted mask to calculate D_ij
    mask_t = mask[:f, :f, 0, t] #mask at this timepoint
    mask_d = ndi.binary_dilation(mask_t).astype(mask_t.dtype) - mask_t
    mask_e = ndi.binary_erosion(mask_t).astype(mask_t.dtype)
    
    #find center of mass of filled mask (middle of the heart)
    cy, cx = ndi.center_of_mass(ndi.binary_fill_holes(mask[:f, :f, 0, t]))
    
    D = D_ij(V=V, t=t, f=f, mask_ = 1)
    
    #plot ellipse every third index
    plt.imshow(M[:f, :f, 0, t], origin = 'lower', cmap = 'gray', alpha = 1)
    
    # arrays for this timepoint t / this image
    eigvals = np.zeros((f,f), dtype = object)
    eigvals_m = np.zeros((f,f), dtype = object) #highest values
    eigvecs = np.zeros((f,f), dtype = object)
    rad_e = 0 #radial components of eigenvectors, sum will be saved every t
    circ_e = 0 #circumferential ...
    
    #calculate eigenvalues and vectors
    for x in range(0, f, n):
        for y in range(0, f, n):
            #SR tensor for specific point xy
            D_ = np.array([[D[0,0][x,y], D[1,0][x,y]], [D[0,1][x,y], D[1,1][x,y]]])
            val, vec = np.linalg.eig(D_)
            eigvals[x, y] = val 
            eigvecs[x, y] = vec
            eigvals_m[x, y] = val[np.argmax(abs(val))]
            
            #the highest eigenvalue is saved
            #frame_[x,y] = val[np.argmax(abs(val))] #stretch/shortening
            
    #create ellipses, normalized eigenvals at mean
    eigvals = eigvals/abs(np.max((eigvals_m))) #scale with max eigval this frame
    for x in range(0, f, n):
        for y in range(0, f, n):       
            #mask choice here
            if mask_e[y, x] == 1: #why does this work? (Switching x and y)
                    
                # vector between center of mass and point (x, y) 
                r = np.array([x - cx, y - cy])
                plt.quiver(cx, cy, r[0], r[1], scale = 50, width = 0.001)
                
                vec = eigvecs[x, y]
                val = eigvals[x, y]
                
                # index of highest/lowest (absolute) eigenvalue
                val_max_i = np.argmax(abs(val))
                val_min_i = np.argmin(abs(val))
                
                # color code in hex notation, from c value
                # c = val[0]**2 + val[1]**2  # invariant
                c = theta(r, vec[val_max_i])  # angle between highest eigenvector and r
                c_ = theta(r, vec[val_min_i]) # angle between lowest eigenvector and r
                
                # radial/circumferential contributions from each eigenvector
                # eigenvectors contain directional info, thus abs() of eigenvalues
                rad_e += abs(val[val_max_i])*np.cos(c) + abs(val[val_min_i])*np.cos(c_)
                circ_e += abs(val[val_max_i])*np.sin(c) + abs(val[val_min_i])*np.sin(c_)
                
                #print(r, vec[val_i], c)
                
                #hex code, inputs in range (0, 1) so c is scaled
                hx = mpl.colors.rgb2hex(c_cmap(c/(2*np.pi)))
                #print(c_cmap(c), hx)
                
                #directional information lost from eigenvalues
                val = abs(val)
                
                #https://stackoverflow.com/questions/67718828/how-can-i-plot-an-ellipse-from-eigenvalues-and-eigenvectors-in-python-matplotl
                theta_ = np.linspace(0, 2*np.pi, 1000);
                #the first factor is arbitrary scaling
                ellipsis = 1*(np.sqrt(val[None,:]) * vec) @ [np.sin(theta_), np.cos(theta_)]
                
                plt.fill(x + ellipsis[0,:], y + ellipsis[1,:], color = hx)
                #plt.quiver(x, y, vec[0][0], vec[0][1], color = 'red', scale = 10/np.sqrt(val[0]))
                #plt.quiver(x, y, vec[1][0], vec[1][1], color = 'red', scale = 10/np.sqrt(val[1]))
    #I[t] = Invariant(I_[0], I_[1], D)
    
    #frame_e = frame_*mask_e
    
    # graph subplot values
    g1[t] = rad_e #global radial strain rate this frame
    g2[t] = circ_e #global circumferential strain rate

    plt.scatter(cx, cy, marker = 'x', c = 'w')
 
    plt.title(f'Strain Rate at t = {t}', fontsize = 15)
    plt.xlim(0, f-1); plt.ylim(0, f-1)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    sm = plt.cm.ScalarMappable(cmap = c_cmap, norm = norm_)
    cbar = plt.colorbar(sm, cax = cax)
    cbar.set_label('$\Theta$ (radians)', fontsize = 15)
    
    if sub == 1:  # subplot graph
        plt.subplot(1, 2, 2) #TSR
        #plt.title('Invariant $\lambda_1^2 + \lambda_2^2$ in marked position', fontsize = 15)
        plt.title('Global Strain Rate over time')
        plt.grid(1)
        plt.axhline(0, c = 'k', lw = 1)
        plt.plot(np.arange(0, T, 1), g1, 'darkblue', label = 'Radial')
        plt.plot(np.arange(0, T, 1), g2, 'm', label = 'Circumferential')
        
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

plt.figure(figsize=(12, 8))

#plt.subplot(1, 2, 2) #TSR
#plt.title('Invariant $\lambda_1^2 + \lambda_2^2$ in marked position', fontsize = 15)
plt.title('Global Strain rate over time')
plt.grid(1)
plt.axvline(T_es, c = 'm', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed, c = 'm', ls = '--', lw = 1.5, label = 'End Diastole')
plt.xlim(0, T)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('$s^{-1}$', fontsize = 20)
plt.legend()

plt.plot(np.arange(0, T, 1), g1, 'lightgrey')
plt.plot(np.arange(0, T, 1), g2, 'lightgrey')

plt.plot(np.arange(0, T, 1), I_g1, 'darkblue', lw=2, label = 'Radial (Walking Average)') #walking average
plt.plot(np.arange(0, T, 1), I_g2, 'm', lw=2, label = 'Circumferential (Walking Average)') #walking average

plt.legend()

plt.subplots_adjust(wspace=0.25)
plt.savefig(f'R:\Lasse\plots\SHAM_GRSR.PNG')
plt.show()

#%%
#Generate mp4

filenames = [f'R:\Lasse\plots\SRdump\SR(t={t}).PNG' for t in range(T)]

with imageio.get_writer(f'R:\Lasse\plots\MP4\{file}\Ellipses.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
