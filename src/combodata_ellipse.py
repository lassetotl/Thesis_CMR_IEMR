"""
Created on Tue Aug 29 10:46:34 2023

@author: lassetot

Note (Sep 20): a lot of the current code is included for troubleshooting purposes, 
not all are strictly useful at the moment and will be trimmed eventually
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
from lasse_functions import D_ij, D_ij_2D, theta_rad, running_average, clockwise_angle
from lasse_functions import gaussian_2d, theta_extreme
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
res = data['Resolution'][0,0][0][0]  # voxel length in mm?

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
n = 3
sigma = 2

# cyclic colormap, normalize to radians 
c_cmap = plt.get_cmap('plasma')

# you can get color palettes from seaborn like this
#c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 8).as_hex())
norm_ = mpl.colors.Normalize(vmin = 0, vmax = 90)

#%%

range_ = np.arange(0, T, 1)
g1 = np.zeros(T); g1[:] = np.nan #Graph values
g2 = np.zeros(T); g2[:] = np.nan #Graph values

# temporary, separates positive and negative contributions over time
r1_ = np.zeros(T); r1_[:] = np.nan #Graph values, positive
r2_ = np.zeros(T); r2_[:] = np.nan #Graph values, negative
c1_ = np.zeros(T); c1_[:] = np.nan #Graph values
c2_ = np.zeros(T); c2_[:] = np.nan #Graph values

a1 = np.zeros(T, dtype = 'object') #  most positive angle (stretch)
a1_std = np.zeros(T) # std stored for each t
a2 = np.zeros(T, dtype = 'object') # most negative angle (compression)
a2_std = np.zeros(T)

# center of mass at t=0
cx_0, cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, 0]))

sub = 1 # Graph subplot on (1) or off (0)
for t in range(T):
    fig = plt.figure(figsize=(18, 8))
    ax = plt.subplot(1, 2, 1) #SR colormap
    ax = plt.gca()
    ax.text(3, 3, f'Gaussian smoothing ($\sigma = {sigma}$)', color = 'w')
    #ax.set_facecolor('b')
    
    # combodata mask 
    mask_t = mask[:, :, 0, t] #mask at this timepoint
    
    # erode mask
    #mask_t = ndi.binary_erosion(mask_t).astype(mask_t.dtype)
    
    #find center of mass of filled mask (middle of the heart)
    cx, cy = ndi.center_of_mass(ndi.binary_fill_holes(mask_t))
    
    # generate strain rate tensor
    #D = D_ij(V=V, M=M[:f, :f, 0, t], t=t, f=f, mask_ = 1)
    
    # plot magnitude M plot, normalize for certainty values
    # transpose to allign with mask
    M_norm = (M[:, :, 0, t]/np.max(M[:, :, 0, t])).T
    plt.imshow(M_norm, origin = 'lower', cmap = 'gray', alpha = 1)
    #plt.imshow(mask_t.T, origin = 'lower', cmap = 'gray', alpha = 1)
    
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
    g = gaussian_2d(sigma = sigma)
    
    #calculate eigenvalues and vectors
    e_count = 0  # ellipse counter in this frame
    for x in range(0, f, n):
        for y in range(0, f, n):
            # mask choice
            if mask_t[x, y] == 1: #why does this work? (Switching x and y)
                # SR tensor for specific point xy
                #D_ = np.array([[D[0,0][x,y], D[1,0][x,y]], [D[0,1][x,y], D[1,1][x,y]]])
                D_ = D_ij_2D(x, y, V, M_norm, t, g)
            
                val, vec = np.linalg.eig(D_)
                #print(np.sign(val[0]) != np.sign(val[1]))
                 
                eigvals[x, y] = val 
                eigvecs[x, y] = vec
                eigvals_m[x, y] = val[np.argmax(abs(val))]
                # print(eigvals_m[x,y] > np.pi)
                
                # the highest eigenvalue is saved
                #frame_[x,y] = val[np.argmax(abs(val))] #stretch/shortening
                e_count += 1
            
    # create ellipses, normalized eigenvals at mean
    #eigvals = eigvals/abs(np.max((eigvals_m))) # scale with max eigval this frame
    
    r1_[t] = 0; r2_[t] = 0  # remove nan values for this t
    c1_[t] = 0; c2_[t] = 0
    
    a1_ = []; a2_ = []
    
    for x in range(0, f, n):
        for y in range(0, f, n): 
            if mask_t[x, y] == 1: #why does this work? (Switching x and y)
                    
                # vector between center of mass and point (x, y) 
                r = np.array([x - cx, y - cy])
                #plt.quiver(cx, cy, r[0], r[1], scale = 50, width = 0.001)
                
                vec = eigvecs[x, y]
                val = eigvals[x, y]
                
                # index of eigenvalues
                val_max_i = np.argmax(val)  # most positive value
                val_min_i = np.argmin(val)  # most negative
                
                # color code in hex notation, from c value
                # I = val[0]**2 + val[1]**2  # invariant
                
                theta = theta_rad(r, vec[val_max_i])  # angle between highest eigenvector and r
                theta_ = theta_rad(r, vec[val_min_i]) # angle between lowest eigenvector and r
                #print(theta, theta_)
                
                # for color coding; smallest angle between long-axis eigenvector (tends to correspond to positive eigenvalue) and r
                # max value is now 90 degrees (pi/2) relative to radial unit vector
                #theta_r = theta_rad(r, vec[np.argmax(val)])
                
                # radial/circumferential contributions from each eigenvector
                # scaled with amount of ellipses, varies because of dynamic mask
            
                r1 = (val[val_max_i])*abs(np.cos(theta))/e_count
                r2 = (val[val_min_i])*abs(np.cos(theta_))/e_count
                
                c1 = (val[val_max_i])*abs(np.sin(theta))/e_count
                c2 = (val[val_min_i])*abs(np.sin(theta_))/e_count
                
                # filtering positive and negative values
                r1_[t] += r1*int(r1 > 0) + r2*int(r2 > 0) # +
                r2_[t] += r1*int(r1 < 0) + r2*int(r2 < 0) # -
                c1_[t] += c1*int(c1 > 0) + c2*int(c2 > 0)
                c2_[t] += c1*int(c1 < 0) + c2*int(c2 < 0)
                
                # global contribution
                rad_e += r1 + r2
                circ_e += c1 + c2
                
                # angle sum collected, scaled to get average angle each t
                # does not assume that each 2d tensor has a positive and negative eigenvector
                if val[val_max_i] > 0:
                    a1_.append(theta) 
                if val[val_min_i] > 0:
                    a1_.append(theta_)
                if val[val_max_i] < 0:
                    a2_.append(theta) 
                if val[val_min_i] < 0:
                    a2_.append(theta_)
                    
                a2_.append(theta*int(val[val_max_i] < 0) + theta_*int(val[val_min_i] < 0))
                
                #print(r, vec[val_i], c)
                
                # hex code, inputs in range (0, 1) so theta is scaled
                hx = mpl.colors.rgb2hex(c_cmap(theta/(np.pi/2)))
                #print(c_cmap(c), hx)
                
                # angle between eigenvector and x-axis, converted to degrees anti-clockwise
                
                e_angle = -(clockwise_angle([1,0], r) + theta)*180/np.pi
                
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
    
    ax.text(3, 6, f'{e_count} Ellipses', color = 'w')
    res_ = round(f*res, 4)
    ax.text(3, 9, f'{res_} x {res_} mm (?)', color = 'w')
    
    # graph subplot values
    g1[t] = rad_e #global radial strain rate this frame
    g2[t] = circ_e #global circumferential strain rate
    
    # collect average angle in degrees
    a1[t] = np.array(a1_)*180/np.pi #np.mean(a1_)*180/np.pi 
    a2[t] = np.array(a2_)*180/np.pi #np.mean(a2_)*180/np.pi 
    a1_std[t] = (np.mean(a1_))*180/np.pi
    a2_std[t] = (np.mean(a2_))*180/np.pi

    plt.scatter(cx, cy, marker = 'x', c = 'w')
 
    plt.title(f'Strain Rate at t = {t} ({file})', fontsize = 15)
    
    #z = 25
    #plt.xlim(cx_0-z, cx_0+z); plt.ylim(cy_0-z, cy_0+z)
    plt.xlim(0, f); plt.ylim(0, f)
    
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
        plt.axvline(T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
        plt.axvline(T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
        
        plt.plot(range_, r1_, 'darkblue', label = 'Radial+')
        plt.plot(range_, c1_, 'chocolate', label = 'Circumferential+')
        
        
        plt.plot(range_, r2_, 'blue', label = 'Radial-')
        plt.plot(range_, c2_, 'orange', label = 'Circumferential-')
        
        plt.plot(range_, r1_ + r2_, 'blue', ls = '--', alpha = 0.2)
        plt.plot(range_, c1_ + c2_, 'orange', ls = '--', alpha = 0.2)
        
        plt.xlim(0, T)#; plt.ylim(0, 50)
        plt.xlabel('Timepoints', fontsize = 15)
        plt.ylabel('$s^{-1}$', fontsize = 20)
        
        plt.legend()
    
    plt.tight_layout()
    #plt.autoscale()
    plt.savefig(f'R:\Lasse\plots\SRdump\SR(t={t}).PNG')
    plt.show()

#%%
#angles over time

plt.figure(figsize = (10, 8))
plt.title(f'Average radial angles over time ({file})', fontsize = 15)
plt.axvline(T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.xlim(0, T)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('Degrees', fontsize = 20)

for i in range_:
    plt.scatter([i]*len(a1[i]), a1[i], color = 'r', alpha = 0.008)
    plt.scatter([i]*len(a2[i]), a2[i], color = 'g', alpha = 0.008)

# variance 
#plt.fill_between(range_, a1-a1_std, a1+a1_std, facecolor = 'r', alpha = 0.08, label = 'Variance')
#plt.fill_between(range_, a2-a2_std, a2+a2_std, facecolor = 'g', alpha = 0.08, label = 'Variance')

# mean angles
plt.plot(range_, a1_std, 'r', label = 'Positive eigenvectors (stretch)')
plt.plot(range_, a2_std, 'g', label = 'Negative eigenvectors (compression)')
plt.text(1, 2, f'Mean std: {round((np.mean(a1_std) + np.mean(a2_std))/2, 4)} degrees')
plt.legend(loc = 'upper right')
plt.show()

#%%
#last frame with running average

N = 4 #window
I_g1 = running_average(g1, N)
I_g2 = running_average(g2, N)

plt.figure(figsize=(10, 8))

plt.title(f'Global Strain rate over time ({file})', fontsize = 15)
plt.axvline(T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.axhline(0, c = 'k', lw = 1)

plt.xlim(0, T)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('$s^{-1}$', fontsize = 20)

plt.plot(range_, g1, 'lightgrey')
plt.plot(range_, g2, 'lightgrey')

plt.plot(range_, I_g1, 'darkblue', lw=2, label = 'Radial (Walking Average)') #walking average
plt.plot(range_, I_g2, 'chocolate', lw=2, label = 'Circumferential (Walking Average)') #walking average

plt.legend()

plt.subplots_adjust(wspace=0.25)
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GSR.PNG')
plt.show()

#%%
#Generate mp4

filenames = [f'R:\Lasse\plots\SRdump\SR(t={t}).PNG' for t in range(T)]

with imageio.get_writer(f'R:\Lasse\plots\MP4\{file}\Ellipses.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
